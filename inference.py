import sys
import os
import collections
from pathlib import Path
from typing import cast
import numpy as np
import pyaudio
from model.quartznet_model_evo import QuartzNet_evo
import re
import torch
import torchaudio.transforms as T
from pyctcdecode.decoder import build_ctcdecoder
from model.model import ASRResNet_Medium
from model.quartznet_model_evo_heavier import  QuartzNet_evo_2
import time
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))



# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK = 512  # ~32ms a 16kHz
FORMAT = pyaudio.paFloat32
CHANNELS = 1

PROJECT_ROOT = SCRIPT_DIR.parent
LM_PATH = os.getenv("ASR_LM_PATH", str(PROJECT_ROOT / "ASR/lm_5gram.bin"))
MODEL_PTH_PATH = os.getenv("ASR_MODEL_PATH", str(PROJECT_ROOT / "ASR/pth/quartznet_evo_attn_13M.pth"))

asr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad_device = torch.device("cpu")
class ItaTokenizer:
    def __init__(self):
        self.blank_token = "<BLANK>"
        self.unk_token = "<UNK>"
        
        self.vocab = [self.blank_token, self.unk_token, " "] + list("abcdefghijklmnopqrstuvwxyzàèéìòù'")
        
        self.blank_id = 0
        self.unk_id = 1
        self.space_id = 2
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def clean_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-zàèéìòù'\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def encode(self, text: str) -> list[int]:
        cleaned = self.clean_text(text)
        return [self.char_to_idx.get(char, self.unk_id) for char in cleaned]

    def decode(self, indices: list[int]) -> str:
        chars = []
        for idx in indices:
            if idx in (self.blank_id, self.unk_id):
                continue
            chars.append(self.idx_to_char.get(idx, ""))
        
        res = "".join(chars)
        return re.sub(r"\s+", " ", res).strip()

    def __len__(self):
        return len(self.vocab)
# --- TOKENIZER AND DECODER KENLM ---
tokenizer = ItaTokenizer()

labels_for_decoder = []
for idx, char in enumerate(tokenizer.vocab):
    if idx == tokenizer.blank_id: 
        labels_for_decoder.append("")  
    elif idx == tokenizer.unk_id:  
        labels_for_decoder.append("<unk>")  
    else:
        labels_for_decoder.append(char)

valid_chars = {
    c
    for c in labels_for_decoder
    if c not in ("", "<unk>", tokenizer.blank_token, tokenizer.unk_token)
}

# loading unigrams
with open("ASR/unigrams_clean.txt", "r", encoding="utf-8") as f:
    raw_unigrams = f.read().splitlines()

valid_unigrams = []
for word in raw_unigrams:
    word = word.strip().lower()

    if word.startswith("<") and word.endswith(">"):
        continue

    word = word.replace("’", "'")

    if set(word).issubset(valid_chars):
        valid_unigrams.append(word)

valid_unigrams = sorted(list(set(valid_unigrams)))


decoder = build_ctcdecoder(
    labels=labels_for_decoder,
    kenlm_model_path=LM_PATH,
    unigrams=valid_unigrams,
    alpha=0.8,
    beta=2,
)

# --- ACUSTIC MODEL ---
print("loading acustic model...")
checkpoint = torch.load(MODEL_PTH_PATH, map_location=asr_device, weights_only=True)
acoustic_model = QuartzNet_evo_2(num_classes=len(tokenizer)).to(asr_device)
acoustic_model.load_state_dict(checkpoint)
acoustic_model.eval()
print(f"acustic model loaded in: {asr_device}")

# --- SILERO VAD ---
print("Loading Silero VAD...")

hub_result = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False,
    source='github'
)
vad_model, utils = cast(tuple[torch.nn.Module, tuple], hub_result)


(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


vad_iterator = VADIterator(
    vad_model, 
    threshold=0.75, 
    sampling_rate=SAMPLE_RATE
)

# --- PREPROCESSING AUDIO LIVE ---
mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=400,
    win_length=400,
    hop_length=160,
    n_mels=80
).to(asr_device)

amp_to_db = T.AmplitudeToDB(top_db=80.0).to(asr_device)

def preprocess_audio(audio_numpy, current_sr=16000):
    tensor_wave = torch.from_numpy(audio_numpy).float()
    

    if tensor_wave.ndim == 1:
        tensor_wave = tensor_wave.unsqueeze(0)  
    elif tensor_wave.ndim == 2:
        tensor_wave = tensor_wave.T 
        
    # mono conversion 
    if tensor_wave.shape[0] > 1:
        tensor_wave = tensor_wave.mean(dim=0, keepdim=True)  
        
    # Resampling at 16kHz
    if current_sr != 16000:
        resampler = T.Resample(orig_freq=current_sr, new_freq=16000)
        tensor_wave = resampler(tensor_wave)
        
    tensor_wave = tensor_wave.to(asr_device)
    
    with torch.no_grad():
        spec = mel_transform(tensor_wave)       
        spec_db = amp_to_db(spec)               
        
    return spec_db.to(torch.float32)
p = pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK
)

# --- BUFFER VAD ---
padding_buffer = collections.deque(
    maxlen=12
)  
POST_PADDING_CHUNKS = 6  

is_speaking = False
audio_buffer = []
post_padding_counter = 0

# --- UNIFT TEXT PARAMETERS ---
PAUSE_THRESHOLD_SEC = 2.5  
last_speech_time = time.time()
full_transcript_session = []

print("\n" + "=" * 50)
print("[READY] listening...")
print("=" * 50 + "\n")

try:
    while True:
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        chunk_np = np.frombuffer(raw_data, dtype=np.float32)

        tensor_chunk = torch.from_numpy(chunk_np.copy()).to(vad_device)
        speech_dict = vad_iterator(tensor_chunk, return_seconds=False)

        current_time = time.time()

        if speech_dict is not None:
            if "start" in speech_dict:
                is_speaking = True
                last_speech_time = current_time  
                audio_buffer = list(padding_buffer) + [chunk_np]
                padding_buffer.clear()

                sys.stdout.write("\r[VAD] talking...               ")
                sys.stdout.flush()

            elif "end" in speech_dict:
                is_speaking = False
                audio_buffer.append(chunk_np)
                post_padding_counter = POST_PADDING_CHUNKS

        else:
            if is_speaking:
                is_speaking = True
                last_speech_time = (
                    current_time  
                )
                audio_buffer.append(chunk_np)

            elif post_padding_counter > 0:
                audio_buffer.append(chunk_np)
                post_padding_counter -= 1

                if post_padding_counter == 0:
                    full_speech = np.concatenate(audio_buffer)

                    if len(full_speech) >= SAMPLE_RATE * 0.4:
                        spec_final = preprocess_audio(full_speech)
                        input_lengths = torch.tensor(
                            [spec_final.shape[-1]],
                            dtype=torch.long,
                            device=asr_device,
                        )

                        with torch.no_grad():
                            logits, output_lengths = acoustic_model(
                                spec_final, input_lengths
                            )

                        valid_frames = int(output_lengths[0].item())

                        if logits.ndim == 3:
                            logits_single = (
                                logits[0, :valid_frames, :]
                                if logits.shape[0] == 1
                                else logits[:valid_frames, 0, :]
                            )
                        else:
                            logits_single = logits[:valid_frames, :]

                        log_probs = (
                            torch.nn.functional.log_softmax(
                                logits_single, dim=-1
                            )
                            .cpu()
                            .numpy()
                        )

                        chunk_text = decoder.decode(
                            log_probs, beam_width=128
                        ).strip()

                        if len(chunk_text) > 1:
                            full_transcript_session.append(chunk_text)
                            last_speech_time = (
                                time.time()
                            )

                            partial_view = " ".join(full_transcript_session)
                            sys.stdout.write(f"\r-> In corso: {partial_view}...")
                            sys.stdout.flush()

                    audio_buffer = []
                    padding_buffer.clear()
                    vad_iterator.reset_states()

            else:
                padding_buffer.append(chunk_np)

        if (
            full_transcript_session
            and not is_speaking
            and (current_time - last_speech_time) > PAUSE_THRESHOLD_SEC
        ):

            final_sentence = " ".join(full_transcript_session)
            final_sentence = (
                final_sentence[0].upper() + final_sentence[1:] + "."
            )

            print(f"\r\n\n[TESTO UNIFICATO]: {final_sentence}\n")
            print("[PRONTO] In ascolto...")

            full_transcript_session = []

except KeyboardInterrupt:
    print("\n[STOP] stop listening.")


finally:
    stream.stop_stream()
    stream.close()
    p.terminate()