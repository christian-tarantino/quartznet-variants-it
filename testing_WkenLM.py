from random import random
from torch.utils.data import DataLoader, Dataset, Sampler
import torch
import torch.nn as nn
import os
import csv
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset
from tqdm import tqdm
import re
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from model.model import ASRResNet_Medium
from model.quartznet_model import QuartzNet
from model.quartznet_model_evo import QuartzNet_evo
from model.quartznet_model_evo_heavier import QuartzNet_evo_2
import torch.nn.functional as F
import time 
import torch.optim as optim
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
from jiwer import wer, cer
from torchinfo import summary
import soundfile as sf
import numpy as np
import torch.amp
import random
from typing import List, Any
from pyctcdecode.decoder import build_ctcdecoder
from multiprocessing import Pool
import json
import gc



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
        """Decodifica una lista di ID in testo filtrando BLANK e UNK."""
        chars = []
        for idx in indices:
            if idx in (self.blank_id, self.unk_id):
                continue
            chars.append(self.idx_to_char.get(idx, ""))
        
        res = "".join(chars)
        return re.sub(r"\s+", " ", res).strip()

    def __len__(self):
        return len(self.vocab)



def collate_fn(batch):
  specs = [item[0].transpose(0, 1) for item in batch]
  targets = [item[1] for item in batch]

  input_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
  target_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)

  # 2. pad_sequence restituisce [Batch, Time, Freq]
  specs_padded = pad_sequence(specs, batch_first=True)

  # 3. Riportiamo a [Batch, Freq, Time] per le convoluzioni della rete
  specs_padded = specs_padded.transpose(1, 2)

  targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

  return specs_padded, targets_padded, input_lengths, target_lengths


class PackedASRDataset(Dataset):

 
    def __init__(self, root_dir, output_name="common_voice_packed", tokenizer=None):
        if tokenizer is None:
            raise ValueError(
                "no tokenizer passed to PackedASRDataset"
            )
 
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.output_name = output_name
 
        bin_path = os.path.join(root_dir, f"{output_name}.bin")
        meta_path = os.path.join(root_dir, f"{output_name}_meta.json")
 
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
 

        self.bin_path = bin_path
        self.mmap_data = None
 
        offsets = []
        num_elements = []
        shapes = []
        spec_lens = []
        text_lens = []
        tokens_list = []  
 
        for item in metadata:
            text_tokens = self.tokenizer.encode(str(item["sentence"]))
            if not text_tokens:
                continue
 
            spec_len = item["shape"][1]
            output_len = (spec_len + 1) // 2
            output_len = (output_len + 1) // 2
            repeated_tokens = sum(
                left == right for left, right in zip(text_tokens, text_tokens[1:])
            )
            minimum_output_len = len(text_tokens) + repeated_tokens
 
            if minimum_output_len > output_len:
                continue
 
            offsets.append(item["offset"] // 4)
            num_elements.append(item["length"] // 4)
            shapes.append(item["shape"])
            spec_lens.append(spec_len)
            text_lens.append(len(text_tokens))
            tokens_list.append(text_tokens)
 
        n = len(offsets)
 
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.num_elements = np.asarray(num_elements, dtype=np.int64)
        self.shapes = np.asarray(shapes, dtype=np.int64)  # (N, ndim_shape)
        self.spec_lens = np.asarray(spec_lens, dtype=np.int64)
        self.text_lens = np.asarray(text_lens, dtype=np.int64)
 

        token_offsets = np.zeros(n + 1, dtype=np.int64)
        for i, toks in enumerate(tokens_list):
            token_offsets[i + 1] = token_offsets[i] + len(toks)
 
        flat_tokens = np.empty(token_offsets[-1], dtype=np.int64)
        for i, toks in enumerate(tokens_list):
            flat_tokens[token_offsets[i]: token_offsets[i + 1]] = toks
 
        self.flat_tokens = flat_tokens
        self.token_offsets = token_offsets
 
        self._n = n
 
        print(f"Dataset caricato: {self._n} campioni validi.")
 
    def __getitem__(self, index):
        if self.mmap_data is None:
            self.mmap_data = np.memmap(self.bin_path, dtype="float32", mode="r")
 
        start = int(self.offsets[index])
        end = start + int(self.num_elements[index])
 
        flat_array = self.mmap_data[start:end]
        shape = tuple(int(s) for s in self.shapes[index])
        spec = torch.from_numpy(flat_array.copy()).view(shape)
 
        tok_start = int(self.token_offsets[index])
        tok_end = int(self.token_offsets[index + 1])
        tokens = torch.from_numpy(self.flat_tokens[tok_start:tok_end].copy())
 
        spec_len = int(self.spec_lens[index])
        text_len = int(self.text_lens[index])
 
        return spec, tokens, spec_len, text_len
 
    def __len__(self):
        return self._n
 
    def get_lengths(self):
        return self.spec_lens


class LengthBucketBatchSampler(Sampler):
    #builds batches with similar lenghts
 
    def __init__(
        self,
        lengths,
        batch_size,
        shuffle=True,
        pool_multiplier=16,
        drop_last=False,
    ):
        if isinstance(lengths, np.ndarray):
            self.lengths = lengths.tolist()
        else:
            self.lengths = [
                int(x.item()) if isinstance(x, torch.Tensor) else int(x)
                for x in lengths
            ]
 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool_size = batch_size * pool_multiplier
        self.drop_last = drop_last
 

        indices = list(range(len(self.lengths)))
 

        pools = [
            indices[i: i + self.pool_size]
            for i in range(0, len(indices), self.pool_size)
        ]
        flat_sorted_indices = []
        for pool in pools:
            pool.sort(key=lambda idx: self.lengths[idx])
            flat_sorted_indices.extend(pool)
 

        self.precomputed_batches = []
        for i in range(0, len(flat_sorted_indices), self.batch_size):
            batch = flat_sorted_indices[i: i + self.batch_size]
            if len(batch) == self.batch_size:
                self.precomputed_batches.append(batch)
            elif not self.drop_last:
                self.precomputed_batches.append(batch)
 
    def __iter__(self):
        batches = list(self.precomputed_batches)
        if self.shuffle:
            random.shuffle(batches)
        yield from batches
 
    def __len__(self):
        return len(self.precomputed_batches)


def decode_ctc(
    logits: torch.Tensor, 
    lengths: torch.Tensor, 
    tokenizer: Any, 
    blank_id: int = 0,
    time_major: bool = False
) -> List[str]:

    if time_major:
        logits = logits.transpose(0, 1)  

    preds = logits.argmax(dim=-1)  
    preds_cpu = preds.detach().cpu().tolist()
    lengths_cpu = lengths.detach().cpu().tolist()

    batch_indices = []
    for pred, length in zip(preds_cpu, lengths_cpu):
        valid_pred = pred[:length]

        indices = []
        prev = None
        
        for idx in valid_pred:
            if idx != blank_id and idx != prev:
                indices.append(idx)
            prev = idx  
        
        batch_indices.append(indices)

    if hasattr(tokenizer, "batch_decode"):
        try:
            return tokenizer.batch_decode(batch_indices)
        except TypeError:
            pass
        
    return [tokenizer.decode(idx) for idx in batch_indices]



def test(model, test_loader, criterion, tokenizer, device, time_major=True):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    all_preds = []
    all_targets = []
    amp_enabled = (device.type == "cuda")

    with torch.no_grad():
        for specs, targets, input_lengths, target_lengths in tqdm(test_loader, desc="Testing", leave=False):
            specs = specs.to(device, non_blocking=True)
            targets_gpu = targets.to(device, non_blocking=True)
            input_lengths_gpu = input_lengths.to(device, non_blocking=True)
            target_lengths_gpu = target_lengths.to(device, non_blocking=True)

            # 2. Forward pass in bfloat16
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits, new_lengths = model(specs, input_lengths_gpu)
                # Tensore PyTorch float32 for stability
                log_probs_gpu = F.log_softmax(logits.float(), dim=-1)
                
                # CTCLoss on gpu tensors
                loss = criterion(
                    log_probs_gpu, 
                    targets_gpu, 
                    new_lengths.flatten().long(), 
                    target_lengths_gpu.flatten().long()
                )

            running_loss += loss.detach()

            preds_text = decode_ctc(
                logits.float(), 
                new_lengths, 
                tokenizer, 
                time_major=time_major
            )
            all_preds.extend(preds_text)

            targets_cpu = targets.numpy()
            target_lens_cpu = target_lengths.numpy()
            for i in range(len(targets_cpu)):
                real_seq = targets_cpu[i][: target_lens_cpu[i]]
                all_targets.append(tokenizer.decode(real_seq))


    test_loss = (running_loss / len(test_loader)).item()

    p_clean = [p.lower().strip() for p in all_preds]
    t_clean = [t.lower().strip() for t in all_targets]


    print("\n" + "=" * 60)
    print("TEST SET EVALUATION - SAMPLE PREDICTIONS (KenLM 5-gram)")
    print("=" * 60)
    coppie = list(zip(t_clean, p_clean))
    campioni = random.sample(coppie, k=min(5, len(coppie))) if coppie else []
    for ref, pred in campioni:
        print(f"REF : '{ref}'")
        print(f"PRED: '{pred}'")
        print("-" * 60)

    # calculating final CER and WER
    valid_pairs = [(p, t) for p, t in zip(p_clean, t_clean) if len(t.strip()) > 0]
    if valid_pairs:
        p_final, t_final = zip(*valid_pairs)
        test_wer = wer(list(t_final), list(p_final)) * 100.0
        test_cer = cer(list(t_final), list(p_final)) * 100.0
    else:
        test_wer = 100.0
        test_cer = 100.0

    print(f"\nFINAL TEST RESULTS -> Loss: {test_loss:.4f} | WER: {test_wer:.2f}% | CER: {test_cer:.2f}%\n")

    return test_loss, test_wer, test_cer


def main():
    test_loss, test_wer, test_cer = test(model, test_loader, criterion, tokenizer, device)
    print(f"Test Loss: {test_loss:.4f} - Test WER: {test_wer:.2f}% - Test CER: {test_cer:.2f}%")

    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_root_dir = "dataset/Common Voice Scripted Speech 26.0 - Italian ¦ Mozilla Data Collective/cv-corpus-26.0-2026-06-12/it"
    tokenizer = ItaTokenizer()
    MODEL_PTH_PATH="quartznet_evo_attn_13M.pth"
    test_dataset = PackedASRDataset(
        root_dir="ASR/dataset/Common Voice Scripted Speech 26.0 - Italian ¦ Mozilla Data Collective/cv-corpus-26.0-2026-06-12/it",
        output_name="common_voice_test_packed",
        tokenizer=tokenizer,
    )
    test_lengths = test_dataset.get_lengths()

   
    test_bucket_sampler = LengthBucketBatchSampler(
        lengths=test_lengths,
        batch_size=64,
        shuffle=False,
        pool_multiplier=16
    )

   
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_bucket_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    ) 
    checkpoint = torch.load(MODEL_PTH_PATH, map_location=device, weights_only=True)
    model = QuartzNet_evo_2(num_classes=len(tokenizer)).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Modello acustico su: {device}")
    # loss function and optimizer
    grad_scaler = getattr(torch.amp, "GradScaler")
    scaler = grad_scaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"Usando device: {device}")
    
    model = model.to(device)
    
    
    x_dummy = torch.randn(1, 80, 200)       # [batch, freq, time]
    lengths_dummy = torch.tensor([200])       

    summary(
    model, 
    input_data=(x_dummy, lengths_dummy), 
    device=device.type
    )

    main()