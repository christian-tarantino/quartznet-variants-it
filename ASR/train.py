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
from model.quartznet_model_evo import QuartzNet_evo
from torch.utils.tensorboard import SummaryWriter

class SpecAugment(nn.Module):
    def __init__(
        self,
        freq_mask_param=30,
        time_mask_param=45,
        n_freq_masks=2,
        n_time_masks=2,
    ):
        super().__init__()
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask_param)
            for _ in range(n_freq_masks)
        ])
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask_param)
            for _ in range(n_time_masks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected shape: [Batch, Mel_Bands, Time]
        if self.training:
            for f_mask in self.freq_masks:
                x = f_mask(x)
            for t_mask in self.time_masks:
                x = t_mask(x)
        return x


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

def collate_fn(batch):
  specs = [item[0].transpose(0, 1) for item in batch]
  targets = [item[1] for item in batch]

  input_lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
  target_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)

  specs_padded = pad_sequence(specs, batch_first=True)
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
        logits = logits.transpose(0, 1)  # [Batch, Time, Vocab]

    preds = logits.argmax(dim=-1)  # Shape: [Batch, Time]
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

def train(model, train_loader, criterion, optimizer, scheduler,device, epoch , spec_augment=None):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    epoch_start = time.time()
    amp_enabled = (device.type == "cuda")

    for specs, targets, input_lengths, target_lengths in tqdm(train_loader, desc="Training", leave=False):
        specs = specs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        if spec_augment is not None:
            use_aug = epoch < int(num_epochs * 0.8)  # only for the first 80% of the epochs

            if use_aug:
                specs=spec_augment(specs)


            else:
                specs=spec_augment_reduced(specs)

        optimizer.zero_grad(set_to_none=True)

        # 1. Forward pass in bfloat16
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits, new_lengths = model(specs, input_lengths)

        log_probs = F.log_softmax(logits.float(), dim=2)
        invalid_samples = (new_lengths < target_lengths).sum().item()
        if invalid_samples > 0:
            print(f"ATTENZIONE: {invalid_samples} campioni hanno T_out < U_target e vengono ignorati dalla CTCLoss!")
        loss = criterion(log_probs, targets, new_lengths, target_lengths)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.detach()

    epoch_time = time.time() - epoch_start
    samples_per_sec = len(train_loader.dataset) / epoch_time
    epoch_loss = (running_loss / len(train_loader)).item()

    return epoch_loss, epoch_time, samples_per_sec

def validate(model, val_loader, criterion, tokenizer, device, time_major=True):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    amp_enabled = (device.type == "cuda")

    with torch.no_grad():
        for specs, targets, input_lengths, target_lengths in val_loader:
            specs = specs.to(device, non_blocking=True)
            targets_gpu = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths_gpu = target_lengths.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits, new_lengths = model(specs, input_lengths)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                loss = criterion(log_probs, targets_gpu, new_lengths.flatten().long(), target_lengths_gpu)

            running_loss += loss.item()

            preds_text = decode_ctc(logits, new_lengths, tokenizer, time_major=time_major)
            all_preds.extend(preds_text)

            targets_list = targets.tolist()
            target_lens_list = target_lengths.tolist()
            for i in range(len(targets_list)):
                real_seq = targets_list[i][:target_lens_list[i]]
                all_targets.append(tokenizer.decode(real_seq))

    val_loss = running_loss / len(val_loader)

    p_clean = [p.lower().strip() for p in all_preds]
    t_clean = [t.lower().strip() for t in all_targets]

    print("\n" + "=" * 50)
    print("VALIDATION PREDS (REF vs PRED)")

    coppie = list(zip(t_clean, p_clean))
    campioni_casuali = random.sample(coppie, k=min(5, len(coppie)))

    for ref, pred in campioni_casuali:
        print(f"REF:  '{ref}'")
        print(f"PRED: '{pred}'")
        print("-" * 50)

    # WER and CER
    valid_pairs = [(p, t) for p, t in zip(p_clean, t_clean) if len(t.strip()) > 0]
    if valid_pairs:
        p_final, t_final = zip(*valid_pairs)
        
        val_wer = wer(list(t_final), list(p_final))
        val_cer = cer(list(t_final), list(p_final))
        

    else:
        val_wer = 100.0
        val_cer = 100.0

    return val_loss, val_wer, val_cer


def test(model, test_loader, criterion, tokenizer, decoder, device, time_major=True):
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


            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                logits, new_lengths = model(specs, input_lengths_gpu)
                log_probs_gpu = F.log_softmax(logits.float(), dim=-1)

                loss = criterion(
                    log_probs_gpu, 
                    targets_gpu, 
                    new_lengths.flatten().long(), 
                    target_lengths_gpu.flatten().long()
                )

            running_loss += loss.detach()

            log_probs_cpu = log_probs_gpu.detach().cpu().numpy()
            new_lens_cpu = new_lengths.detach().cpu().numpy().flatten().astype(int)

            if time_major:
                # Shape: [time, batch, classes] 
                batch_logits = [log_probs_cpu[: new_lens_cpu[i], i, :] for i in range(len(new_lens_cpu))]
            else:
                # Shape: [batch, time, classes]
                batch_logits = [log_probs_cpu[i, : new_lens_cpu[i], :] for i in range(len(new_lens_cpu))]

            # Beam Search + KenLM 
            with Pool(processes=4) as pool:
                preds_text = decoder.decode_batch(pool, batch_logits, beam_width=128)  # type: ignore
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
    print("TEST SET EVALUATION - SAMPLE PREDICTIONS (KenLM 4-gram)")
    print("=" * 60)
    coppie = list(zip(t_clean, p_clean))
    campioni = random.sample(coppie, k=min(5, len(coppie))) if coppie else []
    for ref, pred in campioni:
        print(f"REF : '{ref}'")
        print(f"PRED: '{pred}'")
        print("-" * 60)

    # final WER and CER
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

    total_start_time = time.perf_counter()
    train_losses = []
    val_losses = []
    val_wers = []  
    val_cers = []  
    print("Testing DataLoader speed...")
    start = time.time()
    for i, batch in enumerate(train_loader):
        if i > 50:
            break
    elapsed = time.time() - start
    print(f"avreage time per batch in DataLoader: {elapsed / 50:.4f} seconds")
    for epoch in range(num_epochs):
        # 1. Train step
        train_loss, train_time, samples_per_sec = train(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            epoch,
            spec_augment=spec_augment,
        )

        # 2. Validation step
        val_loss, val_wer, val_cer = validate(
            model, val_loader, criterion, tokenizer, device, time_major=True
        )

        # 3. metric logs
        t_loss_val = float(train_loss)
        v_loss_val = float(val_loss)
        v_wer_val = float(val_wer)
        v_cer_val = float(val_cer)

        train_losses.append(t_loss_val)
        val_losses.append(v_loss_val)
        val_wers.append(v_wer_val)
        val_cers.append(v_cer_val)

        # 4. Print console
        print(
            f"Epoch {epoch+1:02d}/{num_epochs:02d} | "
            f"Train Loss: {t_loss_val:.4f} | "
            f"Val Loss: {v_loss_val:.4f} | "
            f"Val WER: {v_wer_val * 100:.2f}% | "
            f"Val CER: {v_cer_val * 100:.2f}% | "
            f"Time: {train_time:.1f}s ({samples_per_sec:.0f} samples/s)"
        )


    gc.collect()
    torch.cuda.empty_cache()
    LM_PATH = "ASR/lm_5gram.bin" 
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

    with open("/home/christian/ai/ASR/unigrams_clean.txt", "r", encoding="utf-8") as f:
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
        beta=2.0,
    )


    
    # test set validation
    torch.save(model.state_dict(), "quartznet_pure_13M.pth") 
    test_loss, test_wer, test_cer = test(model, test_loader, criterion, tokenizer, decoder, device)
    print(f"Test Loss: {test_loss:.4f} - Test WER: {test_wer:.2f}% - Test CER: {test_cer:.2f}%")
    total_training_time = time.perf_counter() - total_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        f"Tempo totale di training: {int(hours)}h "
        f"{int(minutes)}m {seconds:.2f}s"
    )
    return train_losses, val_losses, val_wers


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_root_dir = "dataset/Common Voice Scripted Speech 26.0 - Italian ¦ Mozilla Data Collective/cv-corpus-26.0-2026-06-12/it"
    tokenizer = ItaTokenizer()
    

    #  Dataset Training
    train_dataset = PackedASRDataset(
        root_dir="ASR/dataset/Common Voice Scripted Speech 26.0 - Italian ¦ Mozilla Data Collective/cv-corpus-26.0-2026-06-12/it",
        output_name="common_voice_train_packed",
        tokenizer=tokenizer,
    )

    #  Dataset Validation
    val_dataset = PackedASRDataset(
        root_dir="ASR/dataset/Common Voice Scripted Speech 26.0 - Italian ¦ Mozilla Data Collective/cv-corpus-26.0-2026-06-12/it",
        output_name="common_voice_val_packed",
        tokenizer=tokenizer,
    )

    #  Dataset Test
    test_dataset = PackedASRDataset(
        root_dir="ASR/dataset/Common Voice Scripted Speech 26.0 - Italian ¦ Mozilla Data Collective/cv-corpus-26.0-2026-06-12/it",
        output_name="common_voice_test_packed",
        tokenizer=tokenizer,
    )


    train_lengths = train_dataset.get_lengths()
    val_lengths = val_dataset.get_lengths()
    test_lengths = test_dataset.get_lengths()

    train_bucket_sampler = LengthBucketBatchSampler(
        lengths=train_lengths,
        batch_size=64,
        shuffle=True,
        pool_multiplier=16
    )
    val_bucket_sampler = LengthBucketBatchSampler(
        lengths=val_lengths,
        batch_size=64,
        shuffle=False,
        pool_multiplier=16
    )
    test_bucket_sampler = LengthBucketBatchSampler(
        lengths=test_lengths,
        batch_size=64,
        shuffle=False,
        pool_multiplier=16
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_bucket_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,     
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_bucket_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_bucket_sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    ) 

    spec_augment = SpecAugment(
    freq_mask_param=15, time_mask_param=25, n_freq_masks=2, n_time_masks=2
    ).to(device)
    spec_augment_reduced = SpecAugment(
    freq_mask_param=5, time_mask_param=15, n_freq_masks=1, n_time_masks=1
    ).to(device)
    

    model = QuartzNet(num_classes=len(tokenizer)).to(device)
    num_epochs = 25
    # loss function and optimizer
    grad_scaler = getattr(torch.amp, "GradScaler")
    scaler = grad_scaler("cuda", enabled=device.type == "cuda")
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,                       
        steps_per_epoch=len(train_loader), 
        epochs=num_epochs,
        pct_start=0.3, 
        div_factor=10.0,                    # LR iniziale = max_lr / 10
        final_div_factor=1e3,              
        anneal_strategy='cos'
    )

    #off because of the different lenghts in input
    torch.backends.cudnn.benchmark = False
    print(f"Usando device: {device}")
    
    model = model.to(device)
    
    
    x_dummy = torch.randn(1, 80, 200)       # [batch, freq, time]
    lengths_dummy = torch.tensor([200])        

    summary(
    model, 
    input_data=(x_dummy, lengths_dummy), 
    device=device.type
    )
    #torch compile is not really useful with model this tiny
    #model = torch.compile(model, mode="reduce-overhead", dynamic=True)
    main()
