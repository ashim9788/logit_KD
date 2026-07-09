import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class OfflineDistillationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        return {
            "input_ids": torch.tensor(data["input_ids"], dtype=torch.long),
            "topk_vals": torch.tensor(data["topk_vals"], dtype=torch.float32),
            "topk_inds": torch.tensor(data["topk_inds"], dtype=torch.long)
        }

def collate_fn(batch, pad_token_id=151643, assistant_start_id=151644):
    input_ids = [item["input_ids"] for item in batch]
    topk_vals = [item["topk_vals"] for item in batch]
    topk_inds = [item["topk_inds"] for item in batch]

    padded_inputs = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_vals = nn.utils.rnn.pad_sequence(topk_vals, batch_first=True, padding_value=0.0)
    padded_inds = nn.utils.rnn.pad_sequence(topk_inds, batch_first=True, padding_value=0)

    attention_mask = (padded_inputs != pad_token_id).long()
    target_labels = padded_inputs.clone()
    
    for i, seq in enumerate(input_ids):
        seq_list = seq.tolist() if isinstance(seq, torch.Tensor) else list(seq)
        seq_len = len(seq_list)
        if assistant_start_id in seq_list:
            cutoff = seq_len - 1 - seq_list[::-1].index(assistant_start_id) + 1
            target_labels[i, :cutoff] = -100
        else:
            target_labels[i, :] = -100
        target_labels[i, seq_len:] = -100

    return {
        "input_ids": padded_inputs,
        "attention_mask": attention_mask,
        "target_labels": target_labels,
        "topk_vals": padded_vals,
        "topk_inds": padded_inds
    }