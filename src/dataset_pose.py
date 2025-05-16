# src/dataset_pose.py
import torch
from torch.utils.data import Dataset
import os
import json

class PoseSequenceDataset(Dataset):
    def __init__(self, pose_dir, sequence_length=60):
        self.sequence_length = sequence_length
        self.samples = []
        self.labels = {"correcto": 1, "incorrecto": 0}

        for label in self.labels:
            dir_path = os.path.join(pose_dir, label)
            for file in os.listdir(dir_path):
                if file.endswith(".json"):
                    self.samples.append((os.path.join(dir_path, file), self.labels[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, 'r') as f:
            sequence = json.load(f)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label)
