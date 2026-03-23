import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "data", "tinyshakespeare.txt")

class Shakespeare():
    def __init__(self, file_path=file_path, sequence_length=100):
        with open(file_path, 'r') as f:
            self.text = f.read()
        
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.sequence_length = sequence_length
        self.vocab_size = len(self.chars)

        self.char_indices = torch.tensor([self.char_to_idx[c] for c in self.text], dtype=torch.long)
    
    def __len__(self):
        return len(self.text) - self.sequence_length
    
    def __getitem__(self, idx):
        input_seq = self.char_indices[idx:idx + self.sequence_length]
        target_seq = self.char_indices[idx + 1:idx + self.sequence_length + 1]
        return input_seq, target_seq
    
    def train_test_split(self, split_ratio=0.9):
        split_idx = int(len(self) * split_ratio)
        train_data = torch.utils.data.Subset(self, range(split_idx))
        test_data = torch.utils.data.Subset(self, range(split_idx, len(self)))
        return train_data, test_data

def get_shakespeare_loaders(batch_size=64, num_workers=2, sequence_length=100, split_ratio=0.9):
    dataset = Shakespeare(sequence_length=sequence_length)
    train_data, test_data = dataset.train_test_split(split_ratio=split_ratio)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader