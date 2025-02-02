from torch.utils.data import Dataset
import torch

class SignLanguageDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        """
        Args:
            src_data: List of tuples (tokens, attention_mask)
            tgt_data: List of padded animation matrices
        """
        self.src_data = src_data
        self.tgt_data = tgt_data
        
    def __len__(self):
        return len(self.src_data)
        
    def __getitem__(self, idx):
        src_tokens, src_mask = self.src_data[idx]
        tgt_matrix = self.tgt_data[idx]
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'src_mask': torch.tensor(src_mask, dtype=torch.bool),
            'tgt': torch.tensor(tgt_matrix, dtype=torch.float)
        }
