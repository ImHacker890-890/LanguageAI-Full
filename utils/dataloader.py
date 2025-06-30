from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_texts, trg_texts):
        self.src_texts = src_texts
        self.trg_texts = trg_texts

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        return self.src_texts[idx], self.trg_texts[idx]
