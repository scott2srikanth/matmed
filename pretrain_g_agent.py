import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import get_logger, set_seed
from generator_agent import GeneratorAgent
from utils import SMILESTokenizer
from data_utils import load_chembl_sample

logger = get_logger('G-Agent Pretrain')

class SMILESDataset(Dataset):
    def __init__(self, smiles_list: list, tokenizer: SMILESTokenizer, max_len: int = 128):
        self.smiles = smiles_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        tokens = self.tokenizer.encode(smi)
        # truncate or pad
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [self.tokenizer.pad_idx] * (self.max_len - len(tokens))
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def pretrain_g_agent_chembl(
    num_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
):
    set_seed(seed)
    logger.info(f"Phase 4: Starting G-Agent Pretraining on ChEMBL ({device})")

    # 1. Load Data
    smiles_list = load_chembl_sample(num_samples=50000)  # large corpus for proper LM pretraining
    tokenizer = SMILESTokenizer()
    dataset = SMILESDataset(smiles_list, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Init Model
    model = GeneratorAgent(
        tokenizer=tokenizer,
        d_model=256,
        nhead=8,
        num_layers=4,    # matching spec
        max_len=128
    ).to(device)

    # 3. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    # 4. Training Loop
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(x)  # (B, seq_len, vocab_size), embedding
            
            # flatten
            logits = logits.view(-1, tokenizer.vocab_size)
            y = y.view(-1)
            
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        logger.info(f"Epoch {epoch:2d}/{num_epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    # 5. Save model
    save_path = "generator_pretrained.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved pretrained G-Agent to {save_path}")

if __name__ == "__main__":
    pretrain_g_agent_chembl()
