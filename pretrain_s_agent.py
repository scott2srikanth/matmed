"""
pretrain_s_agent.py — Phase 4: Pretrain S-Agent on Tox21
==========================================================
Trains the Safety Critic (S-Agent) on multi-task binary toxicity classification.
Uses BCEWithLogitsLoss for 12 endpoints from Tox21. Tracks AUROC.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import get_logger, set_seed, SMILESTokenizer
from safety_agent import SafetyAgent
from data_utils import load_tox21_sample

logger = get_logger('S-Agent Pretrain')

class Tox21Dataset(Dataset):
    def __init__(self, data_tuples: list, tokenizer: SMILESTokenizer, max_len: int = 128):
        self.data = []
        for smi, labels in data_tuples:
            tokens = tokenizer.encode(smi)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens = tokens + [tokenizer.pad_idx] * (max_len - len(tokens))
            
            x = torch.tensor(tokens, dtype=torch.long)
            y = torch.tensor(labels, dtype=torch.float32)
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pretrain_s_agent(
    num_epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
):
    set_seed(seed)
    logger.info(f"Phase 4: Starting S-Agent Pretraining on Tox21 ({device})")

    # 1. Load Data
    raw_data = load_tox21_sample(2000)
    tokenizer = SMILESTokenizer()
    dataset = Tox21Dataset(raw_data, tokenizer)
    
    # 80/10/10 split
    n = len(dataset)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    n_te = n - n_tr - n_va
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_tr, n_va, n_te])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 2. Init Model (matching Phase 4 prompt specs: dim 256, 4 layers)
    # The SafetyAgent wrapper maps [0,1] probability ordinarily.
    # We will hook into its encoder for raw logits.
    
    # Let's adjust SafetyAgent to output 12 logits if we pass num_classes=12
    # In safety_agent.py its standard output is 1 probability.
    # To keep it modular without breaking Phase 1-3, we'll extract the backbone here:
    class Tox21SafetyModel(nn.Module):
        def __init__(self, base_agent):
            super().__init__()
            self.encoder = base_agent.encoder
            self.embed_dim = base_agent.embed_dim
            self.tox_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(self.embed_dim, 64),
                nn.GELU(),
                nn.Linear(64, 12)  # 12 Tox21 tasks
            )
            
        def forward(self, x):
            emb = self.encoder(x)
            return self.tox_head(emb)
            
    base_sa = SafetyAgent(use_chemberta=False, d_model=256).to(device)
    model = Tox21SafetyModel(base_sa).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    best_val_auc = -1.0
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Eval
        model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_trues.append(y.cpu().numpy())
                
        val_loss /= len(val_loader)
        all_preds = np.vstack(all_preds)
        all_trues = np.vstack(all_trues)
        
        # Calculate AUROC across 12 tasks
        aucs = []
        for i in range(12):
            if len(np.unique(all_trues[:, i])) == 2:
                aucs.append(roc_auc_score(all_trues[:, i], all_preds[:, i]))
        val_auc = np.mean(aucs) if aucs else 0.5
        
        logger.info(f"Epoch {epoch:2d}/{num_epochs} | Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | Val AUROC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Optional: Eval on Test
            
            # Save the ENCODER weights so RL phase generic SafetyAgent can load it
            torch.save(model.encoder.state_dict(), "toxicity_model.pt")

    logger.info("Saved toxicity_model.pt (encoder weights).")

if __name__ == "__main__":
    pretrain_s_agent()
