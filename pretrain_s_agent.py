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
from data_utils import load_tox21_sample, scaffold_split
from calibration_utils import calibrate_model

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
    smiles_list = [s for s, _ in raw_data]
    train_full_idx, test_idx = scaffold_split(smiles_list, test_frac=0.1, seed=seed)
    train_full_smiles = [smiles_list[i] for i in train_full_idx]
    rel_train_idx, rel_val_idx = scaffold_split(train_full_smiles, test_frac=0.1111111111, seed=seed)
    train_idx = [train_full_idx[i] for i in rel_train_idx]
    val_idx = [train_full_idx[i] for i in rel_val_idx]

    train_raw = [raw_data[i] for i in train_idx]
    val_raw = [raw_data[i] for i in val_idx]
    test_raw = [raw_data[i] for i in test_idx]

    train_ds = Tox21Dataset(train_raw, tokenizer)
    val_ds = Tox21Dataset(val_raw, tokenizer)
    test_ds = Tox21Dataset(test_raw, tokenizer)
    
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
    train_labels = torch.tensor(np.stack([y for _, y in train_raw]), dtype=torch.float32)
    num_pos = train_labels.sum(dim=0)
    num_neg = train_labels.size(0) - num_pos
    pos_weight = (num_neg / num_pos.clamp(min=1.0)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info("Class balance pos/neg: %.1f / %.1f", float(num_pos.sum().item()), float(num_neg.sum().item()))

    # 3. Training Loop
    best_val_auc = -1.0
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_logits = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_logits.append(logits.detach().cpu())
            
        train_loss /= max(1, len(train_loader))
        
        # Eval
        model.eval()
        val_loss = 0.0
        val_logits = []
        all_preds, all_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                val_logits.append(logits.detach().cpu())
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_trues.append(y.cpu().numpy())
                
        val_loss /= max(1, len(val_loader))
        if train_logits:
            tr = torch.cat(train_logits, dim=0)
            logger.info("Train output mean/std: %.4f / %.4f", tr.mean().item(), tr.std(unbiased=False).item())
        if val_logits:
            va = torch.cat(val_logits, dim=0)
            logger.info("Val output mean/std: %.4f / %.4f", va.mean().item(), va.std(unbiased=False).item())
        all_preds = np.vstack(all_preds) if all_preds else np.zeros((0, 12), dtype=np.float32)
        all_trues = np.vstack(all_trues) if all_trues else np.zeros((0, 12), dtype=np.float32)
        
        # Calculate AUROC across 12 tasks
        aucs = []
        for i in range(12):
            if all_trues.shape[0] > 0 and len(np.unique(all_trues[:, i])) == 2:
                aucs.append(roc_auc_score(all_trues[:, i], all_preds[:, i]))
        val_auc = np.mean(aucs) if aucs else 0.5
        
        logger.info(f"Epoch {epoch:2d}/{num_epochs} | Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | Val AUROC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # Optional: Eval on Test
            
            # Save the ENCODER weights so RL phase generic SafetyAgent can load it
            torch.save(model.encoder.state_dict(), "toxicity_model.pt")
            torch.save(model.state_dict(), "toxicity_model_raw_head.pt")

    # Post-hoc temperature scaling on validation data for calibrated logits.
    if os.path.exists("toxicity_model_raw_head.pt"):
        model.load_state_dict(torch.load("toxicity_model_raw_head.pt", map_location=device))
        scaler = calibrate_model(model, val_loader, device)
        torch.save(scaler.state_dict(), "safety_temp_scaler.pt")
        logger.info("Saved safety_temp_scaler.pt.")

    logger.info("Saved toxicity_model.pt (encoder weights).")

if __name__ == "__main__":
    pretrain_s_agent()
