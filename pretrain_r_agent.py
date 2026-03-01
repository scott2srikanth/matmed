"""
pretrain_r_agent.py — Phase 4: Pretrain R-Agent on USPTO
=========================================================
Trains the Reaction Feasibility Agent (R-Agent) on USPTO to predict synthetic feasibility
(binary classification proxy). Uses the Phase 3 Vision-Agent architecture.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from utils import get_logger, set_seed
from reaction_agent import ReactionAgent
from vision_agent import simulate_reaction_video
from data_utils import load_uspto_sample

logger = get_logger('R-Agent Pretrain')

class USPTODataset(Dataset):
    def __init__(self, data_tuples: list):
        self.data = []
        for smi, label in data_tuples:
            from reaction_agent import compute_reaction_features
            feats = compute_reaction_features(smi)
            if feats is not None:
                vis_seq = simulate_reaction_video(feats)
                self.data.append((smi, feats, vis_seq, float(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def pretrain_r_agent(
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
):
    set_seed(seed)
    logger.info(f"Phase 4: Starting R-Agent Pretraining on USPTO ({device})")

    # 1. Load Data
    raw_data = load_uspto_sample(1500)
    dataset = USPTODataset(raw_data)
    
    n = len(dataset)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    n_te = n - n_tr - n_va
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_tr, n_va, n_te])
    
    # Custom collate because SMILES are strings
    def collate_fn(batch):
        smiles = [b[0] for b in batch]
        feats  = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        vis    = torch.stack([b[2] for b in batch])
        labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        return smiles, feats, vis, labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Init Model
    # R-Agent (Phase 3 spec) with Vision Agent.
    # Output of forward is (score, emb), but we just need score.
    
    # Since forward() normally returns R = mu - lambda*sigma over MC dropout, 
    # we'll train the underlying head properly:
    model = ReactionAgent(use_vision=True).to(device)
    
    # Switch R-Agent head to output logits for BCE
    # Replace sigmoid with identity just for BCEWithLogitsLoss
    orig_head = list(model.yield_head.children())
    # remove sigmoid (last layer)
    model.yield_head = nn.Sequential(*orig_head[:-1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for smiles_list, feats, vis, labels in train_loader:
            feats, vis, labels = feats.to(device), vis.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Use forward_features directly to bypass mc_dropout_predict for standard pretraining
            # Returns (mu, var, h_fused) normally since we hooked it.
            # But we ripped off the sigmoid, so yield_head returns logits.
            
            # Actually, the simplest is to bypass mc_dropout and just pass h_fused through yield_head once
            mol_emb = model._encode_mol(feats)
            h_vision = model.vision_transformer(vis)
            h_fused = model.cross_attn(mol_emb, h_vision)
            
            logits = model.yield_head(h_fused).squeeze(-1)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for smiles_list, feats, vis, labels in val_loader:
                feats, vis, labels = feats.to(device), vis.to(device), labels.to(device)
                
                mol_emb = model._encode_mol(feats)
                h_vision = model.vision_transformer(vis)
                h_fused = model.cross_attn(mol_emb, h_vision)
                logits = model.yield_head(h_fused).squeeze(-1)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        val_loss /= len(val_loader)
        acc = correct / total
        
        logger.info(f"Epoch {epoch:2d}/{num_epochs} | Train BCE: {train_loss:.4f} | Val BCE: {val_loss:.4f} | Val Acc: {acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save state dict. But FIRST, put the Sigmoid back so Phase 4 RL can load it correctly!
            model.yield_head = nn.Sequential(*orig_head).to(device)
            torch.save(model.state_dict(), "reaction_model.pt")
            # Take it off again for the next epoch iteration
            model.yield_head = nn.Sequential(*orig_head[:-1]).to(device)

    logger.info("Saved reaction_model.pt.")

if __name__ == "__main__":
    pretrain_r_agent()
