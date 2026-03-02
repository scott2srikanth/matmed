"""
pretrain_e_agent.py — Phase 4: Pretrain E-Agent on BindingDB
=============================================================
Trains the Graph Transformer (E-Agent) to predict pIC50 binding affinity.
Dataset is min-max normalized for training stability.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from utils import get_logger, set_seed
from evaluator_agent import EvaluatorAgent, smiles_to_graph
from data_utils import load_bindingdb_sample, scaffold_split

logger = get_logger('E-Agent Pretrain')

class BindingDataset(Dataset):
    def __init__(self, data_tuples: list):
        self.data = []
        raw_targets = []
        for smi, target in data_tuples:
            g = smiles_to_graph(smi)
            if g is not None:
                raw_targets.append(float(target))
                self.data.append((g, float(target)))
                
        # Log-transform to p-like scale: y = -log10(raw + eps)
        self.y = -np.log10(np.array(raw_targets, dtype=np.float64) + 1e-12)
        print("Binding target mean/std:", float(self.y.mean()), float(self.y.std()))

        # Normalize transformed targets to [0,1] for stable regression.
        self.y_min, self.y_max = self.y.min(), self.y.max()
        for i in range(len(self.data)):
            norm_pic50 = (self.y[i] - self.y_min) / (self.y_max - self.y_min + 1e-8)
            self.data[i] = (self.data[i][0], norm_pic50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_graphs(batch):
    # batch is list of (dict, label)
    x_list, edge_index_list, edge_attr_list, labels_list = [], [], [], []
    batch_idx_list = []
    node_offset = 0
    
    for idx, item in enumerate(batch):
        g, label = item
        num_nodes = g['x'].size(0)
        
        x_list.append(g['x'])
        # shift edge indices by the number of nodes already in the batch
        shifted_edge_index = g['edge_index'] + node_offset
        edge_index_list.append(shifted_edge_index)
        edge_attr_list.append(g['edge_attr'])
        labels_list.append(label)
        batch_idx_list.append(torch.full((num_nodes,), idx, dtype=torch.long))
        
        node_offset += num_nodes
        
    batched_x = torch.cat(x_list, dim=0)
    batched_edge_index = torch.cat(edge_index_list, dim=1)
    batched_edge_attr = torch.cat(edge_attr_list, dim=0)
    batched_labels = torch.tensor(labels_list, dtype=torch.float32)
    batched_batch_idx = torch.cat(batch_idx_list, dim=0)
    
    batched_g = {
        'x': batched_x,
        'edge_index': batched_edge_index,
        'edge_attr': batched_edge_attr,
        'batch_idx': batched_batch_idx
    }
    
    return batched_g, batched_labels

def pretrain_e_agent(
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
):
    set_seed(seed)
    logger.info(f"Phase 4: Starting E-Agent Pretraining on BindingDB ({device})")

    # 1. Load Data
    raw_data = load_bindingdb_sample(1000)
    smiles_list = [s for s, _ in raw_data]
    train_full_idx, test_idx = scaffold_split(smiles_list, test_frac=0.1, seed=seed)
    train_full_smiles = [smiles_list[i] for i in train_full_idx]
    rel_train_idx, rel_val_idx = scaffold_split(train_full_smiles, test_frac=0.1111111111, seed=seed)
    train_idx = [train_full_idx[i] for i in rel_train_idx]
    val_idx = [train_full_idx[i] for i in rel_val_idx]

    train_raw = [raw_data[i] for i in train_idx]
    val_raw = [raw_data[i] for i in val_idx]
    test_raw = [raw_data[i] for i in test_idx]

    train_ds = BindingDataset(train_raw)
    val_ds = BindingDataset(val_raw)
    test_ds = BindingDataset(test_raw)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

    # 2. Init Model (max 4 layers, 128 dim from prompt)
    model = EvaluatorAgent(
        hidden_dim=128,
        num_layers=4,
        nhead=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 3. Training Loop
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_outputs = []
        for batch_g, batch_y in train_loader:
            batch_g = {k: v.to(device) for k, v in batch_g.items()}
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred, _ = model.forward_graph(batch_g['x'], batch_g['edge_index'], batch_g['edge_attr'], batch_g['batch_idx'])
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_outputs.append(pred.detach().cpu())
            
        train_loss /= max(1, len(train_loader))
        
        # Eval
        model.eval()
        val_loss = 0.0
        val_outputs = []
        with torch.no_grad():
            for batch_g, batch_y in val_loader:
                batch_g = {k: v.to(device) for k, v in batch_g.items()}
                batch_y = batch_y.to(device)
                pred, _ = model.forward_graph(batch_g['x'], batch_g['edge_index'], batch_g['edge_attr'], batch_g['batch_idx'])
                val_loss += criterion(pred, batch_y).item()
                val_outputs.append(pred.detach().cpu())
        val_loss /= max(1, len(val_loader))
        if train_outputs:
            train_cat = torch.cat(train_outputs, dim=0)
            logger.info("Train output mean/std: %.4f / %.4f", train_cat.mean().item(), train_cat.std(unbiased=False).item())
        if val_outputs:
            val_cat = torch.cat(val_outputs, dim=0)
            logger.info("Val output mean/std: %.4f / %.4f", val_cat.mean().item(), val_cat.std(unbiased=False).item())
        
        logger.info(f"Epoch {epoch:2d}/{num_epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Test metrics
            test_loss = 0.0
            preds, trues = [], []
            with torch.no_grad():
                for batch_g, batch_y in test_loader:
                    batch_g = {k: v.to(device) for k, v in batch_g.items()}
                    batch_y = batch_y.to(device)
                    pred, _ = model.forward_graph(batch_g['x'], batch_g['edge_index'], batch_g['edge_attr'], batch_g['batch_idx'])
                    test_loss += criterion(pred, batch_y).item()
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(batch_y.cpu().numpy())
            
            test_loss /= len(test_loader) if len(test_loader) > 0 else 1 # Handle empty test_loader
            if len(preds) == 0:
                logger.warning("   [New Best] Test set is empty. No test metrics.")
                torch.save(model.state_dict(), "binding_regressor.pt")
                continue
                
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            
            rmse = np.sqrt(np.mean((preds - trues)**2))
            mae = np.mean(np.abs(preds - trues))
            # R2
            ss_tot = np.sum((trues - np.mean(trues))**2) + 1e-8
            ss_res = np.sum((trues - preds)**2)
            r2 = 1 - (ss_res / ss_tot)
            
            logger.info(f"   [New Best] Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
            torch.save(model.state_dict(), "binding_regressor.pt")

    logger.info("Saved binding_regressor.pt.")

if __name__ == "__main__":
    pretrain_e_agent()
