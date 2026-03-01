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
from data_utils import load_bindingdb_sample

logger = get_logger('E-Agent Pretrain')

class BindingDataset(Dataset):
    def __init__(self, data_tuples: list):
        self.data = []
        for smi, pic50 in data_tuples:
            g = smiles_to_graph(smi)
            if g is not None:
                self.data.append((g, float(pic50)))
                
        # Normalize pIC50 to [0,1]
        self.y = np.array([d[1] for d in self.data])
        self.y_min, self.y_max = self.y.min(), self.y.max()
        for i in range(len(self.data)):
            norm_pic50 = (self.data[i][1] - self.y_min) / (self.y_max - self.y_min + 1e-8)
            self.data[i] = (self.data[i][0], norm_pic50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_graphs(batch):
    # batch is list of (dict, label)
    x_list, edge_index_list, edge_attr_list, labels_list = [], [], [], []
    node_offset = 0
    
    for item in batch:
        g, label = item
        num_nodes = g['x'].size(0)
        
        x_list.append(g['x'])
        # shift edge indices by the number of nodes already in the batch
        shifted_edge_index = g['edge_index'] + node_offset
        edge_index_list.append(shifted_edge_index)
        edge_attr_list.append(g['edge_attr'])
        labels_list.append(label)
        
        node_offset += num_nodes
        
    batched_x = torch.cat(x_list, dim=0)
    batched_edge_index = torch.cat(edge_index_list, dim=1)
    batched_edge_attr = torch.cat(edge_attr_list, dim=0)
    batched_labels = torch.tensor(labels_list, dtype=torch.float32)
    
    batched_g = {
        'x': batched_x,
        'edge_index': batched_edge_index,
        'edge_attr': batched_edge_attr
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
    dataset = BindingDataset(raw_data)
    
    # 80/10/10 split
    n = len(dataset)
    n_tr = int(0.8 * n)
    n_va = int(0.1 * n)
    n_te = n - n_tr - n_va
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_tr, n_va, n_te])
    
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
        for batch_g, batch_y in train_loader:
            batch_g = {k: v.to(device) for k, v in batch_g.items()}
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred, _ = model.forward_graph(batch_g['x'], batch_g['edge_index'], batch_g['edge_attr'])
            pred = pred.squeeze(-1)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_g, batch_y in val_loader:
                batch_g = {k: v.to(device) for k, v in batch_g.items()}
                batch_y = batch_y.to(device)
                pred, _ = model.forward_graph(batch_g['x'], batch_g['edge_index'], batch_g['edge_attr'])
                pred = pred.squeeze(-1)
                val_loss += criterion(pred, batch_y).item()
        val_loss /= len(val_loader)
        
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
                    pred, _ = model.forward_graph(batch_g['x'], batch_g['edge_index'], batch_g['edge_attr'])
                    pred = pred.squeeze(-1)
                    test_loss += criterion(pred, batch_y).item()
                    preds.append(pred.cpu().numpy())
                    trues.append(batch_y.cpu().numpy())
            
            test_loss /= len(test_loader)
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
