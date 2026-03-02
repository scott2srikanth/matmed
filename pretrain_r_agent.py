"""
pretrain_r_agent.py — Phase 4: Pretrain R-Agent on USPTO
=========================================================
Trains the Reaction Feasibility Agent (R-Agent) on USPTO to predict synthetic
feasibility (binary classification proxy). Uses negative reaction sampling,
scaffold split, and post-hoc temperature scaling.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from calibration_utils import calibrate_model
from data_utils import load_uspto_sample, scaffold_split
from reaction_agent import ReactionAgent
from utils import get_logger, set_seed
from vision_agent import simulate_reaction_video

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


def generate_negative_samples(smiles_list, frac: float = 1.0):
    negatives = []
    n_neg = max(1, int(len(smiles_list) * frac))
    for _ in range(n_neg):
        smi = random.choice(smiles_list)
        chars = list(smi)
        random.shuffle(chars)
        corrupted = "".join(chars)
        if corrupted != smi:
            negatives.append(corrupted)
    return negatives


def corrupt_frames(frames: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    noise = torch.randn_like(frames) * noise_std
    return frames + noise


def pretrain_r_agent(
    num_epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 42
):
    set_seed(seed)
    logger.info(f"Phase 4: Starting R-Agent Pretraining on USPTO ({device})")

    # 1. Load Data + synthetic negatives
    raw_data = load_uspto_sample(1500)
    positive_smiles = [s for s, y in raw_data if float(y) >= 0.5]
    if not positive_smiles:
        positive_smiles = [s for s, _ in raw_data]
    negative_smiles = generate_negative_samples(positive_smiles, frac=1.0)

    smiles_all = positive_smiles + negative_smiles
    labels_all = [1.0] * len(positive_smiles) + [0.0] * len(negative_smiles)
    all_data = list(zip(smiles_all, labels_all))

    # Scaffold split: 80/10/10 via two-stage split.
    train_full_idx, test_idx = scaffold_split(smiles_all, test_frac=0.1, seed=seed)
    train_full_smiles = [smiles_all[i] for i in train_full_idx]
    rel_train_idx, rel_val_idx = scaffold_split(train_full_smiles, test_frac=0.1111111111, seed=seed)
    train_idx = [train_full_idx[i] for i in rel_train_idx]
    val_idx = [train_full_idx[i] for i in rel_val_idx]

    train_raw = [all_data[i] for i in train_idx]
    val_raw = [all_data[i] for i in val_idx]
    test_raw = [all_data[i] for i in test_idx]

    train_ds = USPTODataset(train_raw)
    val_ds = USPTODataset(val_raw)
    _test_ds = USPTODataset(test_raw)

    def collate_fn(batch):
        smiles = [b[0] for b in batch]
        feats = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        vis = torch.stack([b[2] for b in batch])
        labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        return smiles, feats, vis, labels

    def cal_collate_fn(batch):
        feats = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
        vis = torch.stack([b[2] for b in batch])
        labels = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        return (feats, vis), labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_cal_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=cal_collate_fn)

    # 2. Init model
    model = ReactionAgent(use_vision=True).to(device)
    orig_head = list(model.yield_head.children())
    model.yield_head = nn.Sequential(*orig_head[:-1]).to(device)  # logits head for BCEWithLogits

    class ReactionLogitWrapper(nn.Module):
        def __init__(self, r_model):
            super().__init__()
            self.r_model = r_model
            self.alpha = 1.0
            self.beta = 1.0

        def extract_embeddings(self, feats, vis):
            mol_emb = self.r_model._encode_mol(feats)
            h_vision = self.r_model.vision_transformer(vis)
            smiles_emb = F.normalize(mol_emb, dim=-1)
            vision_emb = F.normalize(h_vision, dim=-1)
            return self.alpha * smiles_emb, self.beta * vision_emb

        def forward_from_embeddings(self, smiles_emb, vision_emb):
            h_fused = self.r_model.cross_attn(smiles_emb, vision_emb)
            return self.r_model.yield_head(h_fused).squeeze(-1)

        def forward(self, feats, vis):
            smiles_emb, vision_emb = self.extract_embeddings(feats, vis)
            return self.forward_from_embeddings(smiles_emb, vision_emb)

    logit_model = ReactionLogitWrapper(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor(float(len(negative_smiles)) / max(1.0, float(len(positive_smiles))), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_outputs = []

        for _smiles, feats, vis, labels in train_loader:
            feats, vis, labels = feats.to(device), vis.to(device), labels.to(device)
            optimizer.zero_grad()
            smiles_emb, vision_emb = logit_model.extract_embeddings(feats, vis)
            logits = logit_model.forward_from_embeddings(smiles_emb, vision_emb)
            classification_loss = criterion(logits, labels)

            # Cross-modal contrastive alignment (InfoNCE-style).
            temperature = 0.1
            sim_matrix = torch.matmul(smiles_emb, vision_emb.T) / temperature
            pos_idx = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            contrastive_loss = F.cross_entropy(sim_matrix, pos_idx)

            # Vision negative augmentation.
            neg_vis = corrupt_frames(vis)
            _, vision_neg_emb = logit_model.extract_embeddings(feats, neg_vis)
            neg_logits = logit_model.forward_from_embeddings(smiles_emb, vision_neg_emb)
            neg_labels = torch.zeros_like(labels)
            neg_loss = criterion(neg_logits, neg_labels)

            loss = classification_loss + 0.1 * contrastive_loss + 0.5 * neg_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_outputs.append(logits.detach().cpu())

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_outputs = []
        with torch.no_grad():
            for _smiles, feats, vis, labels in val_loader:
                feats, vis, labels = feats.to(device), vis.to(device), labels.to(device)
                smiles_emb, vision_emb = logit_model.extract_embeddings(feats, vis)
                logits = logit_model.forward_from_embeddings(smiles_emb, vision_emb)
                classification_loss = criterion(logits, labels)
                temperature = 0.1
                sim_matrix = torch.matmul(smiles_emb, vision_emb.T) / temperature
                pos_idx = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
                contrastive_loss = F.cross_entropy(sim_matrix, pos_idx)
                neg_vis = corrupt_frames(vis)
                _, vision_neg_emb = logit_model.extract_embeddings(feats, neg_vis)
                neg_logits = logit_model.forward_from_embeddings(smiles_emb, vision_neg_emb)
                neg_labels = torch.zeros_like(labels)
                neg_loss = criterion(neg_logits, neg_labels)

                loss = classification_loss + 0.1 * contrastive_loss + 0.5 * neg_loss
                val_loss += loss.item()
                val_outputs.append(logits.detach().cpu())

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= max(1, len(val_loader))
        acc = correct / max(1, total)

        if train_outputs:
            tr = torch.cat(train_outputs, dim=0)
            logger.info("Train output mean/std: %.4f / %.4f", tr.mean().item(), tr.std(unbiased=False).item())
        if val_outputs:
            va = torch.cat(val_outputs, dim=0)
            logger.info("Val output mean/std: %.4f / %.4f", va.mean().item(), va.std(unbiased=False).item())

        logger.info(
            f"Epoch {epoch:2d}/{num_epochs} | Train BCE: {train_loss:.4f} | "
            f"Val BCE: {val_loss:.4f} | Val Acc: {acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.yield_head = nn.Sequential(*orig_head).to(device)  # restore sigmoid for RL compatibility
            torch.save(model.state_dict(), "reaction_model.pt")
            model.yield_head = nn.Sequential(*orig_head[:-1]).to(device)

    # Save vision embedding distribution stats for OOD checks in RL.
    model.eval()
    vis_embs = []
    with torch.no_grad():
        for _smiles, feats, vis, _labels in train_loader:
            feats, vis = feats.to(device), vis.to(device)
            _, vision_emb = logit_model.extract_embeddings(feats, vis)
            vis_embs.append(vision_emb.detach().cpu())
    if vis_embs:
        vis_cat = torch.cat(vis_embs, dim=0)
        vis_mean = vis_cat.mean(dim=0)
        vis_std = vis_cat.std(dim=0, unbiased=False)
        dist = torch.norm(vis_cat - vis_mean.unsqueeze(0), dim=-1)
        dist_std = dist.std(unbiased=False)
        np.savez(
            "vision_embed_stats.npz",
            mean=vis_mean.numpy(),
            std=vis_std.numpy(),
            dist_std=np.array(float(dist_std.item()), dtype=np.float32),
        )
        logger.info("Saved vision_embed_stats.npz.")

    # Temperature scaling for calibrated reaction logits.
    scaler = calibrate_model(logit_model, val_cal_loader, device)
    torch.save(scaler.state_dict(), "reaction_temp_scaler.pt")
    logger.info("Saved reaction_temp_scaler.pt.")
    logger.info("Saved reaction_model.pt.")


if __name__ == "__main__":
    pretrain_r_agent()
