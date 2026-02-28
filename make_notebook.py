"""
make_notebook.py
Run this script to generate matmed_colab.ipynb
"""
import json, textwrap

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)}

def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.splitlines(keepends=True)}

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(md("""# MATMED — Multi-Agent Transformer for Molecular Evolution & Design
### Google Colab Notebook
> **Research prototype** for drug discovery using a multi-agent RL + Transformer framework.
>
> **Agents:**  G-Agent (Generator) | E-Agent (Evaluator) | S-Agent (Safety / ADMET) | R-Agent (Reaction Feasibility) | P-Agent (Policy)
"""))

# ── Install ────────────────────────────────────────────────────────────────
cells.append(md("## 1. Install Dependencies"))
cells.append(code("""\
# Install required packages
# rdkit-pypi is the easiest RDKit install for Colab
!pip install -q rdkit-pypi torch torchvision transformers numpy pandas
print("✅ Dependencies installed")
"""))

# ── Clone / write files ────────────────────────────────────────────────────
cells.append(md("## 2. Write Project Files\nWe write all MATMED source files inline so this notebook is fully self-contained."))

# ── utils.py ──────────────────────────────────────────────────────────────
cells.append(md("### utils.py"))
cells.append(code('''\
%%writefile utils.py
"""
utils.py — Shared utilities for MATMED.
"""
import os, random, csv, logging, math
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs, AllChem

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

SMILES_CHARS = (
    [\'<PAD>\',\'<SOS>\',\'<EOS>\',\'<UNK>\'] +
    list(\'BCNOPSFI\') + list(\'bcnops\') + list(\'0123456789\') +
    list(\'()[]=#@+-\\\\\\./%\') + [\'Cl\',\'Br\',\'Si\',\'Se\',\'se\']
)

class SMILESTokenizer:
    PAD_TOKEN=\'<PAD>\'; SOS_TOKEN=\'<SOS>\'; EOS_TOKEN=\'<EOS>\'; UNK_TOKEN=\'<UNK>\'
    _MULTI = sorted([\'Cl\',\'Br\',\'Si\',\'Se\',\'se\'], key=len, reverse=True)

    def __init__(self, vocab=None):
        self.vocab = vocab or SMILES_CHARS
        self.char2idx = {c:i for i,c in enumerate(self.vocab)}
        self.idx2char = {i:c for c,i in self.char2idx.items()}
        self.pad_idx = self.char2idx[self.PAD_TOKEN]
        self.sos_idx = self.char2idx[self.SOS_TOKEN]
        self.eos_idx = self.char2idx[self.EOS_TOKEN]
        self.unk_idx = self.char2idx[self.UNK_TOKEN]

    @property
    def vocab_size(self): return len(self.vocab)

    def tokenize(self, smiles):
        tokens, i = [], 0
        while i < len(smiles):
            matched = False
            for m in self._MULTI:
                if smiles[i:i+len(m)] == m:
                    tokens.append(m); i += len(m); matched = True; break
            if not matched: tokens.append(smiles[i]); i += 1
        return tokens

    def encode(self, smiles, max_len=128, add_sos=True, add_eos=True):
        ids = [self.char2idx.get(t, self.unk_idx) for t in self.tokenize(smiles)]
        if add_sos: ids = [self.sos_idx] + ids
        if add_eos: ids = ids + [self.eos_idx]
        ids = ids[:max_len]; ids += [self.pad_idx]*(max_len-len(ids)); return ids

    def decode(self, ids, strip_special=True):
        out = []
        for i in ids:
            t = self.idx2char.get(i, self.UNK_TOKEN)
            if strip_special and t in (self.PAD_TOKEN,self.SOS_TOKEN,self.EOS_TOKEN): continue
            out.append(t)
        return \'\'.join(out)

def is_valid_smiles(s):
    if not s or not s.strip(): return False
    return Chem.MolFromSmiles(s) is not None

def canonicalize(s):
    m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else None

def tanimoto_similarity(s1, s2):
    m1,m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None: return 0.0
    f1 = AllChem.GetMorganFingerprintAsBitVect(m1,2,1024)
    f2 = AllChem.GetMorganFingerprintAsBitVect(m2,2,1024)
    return DataStructs.TanimotoSimilarity(f1,f2)

def diversity_score(smiles_list):
    valid = [s for s in smiles_list if is_valid_smiles(s)]
    if len(valid)<2: return 0.0
    total,count = 0.0,0
    for i in range(len(valid)):
        for j in range(i+1,len(valid)):
            total += 1.0 - tanimoto_similarity(valid[i],valid[j]); count+=1
    return total/count if count>0 else 0.0

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        from rdkit.Contrib.SA_Score import sascorer
        sa = sascorer.calculateScore(mol)
    except: sa = 5.0
    return {\'mol_weight\':Descriptors.MolWt(mol),\'logP\':Descriptors.MolLogP(mol),\'sa_score\':sa}

def get_logger(name=\'matmed\'):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(\'[%(asctime)s] %(levelname)s | %(name)s — %(message)s\',datefmt=\'%H:%M:%S\'))
        logger.addHandler(h)
    logger.setLevel(logging.INFO); return logger

def save_metrics_csv(metrics, filepath=\'matmed_metrics.csv\'):
    if not metrics: return
    fieldnames = list(metrics[0].keys())
    write_header = not os.path.exists(filepath)
    with open(filepath,\'a\',newline=\'\') as f:
        w = csv.DictWriter(f,fieldnames=fieldnames)
        if write_header: w.writeheader()
        w.writerows(metrics)

ZINC_SAMPLE = [
    "CC(=O)Oc1ccccc1C(=O)O","CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "OC(=O)c1ccccc1O","CC(=O)Nc1ccc(cc1)O","c1ccncc1",
    "Cc1nc2ccccc2s1","CC(=O)c1ccc(cc1)N","CCCC(=O)O",
    "c1ccc(cc1)O","N[C@@H](Cc1ccccc1)C(=O)O",
    "CC(N)C(=O)O","c1ccc2ncccc2c1","Cc1ccc(cc1)S(=O)(=O)N",
    "COc1ccc(cc1)C=O","N1C=CC=C1","c1cnc2[nH]cnc2c1",
]

def get_zinc_sample():
    return [s for s in ZINC_SAMPLE if is_valid_smiles(s)]

print("✅ utils.py written")
'''))

# ── reward.py ─────────────────────────────────────────────────────────────
cells.append(md("### reward.py"))
cells.append(code('''\
%%writefile reward.py
"""reward.py — MATMED composite reward function."""
from dataclasses import dataclass

@dataclass
class RewardConfig:
    alpha:float=1.0; beta:float=0.5; gamma:float=0.5; delta:float=1.0
    clip_min:float=-5.0; clip_max:float=5.0

class RewardFunction:
    def __init__(self, config=None):
        self.config = config or RewardConfig()
    def compute(self, binding_score, yield_score, admet_score, toxicity):
        c = self.config
        r = c.alpha*binding_score + c.beta*yield_score + c.gamma*admet_score - c.delta*toxicity
        return float(max(c.clip_min, min(c.clip_max, r)))
    def breakdown(self, binding_score, yield_score, admet_score, toxicity):
        c = self.config
        bt = c.alpha*binding_score; yt = c.beta*yield_score
        at = c.gamma*admet_score;  tp = c.delta*toxicity
        total = max(c.clip_min, min(c.clip_max, bt+yt+at-tp))
        return {\'binding_term\':bt,\'yield_term\':yt,\'admet_term\':at,\'toxicity_penalty\':tp,\'total_reward\':total}

print("✅ reward.py written")
'''))

# ── generator_agent.py ────────────────────────────────────────────────────
cells.append(md("### generator_agent.py"))
cells.append(code('''\
%%writefile generator_agent.py
"""generator_agent.py — G-Agent: Decoder-only Transformer for SMILES generation."""
import math
from typing import Optional, Tuple, List
import torch, torch.nn as nn, torch.nn.functional as F
from utils import SMILESTokenizer, is_valid_smiles, get_zinc_sample

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0,d_model,2,dtype=torch.float)*(-math.log(10000.)/d_model))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer(\'pe\', pe.unsqueeze(0))
    def forward(self,x):
        x = x + self.pe[:,:x.size(1),:]; return self.dropout(x)

class GeneratorAgent(nn.Module):
    """G-Agent: Decoder-only Transformer for autoregressive SMILES generation."""
    def __init__(self, tokenizer=None, d_model=256, nhead=4, num_layers=4, d_ff=512, dropout=0.1, max_len=128):
        super().__init__()
        self.tokenizer = tokenizer or SMILESTokenizer()
        self.d_model=d_model; self.max_len=max_len
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, d_model, padding_idx=self.tokenizer.pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len+2, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_ff,
                                               dropout=dropout,batch_first=True,activation=\'gelu\')
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.tokenizer.vocab_size)
        self.ln = nn.LayerNorm(d_model)
        nn.init.xavier_uniform_(self.output_proj.weight); nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _causal_mask(seq_len, device):
        return torch.triu(torch.ones(seq_len,seq_len,device=device),diagonal=1).bool()

    def forward(self, input_ids, key_padding_mask=None):
        seq_len = input_ids.size(1); device = input_ids.device
        x = self.embedding(input_ids)*math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        h = self.transformer(x, mask=self._causal_mask(seq_len,device), src_key_padding_mask=key_padding_mask)
        logits = self.output_proj(h)
        if key_padding_mask is not None:
            act = (~key_padding_mask).float().unsqueeze(-1)
            emb = (self.ln(h)*act).sum(1)/act.sum(1).clamp(min=1)
        else:
            emb = self.ln(h).mean(dim=1)
        return logits, emb

    def compute_loss(self, src, tgt, pad_idx=None):
        pad = pad_idx if pad_idx is not None else self.tokenizer.pad_idx
        logits, _ = self.forward(src, key_padding_mask=(src==pad))
        return F.cross_entropy(logits.reshape(-1,logits.size(-1)), tgt.reshape(-1), ignore_index=pad)

    @torch.no_grad()
    def generate(self, batch_size=1, temperature=1.0, top_k=0, device=None):
        if device is None: device = next(self.parameters()).device
        self.eval()
        sos,eos,pad = self.tokenizer.sos_idx, self.tokenizer.eos_idx, self.tokenizer.pad_idx
        seqs = torch.full((batch_size,1),sos,dtype=torch.long,device=device)
        finished = torch.zeros(batch_size,dtype=torch.bool,device=device)
        embeddings = None
        for _ in range(self.max_len-1):
            logits, emb = self.forward(seqs); embeddings = emb
            next_logits = logits[:,-1,:]/temperature
            if top_k>0:
                tv = next_logits.topk(top_k,dim=-1).values[:,-1:]
                next_logits = next_logits.masked_fill(next_logits<tv, -float(\'inf\'))
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            next_tok[finished] = pad
            seqs = torch.cat([seqs,next_tok],dim=1)
            finished = finished|(next_tok.squeeze(-1)==eos)
            if finished.all(): break
        smiles = [self.tokenizer.decode(seqs[i].tolist()) for i in range(batch_size)]
        return smiles, embeddings

def pretrain_generator(agent, smiles_list, num_epochs=5, lr=1e-3, device=None):
    if device is None: device = torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
    agent = agent.to(device); opt = torch.optim.Adam(agent.parameters(), lr=lr)
    tok = agent.tokenizer; encoded = [tok.encode(s, max_len=agent.max_len) for s in smiles_list]
    agent.train()
    for epoch in range(num_epochs):
        total = 0.0
        for ids in encoded:
            seq = torch.tensor(ids,dtype=torch.long,device=device).unsqueeze(0)
            loss = agent.compute_loss(seq[:,:-1], seq[:,1:])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(),1.0); opt.step()
            total += loss.item()
        print(f"  [G-Agent] Epoch {epoch+1}/{num_epochs}  loss={total/len(encoded):.4f}")

print("✅ generator_agent.py written")
'''))

# ── evaluator_agent.py ────────────────────────────────────────────────────
cells.append(md("### evaluator_agent.py"))
cells.append(code('''\
%%writefile evaluator_agent.py
"""evaluator_agent.py — E-Agent: Graph Transformer for binding score prediction."""
from typing import Tuple, Optional, List
import torch, torch.nn as nn, torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem

ATOM_TYPES = [\'C\',\'N\',\'O\',\'S\',\'F\',\'P\',\'Cl\',\'Br\',\'I\',\'other\']
ATOM_TYPE_IDX = {a:i for i,a in enumerate(ATOM_TYPES)}
HYBRIDIZATION_MAP = {rdchem.HybridizationType.SP:0,rdchem.HybridizationType.SP2:1,
                     rdchem.HybridizationType.SP3:2,rdchem.HybridizationType.SP3D:3,rdchem.HybridizationType.SP3D2:4}
BOND_TYPE_MAP = {rdchem.BondType.SINGLE:0,rdchem.BondType.DOUBLE:1,rdchem.BondType.TRIPLE:2,rdchem.BondType.AROMATIC:3}
NODE_FEAT_DIM = len(ATOM_TYPES)+3
EDGE_FEAT_DIM = 7

def atom_features(atom):
    oh = [0.]*len(ATOM_TYPES); oh[ATOM_TYPE_IDX.get(atom.GetSymbol(),ATOM_TYPE_IDX[\'other\'])]=1.
    return oh+[atom.GetDegree()/6., HYBRIDIZATION_MAP.get(atom.GetHybridization(),5)/5., float(atom.GetIsAromatic())]

def bond_features(bond):
    oh=[0.]*4; oh[BOND_TYPE_MAP.get(bond.GetBondType(),0)]=1.
    return oh+[float(bond.GetIsConjugated()),float(bond.IsInRing()),float(bond.GetStereo())/6.]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()],dtype=torch.float)
    ei,ea = [[],[]],[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); bf=bond_features(b)
        for s,d in [(i,j),(j,i)]: ei[0].append(s); ei[1].append(d); ea.append(bf)
    if not ei[0]: ei[0].append(0);ei[1].append(0);ea.append([0.]*EDGE_FEAT_DIM)
    return {\'x\':x,\'edge_index\':torch.tensor(ei,dtype=torch.long),\'edge_attr\':torch.tensor(ea,dtype=torch.float)}

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.d_model=d_model; self.nhead=nhead; self.dk=d_model//nhead
        self.q=nn.Linear(d_model,d_model); self.k=nn.Linear(d_model,d_model)
        self.v=nn.Linear(d_model,d_model); self.o=nn.Linear(d_model,d_model)
        self.edge_bias=nn.Linear(EDGE_FEAT_DIM,nhead)
        self.ffn=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model*2,d_model))
        self.ln=nn.LayerNorm(d_model); self.drop=nn.Dropout(dropout)
    def forward(self, h, edge_index, edge_attr):
        N=h.size(0); res=h
        attn_bias=torch.zeros(N,N,self.nhead,device=h.device)
        if edge_index.size(1)>0:
            si,di=edge_index[0],edge_index[1]; attn_bias[si,di]=self.edge_bias(edge_attr)
        Q=self.q(h).view(N,self.nhead,self.dk); K=self.k(h).view(N,self.nhead,self.dk); V=self.v(h).view(N,self.nhead,self.dk)
        sc=torch.einsum(\'nhd,mhd->nmh\',Q,K)/(self.dk**.5)+attn_bias
        at=F.softmax(sc,dim=1); out=torch.einsum(\'nmh,mhd->nhd\',at,V).reshape(N,self.d_model)
        h=self.ln(res+self.drop(self.o(out))); return h+self.drop(self.ffn(h))

class EvaluatorAgent(nn.Module):
    """E-Agent: Graph Transformer predicting binding affinity from molecular graph."""
    def __init__(self, hidden_dim=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.node_proj=nn.Linear(NODE_FEAT_DIM,hidden_dim)
        self.gt_layers=nn.ModuleList([GraphTransformerLayer(hidden_dim,nhead,dropout) for _ in range(num_layers)])
        self.head=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1),nn.Sigmoid())
    def forward(self, smiles):
        g=smiles_to_graph(smiles)
        if g is None: raise ValueError(f"Cannot parse: {smiles!r}")
        dev=next(self.parameters()).device
        x,ei,ea=g[\'x\'].to(dev),g[\'edge_index\'].to(dev),g[\'edge_attr\'].to(dev)
        h=self.node_proj(x)
        for layer in self.gt_layers: h=layer(h,ei,ea)
        emb=h.mean(dim=0); score=self.head(emb.unsqueeze(0)).squeeze()
        return float(score.item()), emb

print("✅ evaluator_agent.py written")
'''))

# ── safety_agent.py ───────────────────────────────────────────────────────
cells.append(md("### safety_agent.py"))
cells.append(code('''\
%%writefile safety_agent.py
"""safety_agent.py — S-Agent: ADMET Critic (ChemBERTa or fallback encoder)."""
import math, logging
from typing import Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from utils import SMILESTokenizer, get_logger

logger = get_logger(\'S-Agent\')

class _PE(nn.Module):
    def __init__(self, d, max_len=256, dp=0.1):
        super().__init__(); self.drop=nn.Dropout(dp)
        pe=torch.zeros(max_len,d); pos=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div=torch.exp(torch.arange(0,d,2,dtype=torch.float)*(-math.log(10000.)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer(\'pe\',pe.unsqueeze(0))
    def forward(self,x): return self.drop(x+self.pe[:,:x.size(1),:])

class _SimpleEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, d_ff=256, dp=0.1, max_len=128, pad_idx=0):
        super().__init__(); self.pad_idx=pad_idx; self.d_model=d_model
        self.emb=nn.Embedding(vocab_size,d_model,padding_idx=pad_idx)
        self.pe=_PE(d_model,max_len+2,dp)
        el=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_ff,dropout=dp,batch_first=True,activation=\'gelu\')
        self.enc=nn.TransformerEncoder(el,num_layers=num_layers); self.ln=nn.LayerNorm(d_model)
    def forward(self, input_ids):
        mask=(input_ids==self.pad_idx)
        x=self.emb(input_ids)*math.sqrt(self.d_model); x=self.pe(x)
        h=self.enc(x,src_key_padding_mask=mask)
        act=(~mask).float().unsqueeze(-1)
        return (self.ln(h)*act).sum(1)/act.sum(1).clamp(min=1)

class _ADMETHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,64),nn.GELU(),nn.Linear(64,2))
    def forward(self, emb):
        out=self.net(emb); return torch.sigmoid(out[...,0]), torch.sigmoid(out[...,1])

class SafetyAgent(nn.Module):
    """S-Agent: Predicts toxicity and ADMET score for candidate molecules."""
    CHEMBERTA = "seyonec/ChemBERTa-zinc-base-v1"
    def __init__(self, d_model=128, max_len=128, use_chemberta=True):
        super().__init__(); self.max_len=max_len; self.tok=SMILESTokenizer(); self._cb=False; self._hf=None
        if use_chemberta:
            try:
                from transformers import AutoTokenizer, AutoModel
                self._hf=AutoTokenizer.from_pretrained(self.CHEMBERTA)
                cm=AutoModel.from_pretrained(self.CHEMBERTA)
                self.encoder=cm; self.proj=nn.Linear(cm.config.hidden_size,d_model)
                self.embed_dim=d_model; self._cb=True; logger.info("ChemBERTa loaded.")
            except Exception as e:
                logger.warning(f"ChemBERTa unavailable ({e}). Using fallback.")
        if not self._cb:
            self.encoder=_SimpleEncoder(self.tok.vocab_size,d_model,max_len=max_len,pad_idx=self.tok.pad_idx)
            self.embed_dim=d_model
        self.head=_ADMETHead(self.embed_dim)
    def _enc(self, smiles):
        dev=next(self.parameters()).device
        if self._cb:
            inp=self._hf(smiles,return_tensors=\'pt\',max_length=self.max_len,truncation=True,padding=True)
            inp={k:v.to(dev) for k,v in inp.items()}
            with torch.no_grad(): out=self.encoder(**inp)
            return self.proj(out.last_hidden_state[:,0,:]).squeeze(0)
        else:
            ids=self.tok.encode(smiles,max_len=self.max_len)
            t=torch.tensor([ids],dtype=torch.long,device=dev)
            return self.encoder(t).squeeze(0)
    def forward(self, smiles):
        emb=self._enc(smiles); tox,admet=self.head(emb)
        return float(tox.item()), float(admet.item()), emb

print("✅ safety_agent.py written")
'''))

# ── reaction_agent.py ─────────────────────────────────────────────────────
cells.append(md("### reaction_agent.py"))
cells.append(code('''\
%%writefile reaction_agent.py
"""reaction_agent.py — R-Agent: Reaction Feasibility Predictor (heuristic MLP)."""
import math
from typing import Optional, Tuple
import numpy as np
import torch, torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors

REACTION_FEAT_DIM = 3

def _sa_score(mol):
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return float(sascorer.calculateScore(mol))
    except: return min(10., 1.+mol.GetNumHeavyAtoms()/10.)

def compute_reaction_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    sa   = _sa_score(mol)
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    return np.array([1.-(sa-1.)/9., math.exp(-mw/500.), math.exp(-((logp-2.5)**2)/4.)], dtype=np.float32)

class ReactionAgent(nn.Module):
    """R-Agent: Predicts synthetic yield score from physicochemical descriptors."""
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__(); self.hidden_dim=hidden_dim
        self.encoder=nn.Sequential(nn.Linear(REACTION_FEAT_DIM,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,hidden_dim),nn.GELU())
        self.ln=nn.LayerNorm(hidden_dim)
        self.head=nn.Sequential(nn.Linear(hidden_dim,32),nn.GELU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self, smiles):
        f=compute_reaction_features(smiles)
        dev=next(self.parameters()).device
        if f is None: return 0.0, torch.zeros(self.hidden_dim,device=dev)
        t=torch.tensor(f,dtype=torch.float,device=dev).unsqueeze(0)
        emb=self.ln(self.encoder(t)); score=self.head(emb).squeeze(-1)
        return float(score[0].item()), emb[0]

print("✅ reaction_agent.py written")
'''))

# ── policy_agent.py ───────────────────────────────────────────────────────
cells.append(md("### policy_agent.py"))
cells.append(code('''\
%%writefile policy_agent.py
"""policy_agent.py — P-Agent: Multi-objective Policy Transformer."""
from typing import Optional, Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from utils import get_logger

logger = get_logger(\'P-Agent\')
ACTION_ACCEPT=0; ACTION_MODIFY=1; ACTION_REGENERATE=2; ACTION_STOP=3
ACTION_NAMES=[\'ACCEPT\',\'MODIFY\',\'REGENERATE\',\'STOP\']; NUM_ACTIONS=4

def _discount_returns(rewards, gamma=0.99):
    T=rewards.size(0); returns=torch.zeros_like(rewards); G=0.
    for t in reversed(range(T)): G=rewards[t].item()+gamma*G; returns[t]=G
    return returns

class PolicyAgent(nn.Module):
    """P-Agent: Integrates G/E/S/R embeddings via Transformer → discrete action."""
    def __init__(self, d_g=256, d_e=128, d_s=128, d_r=128, d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.proj_g=nn.Linear(d_g,d_model); self.proj_e=nn.Linear(d_e,d_model)
        self.proj_s=nn.Linear(d_s,d_model); self.proj_r=nn.Linear(d_r,d_model)
        self.proj_rew=nn.Linear(1,d_model)
        self.cls=nn.Parameter(torch.randn(1,1,d_model))
        self.pos=nn.Embedding(7,d_model)
        el=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*2,dropout=dropout,batch_first=True,activation=\'gelu\')
        self.transformer=nn.TransformerEncoder(el,num_layers=num_layers)
        self.ln=nn.LayerNorm(d_model)
        self.decision=nn.Sequential(nn.Linear(d_model,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,NUM_ACTIONS))
        self.value=nn.Sequential(nn.Linear(d_model,64),nn.GELU(),nn.Linear(64,1))
    def forward(self, g_emb, e_emb, s_emb, r_emb, reward=0.):
        if g_emb.dim()==1: g_emb=g_emb.unsqueeze(0);e_emb=e_emb.unsqueeze(0);s_emb=s_emb.unsqueeze(0);r_emb=r_emb.unsqueeze(0)
        B=g_emb.size(0); dev=g_emb.device
        g=self.proj_g(g_emb).unsqueeze(1); e=self.proj_e(e_emb).unsqueeze(1)
        s=self.proj_s(s_emb).unsqueeze(1); r=self.proj_r(r_emb).unsqueeze(1)
        rw=self.proj_rew(torch.tensor([[reward]],dtype=torch.float,device=dev).expand(B,1)).unsqueeze(1)
        cls=self.cls.expand(B,-1,-1)
        seq=torch.cat([cls,g,e,s,r,rw],dim=1)
        seq=seq+self.pos(torch.arange(seq.size(1),device=dev)).unsqueeze(0)
        h=self.transformer(seq); cls_out=self.ln(h[:,0,:])
        logits=self.decision(cls_out); probs=F.softmax(logits,dim=-1); val=self.value(cls_out)
        return logits, probs, val
    @torch.no_grad()
    def select_action(self, g_emb, e_emb, s_emb, r_emb, reward=0., greedy=False):
        _,probs,val=self.forward(g_emb,e_emb,s_emb,r_emb,reward)
        if greedy: action=int(probs.argmax(-1).item())
        else: action=int(torch.distributions.Categorical(probs).sample().item())
        lp=float(torch.log(probs[0,action]+1e-8).item())
        return action,lp,val.squeeze()
    def compute_policy_loss(self, log_probs, rewards, values, gamma=0.99, entropy_coeff=0.01, action_probs_all=None):
        ret=_discount_returns(rewards,gamma)
        adv=(ret-values.detach()); adv=(adv-adv.mean())/(adv.std()+1e-8)
        pl=-(log_probs*adv).mean(); vl=F.mse_loss(values,ret)
        ent=torch.tensor(0.)
        if action_probs_all is not None:
            ent=-(action_probs_all*torch.log(action_probs_all+1e-8)).sum(-1).mean()
        total=pl+0.5*vl-entropy_coeff*ent
        return {\'policy_loss\':pl,\'value_loss\':vl,\'entropy\':ent,\'total_loss\':total}

print("✅ policy_agent.py written")
'''))

# ── train_matmed.py ───────────────────────────────────────────────────────
cells.append(md("### train_matmed.py"))
cells.append(code('''\
%%writefile train_matmed.py
"""train_matmed.py — MATMED full training loop."""
import torch, torch.nn as nn
from typing import List, Dict, Optional
from utils import set_seed, get_logger, is_valid_smiles, diversity_score, get_zinc_sample, save_metrics_csv
from reward import RewardFunction, RewardConfig
from generator_agent import GeneratorAgent, pretrain_generator
from evaluator_agent import EvaluatorAgent
from safety_agent import SafetyAgent
from reaction_agent import ReactionAgent
from policy_agent import PolicyAgent, ACTION_NAMES, _discount_returns

logger = get_logger(\'MATMED\')

class Transition:
    __slots__ = (\'log_prob\',\'reward\',\'value\',\'action_probs\')
    def __init__(self,lp,r,v,ap): self.log_prob=lp;self.reward=r;self.value=v;self.action_probs=ap

class MATMEDRunner:
    """Orchestrates all MATMED agents through RL episodes with REINFORCE updates."""
    def __init__(self, reward_config=None, num_pretrain_epochs=5, lr_policy=3e-4,
                 gamma=0.99, entropy_coeff=0.01, seed=42, device=None, use_chemberta=False):
        set_seed(seed)
        if device is None: device=torch.device(\'cuda\' if torch.cuda.is_available() else \'cpu\')
        self.device=device; logger.info(f"Running on {device}")
        self.tokenizer=__import__(\'utils\').SMILESTokenizer()
        self.g_agent=GeneratorAgent(tokenizer=self.tokenizer).to(device)
        self.e_agent=EvaluatorAgent().to(device)
        self.s_agent=SafetyAgent(use_chemberta=use_chemberta).to(device)
        self.r_agent=ReactionAgent().to(device)
        self.p_agent=PolicyAgent(d_g=self.g_agent.d_model,d_e=self.e_agent.hidden_dim,
                                  d_s=self.s_agent.embed_dim,d_r=self.r_agent.hidden_dim).to(device)
        self.reward_fn=RewardFunction(reward_config)
        self.policy_optim=torch.optim.Adam(self.p_agent.parameters(),lr=lr_policy)
        self.gamma=gamma; self.entropy_coeff=entropy_coeff
        self.best_reward=-float(\'inf\'); self.best_smiles=None; self.all_metrics=[]
        logger.info("Pretraining G-Agent …")
        pretrain_generator(self.g_agent, get_zinc_sample(), num_epochs=num_pretrain_epochs)

    def _step(self, prev_reward=0.):
        self.g_agent.eval()
        smiles_list, g_emb_batch=self.g_agent.generate(1,temperature=1.0)
        smiles=smiles_list[0]; g_emb=g_emb_batch[0].detach()
        valid=is_valid_smiles(smiles); safe_smi=smiles if valid else \'C\'
        try: binding,e_emb=self.e_agent.forward(safe_smi); e_emb=e_emb.detach()
        except: binding=0.;e_emb=torch.zeros(self.e_agent.hidden_dim,device=self.device)
        try: tox,admet,s_emb=self.s_agent.forward(safe_smi); s_emb=s_emb.detach()
        except: tox,admet=0.5,0.5;s_emb=torch.zeros(self.s_agent.embed_dim,device=self.device)
        yield_sc,r_emb=self.r_agent.forward(safe_smi); r_emb=r_emb.detach()
        reward=-1. if not valid else self.reward_fn.compute(binding,yield_sc,admet,tox)
        self.p_agent.train()
        _,probs,val=self.p_agent.forward(g_emb,e_emb,s_emb,r_emb,reward=prev_reward)
        dist=torch.distributions.Categorical(probs); a=dist.sample()
        tr=Transition(float(dist.log_prob(a).item()),reward,val.squeeze(),probs.squeeze().detach())
        return smiles,reward,{\'binding\':binding,\'yield\':yield_sc,\'admet\':admet,\'toxicity\':tox},tr

    def _update_policy(self, transitions):
        rewards=torch.tensor([t.reward for t in transitions],dtype=torch.float,device=self.device)
        lps=torch.tensor([t.log_prob for t in transitions],dtype=torch.float,device=self.device)
        vals=torch.stack([t.value for t in transitions])
        probs_all=torch.stack([t.action_probs for t in transitions])
        ld=self.p_agent.compute_policy_loss(lps,rewards,vals,self.gamma,self.entropy_coeff,probs_all)
        self.policy_optim.zero_grad(); ld[\'total_loss\'].backward()
        nn.utils.clip_grad_norm_(self.p_agent.parameters(),0.5); self.policy_optim.step()
        return {k:float(v.item()) for k,v in ld.items()}

    def run_episode(self, ep_idx, max_steps=10):
        transitions=[]; ep_smiles=[]; ep_rewards=[]; prev_r=0.
        for _ in range(max_steps):
            s,r,_,tr=self._step(prev_r); transitions.append(tr); ep_smiles.append(s); ep_rewards.append(r); prev_r=r
            if is_valid_smiles(s) and r>self.best_reward: self.best_reward=r;self.best_smiles=s;logger.info(f"★ New best={r:.4f}  {s}")
        li=self._update_policy(transitions)
        pct=100.*sum(is_valid_smiles(s) for s in ep_smiles)/max(1,max_steps)
        metrics={\'episode\':ep_idx,\'avg_reward\':sum(ep_rewards)/len(ep_rewards),\'best_reward\':self.best_reward,
                 \'pct_valid\':pct,\'diversity\':diversity_score(ep_smiles),**li}
        logger.info(f"Ep {ep_idx:3d} | rwd={metrics[\'avg_reward\']:.3f} | best={self.best_reward:.3f} | "
                    f"valid={pct:.0f}% | div={metrics[\'diversity\']:.3f} | loss={li.get(\'total_loss\',0):.4f}")
        self.all_metrics.append(metrics); return metrics

    def train(self, num_episodes=50, steps_per_episode=10, save_csv=\'matmed_metrics.csv\'):
        logger.info(f"MATMED training: {num_episodes} episodes × {steps_per_episode} steps")
        for ep in range(1,num_episodes+1): self.run_episode(ep,steps_per_episode)
        save_metrics_csv(self.all_metrics, filepath=save_csv)
        logger.info(f"Done! Best reward={self.best_reward:.4f}  SMILES={self.best_smiles}")

print("✅ train_matmed.py written")
'''))

# ── Smoke test ────────────────────────────────────────────────────────────
cells.append(md("## 3. Quick Smoke Tests\nVerify individual agents load and produce output."))
cells.append(code("""\
from utils import set_seed, get_zinc_sample, is_valid_smiles, SMILESTokenizer
set_seed(42)
zinc = get_zinc_sample()
print(f"ZINC sample: {len(zinc)} molecules")
print("Samples:", zinc[:3])
"""))

cells.append(code("""\
# Test G-Agent
from generator_agent import GeneratorAgent, pretrain_generator
tok = SMILESTokenizer()
g = GeneratorAgent(tokenizer=tok)
print(f"G-Agent params: {sum(p.numel() for p in g.parameters()):,}")
pretrain_generator(g, zinc, num_epochs=3)
smiles_batch, embs = g.generate(batch_size=4, temperature=1.0)
print("\\nGenerated SMILES:")
for s in smiles_batch:
    print(f"  {'✓' if is_valid_smiles(s) else '✗'}  {s}")
print(f"Embedding shape: {embs.shape}")
"""))

cells.append(code("""\
# Test E-Agent
from evaluator_agent import EvaluatorAgent
e = EvaluatorAgent()
print(f"E-Agent params: {sum(p.numel() for p in e.parameters()):,}")
for smi in zinc[:3]:
    sc, emb = e.forward(smi)
    print(f"  binding={sc:.4f}  emb={emb.shape}  {smi[:35]}")
"""))

cells.append(code("""\
# Test S-Agent (fallback encoder, no network needed)
from safety_agent import SafetyAgent
s = SafetyAgent(use_chemberta=False)   # set True if network available
print(f"S-Agent embed_dim: {s.embed_dim}  params: {sum(p.numel() for p in s.parameters()):,}")
for smi in zinc[:3]:
    tox, admet, emb = s.forward(smi)
    print(f"  tox={tox:.4f}  admet={admet:.4f}  emb={emb.shape}  {smi[:35]}")
"""))

cells.append(code("""\
# Test R-Agent
from reaction_agent import ReactionAgent, compute_reaction_features
import numpy as np
r = ReactionAgent()
print(f"R-Agent params: {sum(p.numel() for p in r.parameters()):,}")
for smi in zinc[:3]:
    sc, emb = r.forward(smi)
    feats = compute_reaction_features(smi)
    print(f"  yield={sc:.4f}  feats={np.round(feats,3)}  {smi[:35]}")
"""))

cells.append(code("""\
# Test P-Agent
import torch
from policy_agent import PolicyAgent, ACTION_NAMES
p = PolicyAgent()
print(f"P-Agent params: {sum(p.numel() for p in p.parameters()):,}")
g_emb=torch.randn(256); e_emb=torch.randn(128); s_emb=torch.randn(128); r_emb=torch.randn(128)
logits, probs, val = p.forward(g_emb, e_emb, s_emb, r_emb, reward=0.42)
print(f"Action probs: {probs.squeeze().detach().numpy().round(4)}")
print(f"Action names: {ACTION_NAMES}")
print(f"Value est.: {val.item():.4f}")
"""))

# ── Full Training ─────────────────────────────────────────────────────────
cells.append(md("""## 4. Full MATMED Training
Run the complete multi-agent RL loop.  
Adjust `num_episodes` and `steps_per_episode` for shorter / longer runs.
- Set `use_chemberta=True` if you have internet access in Colab for a better S-Agent.
"""))

cells.append(code("""\
from reward import RewardConfig
from train_matmed import MATMEDRunner

reward_cfg = RewardConfig(alpha=1.0, beta=0.5, gamma=0.5, delta=1.0)

runner = MATMEDRunner(
    reward_config       = reward_cfg,
    num_pretrain_epochs = 5,
    lr_policy           = 3e-4,
    gamma               = 0.99,
    entropy_coeff       = 0.01,
    seed                = 42,
    use_chemberta       = False,   # flip to True if online
)

runner.train(
    num_episodes       = 30,
    steps_per_episode  = 8,
    save_csv           = 'matmed_metrics.csv',
)
"""))

# ── Results ───────────────────────────────────────────────────────────────
cells.append(md("## 5. Analyse Results"))
cells.append(code("""\
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv('matmed_metrics.csv')
print(df.tail(10).to_string(index=False))
"""))

cells.append(code("""\
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('MATMED Training Metrics', fontsize=14, fontweight='bold')

df['avg_reward'].plot(ax=axes[0,0], title='Avg Reward per Episode', color='steelblue')
df['best_reward'].plot(ax=axes[0,1], title='Best Reward (cumulative)', color='darkorange')
df['pct_valid'].plot(ax=axes[1,0], title='% Valid SMILES', color='green')
df['diversity'].plot(ax=axes[1,1], title='Chemical Diversity (Tanimoto)', color='purple')

for ax in axes.flat:
    ax.set_xlabel('Episode'); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('matmed_training.png', dpi=120)
plt.show()
print(f"\\nBest molecule: {runner.best_smiles}")
print(f"Best reward  : {runner.best_reward:.4f}")
"""))

# ── Molecule display ──────────────────────────────────────────────────────
cells.append(md("## 6. Visualise Best Molecule"))
cells.append(code("""\
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display

best = runner.best_smiles or 'CC(=O)Oc1ccccc1C(=O)O'
mol = Chem.MolFromSmiles(best)
if mol:
    img = Draw.MolToImage(mol, size=(400, 300))
    display(img)
    print(f"Best SMILES: {best}")
else:
    print("Could not draw molecule. Try a specific SMILES:")
    test = 'CC(=O)Oc1ccccc1C(=O)O'
    display(Draw.MolToImage(Chem.MolFromSmiles(test), size=(400,300)))
"""))

# ── Reward tuning ─────────────────────────────────────────────────────────
cells.append(md("""## 7. Experiment — Custom Reward Weights
Tune the reward coefficients to explore different drug-design goals.
"""))
cells.append(code("""\
# Example: emphasise binding affinity, penalise toxicity heavily
from reward import RewardFunction, RewardConfig

r_fn = RewardFunction(RewardConfig(alpha=2.0, beta=0.3, gamma=0.4, delta=2.0))
breakdown = r_fn.breakdown(
    binding_score=0.85,
    yield_score=0.60,
    admet_score=0.70,
    toxicity=0.10,
)
print("Reward breakdown:")
for k, v in breakdown.items():
    print(f"  {k:20s}: {v:.4f}")
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"name": "MATMED_Colab.ipynb", "provenance": []}
    },
    "cells": cells,
}

with open("matmed_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("✅  matmed_colab.ipynb generated successfully.")
'''))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"name": "MATMED_Colab.ipynb", "provenance": []}
    },
    "cells": cells,
}

import json
with open("/Users/srikanthreddy/.gemini/antigravity/scratch/matmed/matmed_colab.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook written.")
