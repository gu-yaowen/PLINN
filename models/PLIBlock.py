import torch
import torch.nn as nn
import torch.nn.functional as F

class PLInteractionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads,
                 dropout=0.1, use_batchnorm=False, use_layernorm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_bn = use_batchnorm
        self.use_ln = use_layernorm

        # Norm layers
        if use_layernorm:
            self.ln_P = nn.LayerNorm(hidden_dim)
            self.ln_L = nn.LayerNorm(hidden_dim)
        if use_batchnorm:
            self.bn_P = nn.BatchNorm1d(hidden_dim)
            self.bn_L = nn.BatchNorm1d(hidden_dim)

        # Self-attention for protein and ligand
        self.attn_P = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_L = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention: protein->ligand and ligand->protein
        self.attn_PL = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_LP = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward networks
        def make_ffn():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
        self.ffn_P = make_ffn()
        self.ffn_L = make_ffn()

    def forward(self, single_P, single_L, mask_P=None, mask_L=None):
        """
        single_P: [B, R, D]
        single_L: [B, A, D]
        mask_P: [B, R] True for valid, False for pad
        mask_L: [B, A] True for valid
        """
        # Protein self-attention
        P = single_P
        if self.use_ln:
            P = self.ln_P(P)
        if self.use_bn:
            P = self.bn_P(P.transpose(1,2)).transpose(1,2)
        P2, _ = self.attn_P(P, P, P, key_padding_mask=~mask_P if mask_P is not None else None)
        P = P + P2
        P = P + self.ffn_P(P)

        # Ligand self-attention
        L = single_L
        if self.use_ln:
            L = self.ln_L(L)
        if self.use_bn:
            L = self.bn_L(L.transpose(1,2)).transpose(1,2)
        L2, _ = self.attn_L(L, L, L, key_padding_mask=~mask_L if mask_L is not None else None)
        L = L + L2
        L = L + self.ffn_L(L)

        # Protein->Ligand cross-attention
        L2, _ = self.attn_PL(L, P, P,
                             key_padding_mask=~mask_P if mask_P is not None else None)
        L = L + L2
        L = L + self.ffn_L(L)

        # Ligand->Protein cross-attention
        P2, _ = self.attn_LP(P, L, L,
                             key_padding_mask=~mask_L if mask_L is not None else None)
        P = P + P2
        P = P + self.ffn_P(P)

        return P, L


class PLInteractionModel(nn.Module):
    def __init__(self, hidden_dim, num_heads,
                 num_layers, dropout=0.1, use_batchnorm=False, use_layernorm=True):
        super().__init__()
        # stack of interaction blocks
        self.blocks = nn.ModuleList([
            PLInteractionBlock(hidden_dim, num_heads, dropout, use_batchnorm, use_layernorm)
            for _ in range(num_layers)
        ])
        # prediction head
        self.pool = lambda x, mask: (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    def forward(self, L, P, mask_L, mask_P):
        """
        L: [B, A, N1]
        P: [B, R, N2]
        mask_L: [B, A] True for valid atoms
        mask_P: [B, R] True for valid residues
        """
        output = {}
        # apply stacked blocks
        for block in self.blocks:
            P, L = block(P, L, mask_P, mask_L)

        # pooling with masks
        P_pool = self.pool(P, mask_P)  # [B, D]
        L_pool = self.pool(L, mask_L)  # [B, D]
        LP = torch.cat([P_pool, L_pool], dim=-1)  # [B, 2D]
        output['lig_prot'] = LP
        output['lig_pli'] = self.pool(L, mask_L)
        output['prot_pli'] = self.pool(P, mask_P)
        return output