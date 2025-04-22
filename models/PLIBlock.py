import torch
import torch.nn as nn

class IntraMolecularBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.use_ln = use_layernorm
        # LayerNorm for pre‑LN
        if use_layernorm:
            self.ln1_P = nn.LayerNorm(hidden_dim)
            self.ln2_P = nn.LayerNorm(hidden_dim)
            self.ln1_L = nn.LayerNorm(hidden_dim)
            self.ln2_L = nn.LayerNorm(hidden_dim)
        # 自注意力
        self.attn_P = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_L = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        # 2层 FFN
        self.ffn_P = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_L = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, P, L, mask_P=None, mask_L=None):
        # Protein self-attention
        res_P = P
        P_norm = self.ln1_P(P) if self.use_ln else P
        P_attn, _ = self.attn_P(
            P_norm, P_norm, P_norm,
            key_padding_mask=(~mask_P if mask_P is not None else None),
            need_weights=False
        )
        P = res_P + P_attn
        # Protein FFN
        res_P2 = P
        P_norm2 = self.ln2_P(P) if self.use_ln else P
        P_ffn = self.ffn_P(P_norm2)
        P = res_P2 + P_ffn

        # Ligand 自注意力
        res_L = L
        L_norm = self.ln1_L(L) if self.use_ln else L
        L_attn, _ = self.attn_L(
            L_norm, L_norm, L_norm,
            key_padding_mask=(~mask_L if mask_L is not None else None),
            need_weights=False
        )
        L = res_L + L_attn
        # Ligand FFN
        res_L2 = L
        L_norm2 = self.ln2_L(L) if self.use_ln else L
        L_ffn = self.ffn_L(L_norm2)
        L = res_L2 + L_ffn

        return P, L


class InterMolecularBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.use_ln = use_layernorm
        # LayerNorm for pre‑LN
        if use_layernorm:
            self.ln1_PL_L = nn.LayerNorm(hidden_dim)
            self.ln1_PL_P = nn.LayerNorm(hidden_dim)
            self.ln2_PL_L = nn.LayerNorm(hidden_dim)
            self.ln2_PL_P = nn.LayerNorm(hidden_dim)
        # 交叉注意力
        self.attn_PL = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_LP = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        # 2层 FFN
        self.ffn_L = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_P = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, P, L, mask_P=None, mask_L=None):
        # Protein->Ligand
        res_L = L
        L_norm = self.ln1_PL_L(L) if self.use_ln else L
        P_norm = self.ln1_PL_P(P) if self.use_ln else P
        L_attn, _ = self.attn_PL(
            L_norm, P_norm, P_norm,
            key_padding_mask=(~mask_P if mask_P is not None else None),
            need_weights=False
        )
        L = res_L + L_attn
        # FFN on Ligand
        res_L2 = L
        L_norm2 = self.ln2_PL_L(L) if self.use_ln else L
        L_ffn = self.ffn_L(L_norm2)
        L = res_L2 + L_ffn

        # Ligand->Protein
        res_P = P
        P_norm = self.ln1_PL_P(P) if self.use_ln else P
        L_norm_c = self.ln1_PL_L(L) if self.use_ln else L
        P_attn, _ = self.attn_LP(
            P_norm, L_norm_c, L_norm_c,
            key_padding_mask=(~mask_L if mask_L is not None else None),
            need_weights=False
        )
        P = res_P + P_attn
        # FFN on Protein
        res_P2 = P
        P_norm2 = self.ln2_PL_P(P) if self.use_ln else P
        P_ffn = self.ffn_P(P_norm2)
        P = res_P2 + P_ffn

        return P, L


class PLInteractionModel(nn.Module):
    """交替堆叠 intra- 和 inter- Block 构建 PLI 模块"""
    def __init__(self, hidden_dim, num_heads, num_layers,
                 dropout=0.1, use_layernorm=True):
        super().__init__()
        self.layers = num_layers
        # 分子内部 & 分子间 模块列表
        self.intra_blocks = nn.ModuleList([
            IntraMolecularBlock(hidden_dim, num_heads, dropout, use_layernorm)
            for _ in range(num_layers)
        ])
        self.inter_blocks = nn.ModuleList([
            InterMolecularBlock(hidden_dim, num_heads, dropout, use_layernorm)
            for _ in range(num_layers)
        ])
        # 池化
        self.pool = lambda x, mask: (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

    def forward(self, L, P, mask_L, mask_P):
        # L: [B, A, D], P: [B, R, D]
        for intra, inter in zip(self.intra_blocks, self.inter_blocks):
            P, L = intra(P, L, mask_P, mask_L)
            P, L = inter(P, L, mask_P, mask_L)
        # 最终池化
        lig_pli = self.pool(L, mask_L)
        prot_pli = self.pool(P, mask_P)
        return { 'lig_pli': lig_pli, 'prot_pli': prot_pli }
