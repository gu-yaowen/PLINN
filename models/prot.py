import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtSeq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prot_encoder = nn.ModuleList()
        in_dim  = config['ProtEncoder']['input_dim']
        hid_dim = config['ProtEncoder']['emb_dim']
        n_layer = config['ProtEncoder']['num_layer']
        for i in range(n_layer):
            dim_in = in_dim if i == 0 else hid_dim
            self.prot_encoder.append(nn.Linear(dim_in, hid_dim))

        self.dropout = nn.Dropout(config['ProtEncoder']['dropout_ratio'])
        self.layernorm = nn.LayerNorm(hid_dim) if config['ProtEncoder']['layernorm'] else None
        self.batchnorm = nn.BatchNorm1d(hid_dim) if config['ProtEncoder']['batchnorm'] else None

        # self-attention aggregation
        self.attn = nn.MultiheadAttention(hid_dim, config['ProtEncoder']['heads'], batch_first=True)
        # pooling with mask
        self.pool = lambda x, mask: torch.sum(x * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
        
    def forward(self, x):
        emb_P, mask_P = x.batch_prot, x.mask_P
        output = {}
        # emb_P: [B, R, D]
        for layer in self.prot_encoder:
            emb_P = layer(emb_P)           # still [B, R, D]
            if self.layernorm:
                emb_P = self.layernorm(emb_P)
            if self.batchnorm:
                # feed [B, D, R] into BatchNorm1d
                emb_P = emb_P.permute(0, 2, 1)
                emb_P = self.batchnorm(emb_P)
                emb_P = emb_P.permute(0, 2, 1)
            emb_P = F.relu(emb_P)
            emb_P = self.dropout(emb_P)
        global_out, _ = self.attn(emb_P, emb_P, emb_P, key_padding_mask=~mask_P)
        output['prot_global_rep'] = self.pool(emb_P, mask_P)
        output['prot_node_rep'] = emb_P
        output['mask_P'] = mask_P
        return output