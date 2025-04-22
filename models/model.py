import torch
import torch.nn as nn
import torch.nn.functional as F
from models.prot import ProtSeq
from models.molmcl.lig import LigGNN
from models.PLIBlock import PLInteractionModel

class PLINN(nn.Module):
    def __init__(self, config, lig_atom_feat_dim, lig_bond_feat_dim):
        super(PLINN, self).__init__()
        self.LigEncoder = LigGNN(
                                num_layer=config['LigEncoder']['num_layer'],
                                emb_dim=config['LigEncoder']['emb_dim'],
                                num_tasks=1, # currently only supporting one task
                                normalize=config['LigEncoder']['normalize'],
                                atom_feat_dim=lig_atom_feat_dim,
                                bond_feat_dim=lig_bond_feat_dim,
                                drop_ratio=config['LigEncoder']['dropout_ratio'],
                                attn_drop_ratio=config['LigEncoder']['attn_dropout_ratio'],
                                temperature=config['LigEncoder']['temperature'],
                                use_prompt=config['LigEncoder']['use_prompt'],
                                model_head=config['LigEncoder']['heads'],
                                layer_norm_out=config['LigEncoder']['layernorm'], 
                                backbone=config['LigEncoder']['backbone']
                            )
        
        config['ProtEncoder']['input_dim'] = 1280 if config['ProtEncoder']['prot_emb'] == 'ESM2_650M' \
                    else 960 if config['ProtEncoder']['prot_emb'] == 'ESMC_300M' \
                    else 1152 if config['ProtEncoder']['prot_emb'] == 'ESMC_600M' else None
        
        self.ProtEncoder = ProtSeq(
                                config
                            )
        self.PLI = PLInteractionModel(num_layers=config['PLI']['num_layer'],
                                      hidden_dim=config['PLI']['emb_dim'],
                                      num_heads=config['PLI']['heads'],
                                      dropout=config['PLI']['dropout_ratio'],
                                      use_layernorm=config['PLI']['layernorm']
                                     )
        
        pred_input_dim = config['LigEncoder']['emb_dim'] \
                        + config['ProtEncoder']['emb_dim'] \
                        + config['PLI']['emb_dim'] * 2
        # pred_input_dim = config['LigEncoder']['emb_dim'] \
        #                 + config['ProtEncoder']['emb_dim']
        # pred_input_dim = config['LigEncoder']['emb_dim']
        self.classifier = nn.Sequential(
            nn.Linear(pred_input_dim, int(pred_input_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(pred_input_dim / 2), 1),
        )

    def freeze_aggr_module(self):
        for param in self.LigEncoder.aggrs.parameters():
            param.requires_grad = False

    def forward(self, batch_lig, batch_prot, channel_idx=-1):
        # get ligand embedding
        output = self.LigEncoder(batch_lig, channel_idx=channel_idx)

        # get protein embedding
        prot_output = self.ProtEncoder(batch_prot)
        # combine two dict
        output.update(prot_output)

        pli_output = self.PLI(
                            output['lig_node_rep'],    # L: [B, A, D]
                            output['prot_node_rep'],   # P: [B, R, D]
                            output['mask_L'],          # mask_L: [B, A]
                            output['mask_P']           # mask_P: [B, R]
                        )

        output.update(pli_output)

        # concate all the embeddings
        pli_emb = torch.cat([output['lig_graph_rep'], output['prot_global_rep'], 
                             output['lig_pli'], output['prot_pli']], dim=1)
        # pli_emb = torch.cat([lig_graph_rep, prot_output_rep], dim=1)
        
        # apply classifier
        pred = self.classifier(pli_emb)
        return pred, output
