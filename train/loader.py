import os
import pickle
import pandas as pd
import scipy.sparse as sps
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch_geometric.transforms as T
from utils.data import *

class MoleculeDataset(Dataset):
    def __init__(self, data_smiles, data_target, data_labels, feat_type):
        self.feat_type = feat_type
        smiles = data_smiles
        target = data_target.values
        labels = data_labels
        #labels = labels.replace(0, -1)
        labels = labels.values
        
        # convert mol to graph with smiles validity filtering
        self.smiles, self.labels, self.mol_data, self.prot_data = [], [], [], []
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                data = eval('mol_to_graph_data_obj_{}'.format(feat_type))(mol)
                self.smiles.append(smi)
                self.labels.append(labels[i])
                self.mol_data.append(self.transform(data))
                self.prot_data.append(target[i])
        self.num_task = 1

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        graph = self.mol_data[idx]
        graph.label = torch.Tensor(self.labels[idx])
        graph.smi = self.smiles[idx]
        graph.prot = self.prot_data[idx]
        return graph
    

class batch_P(Dataset):
    def __init__(self, config, batch, t_emb_dict):
        self.batch_prot = [t_emb_dict[prot_id] for prot_id in batch.prot]
        self.batch_prot = pad_sequence(self.batch_prot, batch_first=True)
        self.batch_prot = self.batch_prot.to(config['device'])
        self.mask_P = torch.zeros(self.batch_prot.size(0), 
                                  self.batch_prot.size(1), 
                                  dtype=torch.bool).to(config['device'])
        for i, prot in enumerate(self.batch_prot):
            self.mask_P[i, :len(prot)] = True

    def __len__(self):
        return len(self.batch_prot)
    
    def __getitem__(self, idx):
        return self.batch_prot[idx], self.mask_P[idx]
