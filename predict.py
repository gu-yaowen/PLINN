import os
import sys
import pickle
import copy
import csv
import random
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
import argparse
import yaml
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_dense_batch
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from molmcl.finetune.loader import MoleculeDataset
##################################################
from molmcl.finetune.loader_customized import MoleculeDataset_cm
##################################################
from molmcl.finetune.model import GNNPredictor
from molmcl.finetune.prompt_optim import optimize_prompt_weight_ri as optimize_prompt_weight_ri_
from molmcl.splitters import scaffold_split, moleculeace_split
from molmcl.utils.scheduler import PolynomialDecayLR

def get_dataloader(data_path):
    df = pd.read_csv(data_path)
        
    # Assume the first column is SMILES and the second column is labels
    smiles_col = df.columns[0]  # First column
    labels_col = df.columns[1]  # Second column

    dataset = MoleculeDataset_cm(df[smiles_col], df[[labels_col]], config['dataset']['feat_type'])

    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    return dataset, data_loader

def main(config):
    save_dir = config['save_dir']
    try:
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    except OSError as error:
        print(f"Error creating directory {save_dir}: {error}")

    for data_path in config['dataset']['data_paths']:  # Process each file separately
        
        dataset, data_loader = get_dataloader(data_path)

        if config['dataset']['feat_type'] == 'basic':
            atom_feat_dim, bond_feat_dim = None, None
        elif config['dataset']['feat_type'] == 'rich':
            atom_feat_dim, bond_feat_dim = 143, 14
        elif config['dataset']['feat_type'] == 'super_rich':
            atom_feat_dim, bond_feat_dim = 170, 14
        else:
            raise NotImplementedError('Unrecognized feature type. Please choose from [basic/rich/super_rich].')

        # Setup model
        model = GNNPredictor(num_layer=config['model']['num_layer'],
                             emb_dim=config['model']['emb_dim'],
                             num_tasks=dataset.num_task,
                             normalize=config['model']['normalize'],
                             atom_feat_dim=atom_feat_dim,
                             bond_feat_dim=bond_feat_dim,
                             drop_ratio=config['model']['dropout_ratio'],
                             attn_drop_ratio=config['model']['attn_dropout_ratio'],
                             temperature=config['model']['temperature'],
                             use_prompt=config['model']['use_prompt'],
                             model_head=config['model']['heads'],
                             layer_norm_out=config['model']['layernorm'], 
                             backbone=config['model']['backbone'])

        if config['model']['checkpoint']:
            print(f'Loading checkpoint from {config["model"]["checkpoint"]}')
            model.load_state_dict(torch.load(config['model']['checkpoint'])['wrapper'], strict=False)
        
        model.to(config['device'])
        model.eval()

        y_true, y_scores = [], []
        for step, batch in enumerate(data_loader):
            batch = batch.to(config['device'])
            with torch.no_grad():
                predict = model(batch)['predict']

            y_true.append(batch.label.view(predict.shape))
            y_scores.append(predict)

        # Convert to NumPy
        y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
        y_true = torch.cat(y_true, dim=0).cpu().numpy()

        # Generate output filename for each input file
        data_file = data_path.split('/')[-1].split('.')[0]  # Extract filename without extension
        model_name = config['model']['checkpoint'].split('/')[-1].split('.')[0]
        df_results = pd.DataFrame({'true': y_true.flatten(), 'pred': y_scores.flatten()})

        output_path = os.path.join(save_dir, f'{data_file}_{model_name}_predictions.csv')
        df_results.to_csv(output_path, index=False)

        print(f"Predictions for {data_path} saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        raise Exception('Number of arguments is wrong.')

    with open(sys.argv[1], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if len(sys.argv) == 3:
        config['dataset']['feat_type'] = sys.argv[2]
        
    print(config)
    main(config)