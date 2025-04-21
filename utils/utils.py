import pandas as pd
import pickle
import os
from tqdm import tqdm
import torch


def get_protein_embedding(config, dataset):
    target_list = dataset['target'].unique()
    t_emb_dict = {}
    emb_idx = 1 if config['ProtEncoder']['prot_emb'] == 'ESM2_650M' \
        else 2 if config['ProtEncoder']['prot_emb'] == 'ESMC_300M' \
        else 3 if config['ProtEncoder']['prot_emb'] == 'ESMC_600M' \
        else None
    if emb_idx is None:
        raise ValueError('Invalid protein embedding type. Please check the config file.')

    print('Loading protein embeddings...')
    for target in tqdm(target_list):
        if os.path.exists(f'data/ProteinFeature/{target}.pkl'):
            with open(f'data/ProteinFeature/{target}.pkl', 'rb') as f:
                emb_f = pickle.load(f)
                t_emb_dict[target] = torch.tensor(emb_f[target][emb_idx])
        else:
            print(f'Protein embedding for {target} not found.')
            continue
    print('Total proteins:', len(t_emb_dict))
    return t_emb_dict