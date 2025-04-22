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

def filter_by_prot_length(dataset, prot_emb_dict, max_len=2000):
    """
    Filter the dataset by protein length.
    :param dataset: The dataset to filter.
    :param prot_emb_dict: The protein embedding dictionary.
    :param max_len: The maximum length of the protein sequence.
    :return: The filtered dataset.
    """
    t_list = dataset['target'].unique()
    filter_t = []
    for t in t_list:
        if len(prot_emb_dict[t][0]) > max_len:
            filter_t.append(t)
    filtered_dataset = dataset[~dataset['target'].isin(filter_t)].reset_index(drop=True)
    print(f'Filtered out {len(filter_t)} targets with protein length > {max_len}')
    return filtered_dataset