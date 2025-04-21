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
import yaml
from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.model import PLINN
from train.loader import MoleculeDataset, batch_P
from train.prompt_optim import optimize_prompt_weight_ri as optimize_prompt_weight_ri_
from utils.scheduler import PolynomialDecayLR
from utils.utils import get_protein_embedding
import shutup; shutup.please()

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')  # Ensure reduction='mean'

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))  # RMSE already reduces to a scalar

def get_optimizer(model, lr_params):
    assert isinstance(lr_params, dict)

    pretrain_name, prompt_name, finetune_name = [], [], []
    for name, param in model.named_parameters():
        if 'gnn' in name or 'aggr' in name:
            pretrain_name.append(name)
        elif 'graph_pred_linear' in name:
            finetune_name.append(name)
        else:
            prompt_name.append(name)


    pretrain_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in pretrain_name, model.named_parameters()))))
    finetune_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in finetune_name, model.named_parameters()))))
    prompt_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in prompt_name, model.named_parameters()))))

    # Adam, (Adadelta), Adagrad, RAdam
    optimizer = torch.optim.Adam([
        {'params': finetune_params},
        {'params': pretrain_params, 'lr': float(lr_params['pretrain_lr'])},
        {'params': prompt_params, 'lr': float(lr_params['prompt_lr'])}
    ], lr=float(lr_params['finetune_lr']), weight_decay=float(lr_params['decay']))

    return optimizer


def get_dataloader(config):

    # Directly load pre-defined DataFrames
    df_train = pd.read_csv(config['dataset']['custom_train_path'])
    df_val = pd.read_csv(config['dataset']['custom_val_path'])
    df_test = pd.read_csv(config['dataset']['custom_test_path'])

    print(f"Train set size: {len(df_train)}, Validation set size: {len(df_val)}, Test set size: {len(df_test)}")    
    # Assume the first column is SMILES and the second column is labels
    # smiles_col = df_train.columns[0]  # First column
    # labels_col = df_train.columns[1]  # Second column
    smiles_col = 'SMILES'
    target_col = 'target'
    labels_col = 'y_cls'

    df = pd.concat([df_train, df_val, df_test], axis=0)
    prot_emb_dict = get_protein_embedding(config, df)

    dataset = MoleculeDataset(df_train[smiles_col], 
                              df_train[target_col],
                              df_train[[labels_col]],
                              config['dataset']['feat_type'])
    train_dataset = MoleculeDataset(df_train[smiles_col], 
                                    df_train[target_col],
                                    df_train[[labels_col]], 
                                    config['dataset']['feat_type'])
    val_dataset = MoleculeDataset(df_val[smiles_col],
                                  df_val[target_col],
                                  df_val[[labels_col]], 
                                  config['dataset']['feat_type'])
    test_dataset = MoleculeDataset(df_test[smiles_col],
                                   df_test[target_col], 
                                   df_test[[labels_col]], 
                                   config['dataset']['feat_type'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return dataset, train_loader, val_loader, test_loader, prot_emb_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval(model, val_loader, prot_emb_dict, config, metric='rmse'):
    assert metric in ['rmse', 'r2']
    model.eval()
    y_true, y_scores = [], []
    for step, batch_lig in enumerate(val_loader):
        batch_prot = batch_P(config, batch_lig, prot_emb_dict)
        batch_lig = batch_lig.to(config['device'])

        with torch.no_grad():
            predict, _ = model(batch_lig, batch_prot)

        y_true.append(batch_lig.label.view(predict.shape))
        y_scores.append(predict)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    score = np.mean(roc_list)

    return score


def train(model, train_loader, prot_emb_dict, criterion, optimizer, scheduler, config, channel_idx=-1):
    model.train()
    loss_history = []
    channel_weight = 0
    for idx, batch_lig in enumerate(train_loader):
        batch_prot = batch_P(config, batch_lig, prot_emb_dict)
        batch_lig.to(config['device'])
        predict, _ = model(batch_lig, batch_prot, channel_idx=channel_idx)
        label = batch_lig.label.view(predict.shape)

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            mask = label == 0  # nan entry
            loss = criterion(predict.double(), (label + 1) / 2) * (~mask)
            loss = loss.sum() / (~mask).sum()
        elif isinstance(criterion, nn.MSELoss):
            loss = criterion(predict, label)
            loss = loss.mean()
        elif isinstance(criterion, RMSELoss):  # Include RMSELoss here
            loss = criterion(predict, label)
            loss = loss.mean()
        elif isinstance(criterion, nn.L1Loss):  # Include MAE here
            loss = criterion(predict, label)
            loss = loss.mean()
        else:
            raise Exception("Unsupported loss function")

        optimizer.zero_grad()
        loss.backward()
        if config['optim']['gradient_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['optim']['gradient_clip'])
        optimizer.step()

        if config['optim']['scheduler'] == 'poly_decay':
            scheduler.step()

        loss_history.append(loss.item())

    channel_weight = channel_weight / len(train_loader)

    return np.mean(loss_history), channel_weight


def optimize_prompt_weight_ri(model, train_loader, val_loader, config, metric='euclidean', act='softmax', max_num=5000):
    temperature = config['model']['temperature']
    skip_bo = config['prompt_optim']['skip_bo']

    # Extract channel-wise embeddings for all training data
    num = 0
    model.eval()
    graph_rep_list, label_list = [], []
    for loader in [train_loader, val_loader]:
        if loader is None:
            continue
        for batch in loader:
            batch.to(config['device'])
            with torch.no_grad():
                graph_reps = []
                if model.backbone == 'gps':
                    h_g, node_repres = model.LigEncoder.gnn(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    h_g, node_repres = model.LigEncoder.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                # map back to batched nodes for aggregation
                batch_x, batch_mask = to_dense_batch(node_repres, batch.batch)

                # conditional aggregation given the prompt_inds
                for i in range(len(model.prompt_token)):
                    h_g, h_x, _ = model.LigEncoder.aggrs[i](batch_x, batch_mask)
                    if config['model']['normalize']:
                        h_g = F.normalize(h_g, dim=-1)
                    graph_reps.append(h_g)

            graph_reps_batch = torch.stack(graph_reps)
            labels_batch = batch.label.view(-1, model.num_tasks)

            is_valid = (labels_batch != 0).sum(-1) == labels_batch.size(1)
            graph_rep_list.append(graph_reps_batch[:, is_valid])
            label_list.append(labels_batch[is_valid])

            num += graph_rep_list[-1].size(1)
            if num > max_num:
                break

    graph_reps = torch.concat(graph_rep_list, dim=1).cpu()  # (num_prompt, N, emb_dim)
    labels = torch.concat(label_list, dim=0).cpu()  # (N, 1)

    return optimize_prompt_weight_ri_(graph_reps, labels, n_runs=50, n_inits=50, n_points=5, n_restarts=512,
                                      n_samples=512, temperature=temperature, metric=metric,
                                      skip_bo=skip_bo, verbose=config['verbose'])


def main(config):
    if not config['LigEncoder']['checkpoint']:
        config['LigEncoder']['use_prompt'] = False
    save_dir = config['save_dir']
    try:
        os.makedirs(save_dir, exist_ok=True)  # Creates all necessary directories
    except OSError as error:
        print(f"Error creating directory {save_dir}: {error}")
        
    # Setup model
    if config['dataset']['feat_type'] == 'basic':
        atom_feat_dim, bond_feat_dim = None, None
    elif config['dataset']['feat_type'] == 'rich':
        atom_feat_dim, bond_feat_dim = 143, 14
    elif config['dataset']['feat_type'] == 'super_rich':
        atom_feat_dim, bond_feat_dim = 170, 14
    else:
        raise NotImplementedError('Unrecognized feature type. Please choose from [basic/rich/super_rich].')

    # Main:
    #avg_auc_last, avg_auc_best = [], []

    best_initialization = None
    if config['prompt_optim']['inits']:
        best_initialization = torch.Tensor(config['prompt_optim']['inits'])

    # Setup dataset and dataloader
    print('Loading dataset and creating dataloader...')
    _, train_loader, val_loader, test_loader, prot_emb_dict = get_dataloader(config)
    # Setup model
    model = PLINN(config, atom_feat_dim, bond_feat_dim)

    if config['LigEncoder']['checkpoint']:
        print('Loading checkpoint from {}'.format(config['LigEncoder']['checkpoint']))
        stat_dict = torch.load(config['LigEncoder']['checkpoint'])['wrapper']
        revise_stat_dict = OrderedDict()
        for k, v in stat_dict.items():
            new_key = 'LigEncoder.' + k
            revise_stat_dict[new_key] = v
        model.load_state_dict(revise_stat_dict, strict=False)
        del revise_stat_dict, stat_dict
    else:
        print('No checkpoint provided. Initializing model from scratch.')
    model.to(config['device'])

    # Train prompt:
    # NOTE: unfinished
    if config['LigEncoder']['use_prompt']:
        if best_initialization is None:
            best_initialization = optimize_prompt_weight_ri(model, train_loader, val_loader, config)

        model.LigEncoder.set_prompt_weight(best_initialization.to(config['device']))

        # if args.verbose and use_prompt:
        initial_prompt_probs = model.LigEncoder.get_prompt_weight('softmax').data.cpu()
        initial_prompt_weights = model.LigEncoder.get_prompt_weight('none').data.cpu()
        print('Initial prompt weight:', initial_prompt_weights)
        print('Initial prompt prob:  ', initial_prompt_probs)

    # Setup optimizer
    optimizer = get_optimizer(model, config['optim'])
    scheduler = None
    if config['optim']['scheduler'] == 'cos_anneal':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0.0001)
    elif config['optim']['scheduler'] == 'poly_decay':
        scheduler = PolynomialDecayLR(optimizer, warmup_updates=config['epochs'] * len(train_loader) // 10,
                                        tot_updates=config['epochs'] * len(train_loader),
                                        lr=config['optim']['finetune_lr'], end_lr=1e-9, power=1)
    
    best_score, best_checkpoint = None, None  # Initialize here
    # Setup loss function
    if config['dataset']['task'] == 'regression':
        print('Initialize best score and best checkpoint for regression tasks.')
        best_score = float('inf')
        if config['dataset']['loss_func'] == 'MSE':
            criterion = nn.MSELoss(reduction='none')
        elif config['dataset']['loss_func'] == 'RMSE':
            criterion = RMSELoss()
        elif config['dataset']['loss_func'] == 'MAE':
            criterion = nn.L1Loss(reduction='none')
        
    elif config['dataset']['task'] == 'classification':
        print('Initialize best score and best checkpoint for classification tasks.')
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        best_score = -float('inf')
    else:
        raise NotImplementedError

    # Setup learnable parameters:
    model.freeze_aggr_module()

    # Setup random seed
    model_path = os.path.join(save_dir, f'model.pt')

    csv_filename = os.path.join(save_dir, f'scores.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "train_score", "val_score", "test_score"])  # Header row
    

    for epoch in tqdm(range(1, config['epochs'] + 1)):
        # train one epoch
        train(model, train_loader, prot_emb_dict, criterion, optimizer, scheduler, config)
        # evaluate validation
        score = eval(model, val_loader, prot_emb_dict, config)
        test_score = eval(model, test_loader, prot_emb_dict, config)
        train_score = eval(model, train_loader, prot_emb_dict, config)
        
        with open(csv_filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch, train_score, score, test_score])
            
        if config['optim']['scheduler'] == 'cos_anneal':
            scheduler.step()

        if config['verbose'] and config['LigEncoder']['use_prompt']:
            weight = model.LigEncoder.get_prompt_weight('softmax').data.cpu().numpy()
            cur_lr = optimizer.param_groups[-1]['lr']
            tqdm.write(
                f"[ep{epoch}] {score:>4.4f} {test_score:>4.4f} {cur_lr} [{weight[0]:>4.3f} {weight[1]:>4.3f} {weight[2]:>4.3f}]")
        elif config['verbose']:
            cur_lr = optimizer.param_groups[-1]['lr']
            tqdm.write(f"[ep{epoch}] {score:>4.4f} {test_score:>4.4f} {cur_lr}")
        
        if config['dataset']['task'] == 'regression':
            if score < best_score:
                best_score = score
                #best_checkpoint = copy.deepcopy(model.state_dict())
                #torch.save(best_checkpoint, model_path)
                torch.save({'wrapper': model.state_dict()}, model_path)
                print(f"Best model saved at epoch {epoch} with score {score:.4f}")
        elif config['dataset']['task'] == 'classification':
            if score > best_score:
                best_score = score
                #best_checkpoint = copy.deepcopy(model.state_dict())
                torch.save({'wrapper': model.state_dict()}, model_path)
                print(f"Best model saved at epoch {epoch} with score {score:.4f}")

    score_last_checkpoint = eval(model, test_loader, config)
    #avg_auc_last.append(score_last_checkpoint)
    if config['LigEncoder']['use_prompt']:
        print('Prompt weight of last checkpoint:', model.get_prompt_weight('softmax').data.cpu())

    #model.load_state_dict(best_checkpoint)
    score_best_checkpoint = eval(model, test_loader, config)
    #avg_auc_best.append(score_best_checkpoint)
    if config['LigEncoder']['use_prompt']:
        print('Prompt weight of best checkpoint:', model.get_prompt_weight('softmax').data.cpu())

    print('[Best score]: {:.4f}'.format(score_best_checkpoint))


if __name__ == '__main__':
    from args import add_args

    args = add_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    main(config)
