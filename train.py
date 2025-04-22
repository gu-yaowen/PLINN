import os
import csv
import random
import numpy as np
import pandas as pd
import yaml
import shutup; shutup.please()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from tqdm import tqdm

from models.model import PLINN
from utils.loader import MoleculeDataset, batch_P
from utils.utils import get_protein_embedding, filter_by_prot_length
from utils.scheduler import PolynomialDecayLR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


def get_optimizer(model: nn.Module, lr_params: dict):
    # group parameters for different LR
    pretrain, prompt, finetune = [], [], []
    for name, param in model.named_parameters():
        if 'gnn' in name or 'aggr' in name:
            pretrain.append(param)
        elif 'classifier' in name:
            finetune.append(param)
        else:
            prompt.append(param)
    return optim.Adam([
        {'params': finetune},
        {'params': pretrain, 'lr': lr_params['pretrain_lr']},
        {'params': prompt,   'lr': lr_params['prompt_lr']}
    ], lr=lr_params['finetune_lr'], weight_decay=lr_params['decay'])


def get_dataloaders(config: dict):
    df_train = pd.read_csv(config['dataset']['custom_train_path'])
    df_val   = pd.read_csv(config['dataset']['custom_val_path'])
    df_test  = pd.read_csv(config['dataset']['custom_test_path'])

    prot_emb_dict = get_protein_embedding(config, pd.concat([df_train, df_val, df_test]))
    df_train = filter_by_prot_length(df_train, prot_emb_dict)

    # use same feat_type for all
    args = dict(data_smiles=df_train['SMILES'], data_target=df_train['target'],
                data_labels=df_train[['y_cls']], feat_type=config['dataset']['feat_type'])
    train_ds = MoleculeDataset(**args)
    args.update(dict(data_smiles=df_val['SMILES'], data_target=df_val['target'], data_labels=df_val[['y_cls']]))
    val_ds   = MoleculeDataset(**args)
    args.update(dict(data_smiles=df_test['SMILES'], data_target=df_test['target'], data_labels=df_test[['y_cls']]))
    test_ds  = MoleculeDataset(**args)

    loader_args = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': config.get('num_workers', 8),
        'pin_memory': True,
        'prefetch_factor': 2,
    }
    train_loader = DataLoader(train_ds, **loader_args)
    # for validation/test no shuffle
    loader_args.update({'shuffle': False})
    val_loader   = DataLoader(val_ds, **loader_args)
    test_loader  = DataLoader(test_ds, **loader_args)

    return train_loader, val_loader, test_loader, prot_emb_dict


def eval_model(model: nn.Module, loader: DataLoader, prot_emb_dict: dict, device: str, task: str):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch_lig = batch.to(device, non_blocking=True)
            batch_prot = batch_P({'device': device}, batch, prot_emb_dict)
            pred, _ = model(batch_lig, batch_prot)
            all_pred.append(pred.detach().cpu())
            all_true.append(batch_lig.label.view(pred.shape).cpu())
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    if task == 'regression':
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        return {'rmse': rmse, 'r2': r2}
    else:
        auc = roc_auc_score(y_true, torch.sigmoid(torch.from_numpy(y_pred)).numpy())
        return {'auc': auc}


def train_one_epoch(model: nn.Module, loader: DataLoader, prot_emb_dict: dict,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    scheduler, device: str, config: dict):
    model.train()
    scaler = GradScaler()
    running_loss = 0.0
    for batch in tqdm(loader, desc='Train'):  # leave tqdm for monitoring
        batch_lig = batch.to(device, non_blocking=True)
        batch_prot = batch_P({'device': device}, batch, prot_emb_dict)

        optimizer.zero_grad()
        with autocast():
            pred, _ = model(batch_lig, batch_prot)
            label = batch_lig.label.view(pred.shape)
            loss = criterion(pred, label)
            loss = loss.mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config['optim']['gradient_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['optim']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()

        if config['optim']['scheduler'] == 'poly_decay':
            scheduler.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    return avg_loss


def main(config):
    set_seed(config.get('seed', 42))
    device = config['device']

    train_loader, val_loader, test_loader, prot_emb_dict = get_dataloaders(config)

    model = PLINN(config, *{
        'basic': (None, None), 'rich': (143, 14), 'super_rich': (170, 14)
    }[config['dataset']['feat_type']])
    model.to(device)

    optimizer = get_optimizer(model, config['optim'])
    if config['optim']['scheduler'] == 'cos_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6)
    else:
        scheduler = PolynomialDecayLR(
            optimizer, warmup_updates=config['epochs'] * len(train_loader) // 10,
            tot_updates=config['epochs'] * len(train_loader),
            lr=config['optim']['finetune_lr'], end_lr=1e-9)

    # loss setup
    if config['dataset']['task'] == 'regression':
        criterion = {'MSE': nn.MSELoss(reduction='none'),
                     'RMSE': RMSELoss(),
                     'MAE': nn.L1Loss(reduction='none')}[config['dataset']['loss_func']]
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    # logging
    os.makedirs(config['save_dir'], exist_ok=True)
    csv_path = os.path.join(config['save_dir'], 'scores.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_score', 'test_score'])

    best_score = float('inf') if config['dataset']['task']=='regression' else -float('inf')

    scaler = GradScaler()
    for epoch in range(1, config['epochs']+1):
        train_loss = train_one_epoch(model, train_loader, prot_emb_dict,
                                     criterion, optimizer, scheduler,
                                     device, config)
        val_metrics  = eval_model(model, val_loader, prot_emb_dict, device, config['dataset']['task'])
        test_metrics = eval_model(model, test_loader, prot_emb_dict, device, config['dataset']['task'])

        # scheduler step for cosine
        if config['optim']['scheduler']=='cos_anneal':
            scheduler.step()

        # save best
        current_score = val_metrics.get('rmse', val_metrics.get('auc', None))
        if ((config['dataset']['task']=='regression' and current_score<best_score) or
            (config['dataset']['task']=='classification' and current_score>best_score)):
            best_score = current_score
            torch.save(model.state_dict(), os.path.join(config['save_dir'],'best.pt'))

        # log
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}",
                             f"{current_score:.4f}", f"{test_metrics.get('rmse', test_metrics.get('auc',0)):.4f}"])  
        if config['verbose']:
            print(f"Epoch {epoch}: loss={train_loss:.4f}, val={current_score:.4f}, test={test_metrics}")

    print('Training complete. Best score:', best_score)


if __name__ == '__main__':
    import sys
    from args import add_args
    args = add_args()
    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
