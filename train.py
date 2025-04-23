import os
import csv
import random
import numpy as np
import pandas as pd
import yaml
import time
import shutup; shutup.please()
from collections import OrderedDict
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from sklearn.metrics import (
    mean_squared_error, r2_score, roc_auc_score,
    average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_absolute_error
)
from tqdm import tqdm
from models.model import PLINN
from utils.loader import MoleculeDataset, batch_P
from utils.utils import get_protein_embedding, filter_by_prot_length
from utils.prompt_optim import optimize_prompt_weight_ri_
from utils.scheduler import PolynomialDecayLR
os.environ["WANDB_MODE"]="offline"


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
    df_train['y_cls'] = df_train['y_cls'].map(lambda x: 1 if x>0 else -1)
    df_train = df_train.sample(frac=1, random_state=config['split_seed']).reset_index(drop=True)
    df_val   = pd.read_csv(config['dataset']['custom_val_path'])
    df_val['y_cls'] = df_val['y_cls'].map(lambda x: 1 if x>0 else -1)
    df_test  = pd.read_csv(config['dataset']['custom_test_path'])
    df_test['y_cls'] = df_test['y_cls'].map(lambda x: 1 if x>0 else -1)

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


def optimize_prompt_weight_ri(model, train_loader, val_loader, config, metric='euclidean', act='softmax', max_num=5000):
    temperature = config['model']['temperature']
    skip_bo = config['prompt_optim']['skip_bo']

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
                    h_g, node_repres = model.LigEncoder.gnn(batch.x, 
                                        batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
                else:
                    h_g, node_repres = model.LigEncoder.gnn(batch.x, 
                                        batch.edge_index, batch.edge_attr, batch.batch)
                # map back to batched nodes for aggregation
                batch_x, batch_mask = to_dense_batch(node_repres, batch.batch)
                # conditional aggregation given the prompt_inds
                for i in range(len(model.prompt_token)):
                    h_g, h_x, _ = model.aggrs[i](batch_x, batch_mask)
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


def eval_model(model: nn.Module, loader: DataLoader, prot_emb_dict: dict, 
               device: str, task: str, dataset: str):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'{dataset} Eval'):
            batch_lig = batch.to(device, non_blocking=True)
            batch_prot = batch_P({'device': device}, batch, prot_emb_dict)
            pred, _ = model(batch_lig, batch_prot)
            all_pred.append(pred.detach().cpu())
            all_true.append(batch_lig.label.view(pred.shape).cpu())
    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()

    metrics = {}
    if task == 'regression':
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2']   = r2_score(y_true, y_pred)
        metrics['mae']  = mean_absolute_error(y_true, y_pred)
        # Pearson Correlation Coefficient
        if len(y_true) > 1:
            metrics['pcc'] = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            metrics['pcc'] = 0.0
    else:
        y_label = (y_true + 1) / 2
        y_prob = torch.sigmoid(torch.from_numpy(y_pred)).numpy()
        y_pred_label = (y_prob >= 0.5).astype(int)
        metrics['auc']       = roc_auc_score(y_label, y_prob)
        metrics['aupr']      = average_precision_score(y_label, y_prob)
        metrics['accuracy']  = accuracy_score(y_label, y_pred_label)
        metrics['precision'] = precision_score(y_label, y_pred_label)
        metrics['recall']    = recall_score(y_label, y_pred_label)
        metrics['f1']        = f1_score(y_label, y_pred_label)        

        return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, prot_emb_dict: dict,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    scheduler, scaler, device: str, config: dict):
    model.train()
    loss_history = []
    # channel_weight = 0
    for batch in tqdm(loader, desc='Training'):
        batch = batch.to(device, non_blocking=True)
        batch_prot = batch_P({'device': device}, batch, prot_emb_dict)
        if scaler is not None:
            optimizer.zero_grad()
            with autocast():
                predict, _ = model(batch, batch_prot)
                label = batch.label.view(predict.shape).to(predict.dtype)
                loss = criterion(predict, (label + 1) / 2)
                loss = loss.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            predict, _ = model(batch, batch_prot)
            label = batch.label.view(predict.shape).to(predict.dtype)
            loss = criterion(predict, (label + 1) / 2)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            if config['optim']['gradient_clip'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config['optim']['gradient_clip'])
            optimizer.step()
    
        # if config['optim']['gradient_clip'] > 0:
        #     nn.utils.clip_grad_norm_(model.parameters(), config['optim']['gradient_clip'])
        if config['optim']['scheduler'] == 'poly_decay':
            scheduler.step()
        loss_history.append(loss.item())
    return np.mean(loss_history)


def main(config):
    set_seed(config.get('seed', 42))
    device = config['device']

    # Initialize Weights & Biases
    wandb.init(
        entity='yg3191',
        project='PLAIN',
        name='test',
        config=config
    ) # NOTE

    train_loader, val_loader, test_loader, prot_emb_dict = get_dataloaders(config)

    model = PLINN(config, *{
        'basic': (None, None), 'rich': (143, 14), 'super_rich': (170, 14)
    }[config['dataset']['feat_type']])

    wandb.watch(model, log='all', log_freq=100) # NOTE

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
    print(model)
    print('Number of parameters:', 
          sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 'M')

    best_initialization = None
    if config['prompt_optim']['inits']:
        best_initialization = torch.Tensor(config['prompt_optim']['inits'])
    if config['LigEncoder']['use_prompt']:
        if best_initialization is None:
            best_initialization = optimize_prompt_weight_ri(model, train_loader, val_loader, config)
        model.LigEncoder.set_prompt_weight(best_initialization.to(config['device']))
        # if args.verbose and use_prompt:
        initial_prompt_probs = model.LigEncoder.get_prompt_weight('softmax').data.cpu()
        initial_prompt_weights = model.LigEncoder.get_prompt_weight('none').data.cpu()
        print('Initial prompt weight:', initial_prompt_weights)
        print('Initial prompt prob:  ', initial_prompt_probs)

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
        header = ['epoch', 'train_loss']
        metric_list = ['rmse', 'r2', 'mae', 'pcc'] if config['dataset']['task']=='regression' else \
                        ['auc', 'aupr', 'accuracy', 'precision', 'recall', 'f1']
        header += [f"train/{m}" for m in metric_list]
        header += [f"val/{m}" for m in metric_list]
        header += [f"test/{m}" for m in metric_list]
        header += ['gpu/alloc_GB', 'gpu/max_alloc_GB']        
        writer.writerow(header)

    best_score = float('inf') if config['dataset']['task']=='regression' else -float('inf')
    
    scaler = GradScaler() if config['amp'] == True else None
    # resume_path = config.get('resume_from_checkpoint', None)
    # if resume_path and os.path.isfile(resume_path):
    #     ckpt = torch.load(resume_path, map_location=device)
    #     model.load_state_dict(ckpt['model_state_dict'])
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    #     scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    #     best_score = ckpt.get('best_score', best_score)
    #     start_epoch = ckpt.get('epoch', 0) + 1
    #     print(f"Resuming training from epoch {start_epoch}, best_score={best_score}")

    for epoch in range(1, config['epochs']+1):
        time1 = time.time()
        train_loss = train_one_epoch(model, train_loader, prot_emb_dict,
                                     criterion, optimizer, scheduler, scaler,
                                     device, config)
        train_metrics = eval_model(model, train_loader, prot_emb_dict, 
                                   device, config['dataset']['task'], 'Train')
        val_metrics  = eval_model(model, val_loader, prot_emb_dict, 
                                  device, config['dataset']['task'], 'Val')
        test_metrics = eval_model(model, test_loader, prot_emb_dict, 
                                  device, config['dataset']['task'], 'Test')

        # scheduler step for cosine
        if config['optim']['scheduler']=='cos_anneal':
            scheduler.step()
        print(optimizer.param_groups[0]['lr'])

        # GPU stats
        gpu_alloc = torch.cuda.memory_allocated(device) / (1024**3)
        gpu_max   = torch.cuda.max_memory_allocated(device) / (1024**3)

        # Log to wandb
        log_dict = {'train/loss': train_loss}
        log_dict.update({f'train/{k}': v for k, v in train_metrics.items()})
        log_dict.update({f'val/{k}': v for k, v in val_metrics.items()})
        log_dict.update({f'test/{k}': v for k, v in test_metrics.items()})
        log_dict['gpu/alloc_GB'] = gpu_alloc
        log_dict['gpu/max_alloc_GB'] = gpu_max
        wandb.log(log_dict, step=epoch)

        # CSV logging
        row = [epoch, f"{train_loss:.4f}"]
        row += [f"{train_metrics[m]:.4f}" for m in train_metrics]
        row += [f"{val_metrics[m]:.4f}"   for m in val_metrics]
        row += [f"{test_metrics[m]:.4f}"  for m in test_metrics]
        row += [f"{gpu_alloc:.3f}", f"{gpu_max:.3f}"]
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # update best
        current = val_metrics.get('rmse', val_metrics.get('auc'))
        train_cuurrent = train_metrics.get('rmse', train_metrics.get('auc'))
        test_current = test_metrics.get('rmse', test_metrics.get('auc'))
        is_best = (config['dataset']['task']=='regression' and current<best_score) or \
                  (config['dataset']['task']=='classification' and current>best_score)
        if is_best:
            best_score=current
        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score
        }
        torch.save(ckpt, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pt'))

        if is_best:
            torch.save(model.state_dict(), os.path.join(config['save_dir'],'best.pt'))

        if config['LigEncoder']['use_prompt']:
            print('Prompt weight of last checkpoint:', model.LigEncoder.get_prompt_weight('softmax').data.cpu())
        one_epoch_time = time.time() - time1
        approximate_time_to_end = (config['epochs'] - epoch) * one_epoch_time
        if config.get('verbose', False):
            print(f"Epoch {epoch}: loss={train_loss:.4f}, train={train_cuurrent:.4f}, "
                  f"val={current:.4f}, test={test_current:.4f}, "
                  f"Est. finish time (h): {approximate_time_to_end/3600:.2f}")


    print('Training complete. Best score:', best_score)


if __name__ == '__main__':
    import sys
    from args import add_args
    
    args = add_args()
    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)
        
    main(cfg)
