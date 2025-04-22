import os
import yaml
import argparse

def add_args():
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    argparser.add_argument('--mode_type', type=str, default='train',
                            choices=['train', 'inference', 'retrain', 'finetune'],
                            help='Mode to run script in')
    argparser.add_argument('--task_type', type=str, choices=['classification', 'regression'],
                            help='Task type')
    argparser.add_argument('--train_data_path', type=str, default='temp_train.csv',
                            help='Path to CSV file containing training data')
    argparser.add_argument('--val_data_path', type=str, default='temp_val.csv',
                            help='Path to CSV file containing validation data')
    argparser.add_argument('--test_data_path', type=str, default='temp_test.csv',
                            help='Path to CSV file containing test data')
    argparser.add_argument('--model_path', type=str, default=None,
                            help='Path to model checkpoint (.pt file) for inference, retrain, or finetune')
    argparser.add_argument('--save_path', type=str, default=None,
                            help='dir name in exp_results folder where predictions will be saved')
    argparser.add_argument('--seed', type=int, default=42,
                            help='Random seed for reproducibility')
    argparser.add_argument('--gpu', type=int, default=1,
                            # choices=list(range(torch.cuda.device_count())),
                            help='Which GPU to use')
    # training arguments
    argparser.add_argument('--batch_size', type=int, default=32,
                            help='Batch size for training')
    argparser.add_argument('--epochs', type=int, default=100,
                            help='Number of epochs to train for')
    argparser.add_argument('--lr', type=float, default=0.0001,
                            help='Learning rate for finetuning')
    # model arguments
    argparser.add_argument('--prot_emb', type=str, default='ESM2_650M',
                            help='Pre-trained protein embedding model')
    argparser.add_argument('--num_layer', type=int, default=5,
                            help='Number of layers in the model')
    argparser.add_argument('--emb_dim', type=int, default=300,
                            help='Embedding dimension of the model')
    argparser.add_argument('--heads', type=int, default=6,
                            help='Number of attention heads in the model')
    argparser.add_argument('--layernorm', type=bool, default=True,
                            help='Whether to use layer normalization in the model')
    argparser.add_argument('--dropout_ratio', type=float, default=0,
                            help='Dropout ratio in the model')
    args = argparser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    args.config_path = os.path.join(args.save_path, "config.yaml")
    generate_yaml_config(args, output_path=args.config_path)
    return args


def generate_yaml_config(args, 
                         data_name="###", split_type="random",
                         loss_func="MAE", # there are more loss functions (MAE/RMSE/MSE); for classification, default loss: BCEWithLogitsLoss
                         checkpoint="./checkpoint/zinc-gps_best.pt", # download the pre-trained model from the MolMCL github
                         output_path=None):
    config = {
        "mode_type": args.mode_type,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "device": "cuda" if args.gpu else "cpu",
        "seed": args.seed, 
        "num_run": 1,
        "verbose": True,
        "split_seed": 0,
        "save_dir": args.save_path,
        "LigEncoder": {
            "backbone": "gps",
            "num_layer": 5,
            "emb_dim": 300,
            "heads": 6,
            "layernorm": True,
            "dropout_ratio": 0,
            "attn_dropout_ratio": 0.3,
            "temperature": 0.5,
            "use_prompt": True,
            "normalize": False,
            "checkpoint": checkpoint
        },
        "ProtEncoder": {
            "prot_emb": args.prot_emb,
            "input_dim": 1280 if args.prot_emb == 'ESM2_650M' \
                else 960 if args.prot_emb == 'ESMC_300M' \
                else 1152 if args.prot_emb == 'ESMC_600M' else None,
            "emb_dim": 300,
            "num_layer": args.num_layer,
            "heads": args.heads,
            "layernorm": args.layernorm,
            "dropout_ratio": args.dropout_ratio,
        },
        "PLI": {
            "num_layer": args.num_layer,
            "emb_dim": 300,
            "heads": args.heads,
            "layernorm": args.layernorm,
            "dropout_ratio": args.dropout_ratio,
        },
        "optim": {
            "prompt_lr": 0.0005,
            "pretrain_lr": 0.0005,
            "finetune_lr": args.lr,
            "heads": 6,
            "decay": 1e-6,
            "gradient_clip": 5,
            "scheduler": "cos_anneal"
        },
        "prompt_optim": {
            "skip_bo": True,
            "inits": [0.0000, 0.0000, 0.0000]
        },
        "dataset": {
            "data_dir": None,
            "data_name": data_name,
            "split_type": 'customized',
            "custom_train_path": args.train_data_path,
            "custom_val_path": args.val_data_path,
            "custom_test_path": args.test_data_path,
            "num_workers": 0,
            "feat_type": "super_rich",
            "task": args.task_type,
            "loss_func": loss_func
        }
    }
    
    with open(output_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    
    print(f"YAML configuration file saved to {output_path}")
