# Import
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
from torch.utils.data import TensorDataset, random_split, DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from utils import pgd_attack, train_loop, eval_loop, train_loop_apgd, eval_loop_apgd
from models import ResNet1d
import argparse
from warnings import warn
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ast

from dataloader import BatchDataloader, H5Dataset

if __name__ == "__main__":

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='maximum number of epochs (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for number generator (default: 42)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--scheduler', action='store_true',
                        help='use a scheduler for the learning rate')
    parser.add_argument("--patience", type=int, default=7,
                        help='maximum number of epochs without reducing the learning rate (default: 7)')
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help='minimum learning rate (default: 1e-7)')
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help='reducing factor for the lr in a plateu (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='weight decay (default: 1e-2)')
    parser.add_argument('--output', default='./out',
                        help='output folder (default: ./out)')
    parser.add_argument('--dataset', default='ptbxl',
                        help='dataset to use (default: ptbxl)')
    parser.add_argument('--dataset_path', default='./ptbxl',
                        help='path to dataset (default: ./ptbxl)')
    parser.add_argument('--pretrained_model', default=None,
                        help='path to pretrained model (default: None)')
    parser.add_argument('--adv_delay', type=int, default=10,
                        help='delay before adversarial training (default: 10)')
    parser.add_argument('--start_eps', type=float, default=0.001,
                        help = 'starting epsilon for adversarial training (default: 0.001)')
    parser.add_argument('--end_eps', type=float, default=0.01,
                        help = 'ending epsilon for adversarial training (default: 0.01)')
    parser.add_argument('--adv_steps', type=int, default=10,
                        help = 'number of steps for the adversarial attack (default: 10)')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device to use (default: 0)')
    parser.add_argument('--target_variable', choices=['AF', 'six', 'age'], default='AF',
                        help='the target variable to predict (default: AF). "six" stands for predicting the six labels in the CODE dataset')
    parser.add_argument('--freeze', action='store_true',
                        help='freeze all layers except the last one')
    
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    
    # Set device
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    tqdm.write("Using device: {device:}".format(device=device))
    
    # Load the data
    num_epochs = args.epochs
    seed = args.seed
    batch_size = args.batch_size
    learning_rate = args.lr
    use_scheduler = args.scheduler
    patience = args.patience
    min_lr = args.min_lr
    lr_factor = args.lr_factor
    weight_decay = args.weight_decay
    output_model_path = args.output
    pretrained_model_path = args.pretrained_model
    dataset = args.dataset
    dataset_path = args.dataset_path
    adversarial_delay = args.adv_delay
    start_eps = args.start_eps
    end_eps = args.end_eps
    adv_steps = args.adv_steps
    target_variable = args.target_variable
    freeze = args.freeze

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    path_to_train, path_to_val, path_to_test = dataset_path + '/train.h5', dataset_path + '/val.h5', dataset_path + '/test.h5'

    train_traces, val_traces, test_traces = None, None, None
    train_labels, val_labels, test_labels = None, None, None

    if dataset == 'ptbxl':
        path_to_csv = os.path.join(dataset_path, 'ptbxl_database.csv')
        path_to_scp = os.path.join(dataset_path, 'scp_statements.csv')

        df = pd.read_csv(path_to_csv, index_col='ecg_id')
        df = df[df['age'] != 300]
        df = df[df['age'] >= 18]

        # Get labels
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

        def has_key(dic, key_str):
            for key in dic.keys():
                if key == key_str:
                    return 1
            return 0
        
        if target_variable == 'six':
            df['1dAVb'] = df.scp_codes.apply(lambda x: has_key(x, '1AVB'))
            df['RBBB'] = df.scp_codes.apply(lambda x: has_key(x, 'CRBBB'))
            df['LBBB'] = df.scp_codes.apply(lambda x: has_key(x, 'CLBBB'))
            df['SB'] = df.scp_codes.apply(lambda x: has_key(x, 'SBRAD'))
            df['ST'] = df.scp_codes.apply(lambda x: has_key(x, 'STACH'))
            df['AF'] = df.scp_codes.apply(lambda x: has_key(x, 'AFIB'))
        elif target_variable == 'AF': 
            df['AF'] = df.scp_codes.apply(lambda x: has_key(x, 'AFIB'))
        # age is already in the dataframe

        validation_fold = 9
        test_fold = 10

        # Load labels
        train_df = df[(df.strat_fold != validation_fold) & (df.strat_fold != test_fold)]
        val_df = df[df.strat_fold == validation_fold]
        test_df = df[df.strat_fold == test_fold]

    elif dataset == 'code':
        path_to_csv = dataset_path + '/exams.csv'

        # Get labels
        df = pd.read_csv(path_to_csv, index_col='exam_id')

        # Get h5 file
        train_file = h5py.File(path_to_train, 'r')
        train_traces_ids = train_file['exam_id']

        val_file = h5py.File(path_to_val, 'r')
        val_traces_ids = val_file['exam_id']

        test_file = h5py.File(path_to_test, 'r')
        test_traces_ids = test_file['exam_id']

        # Only keep the traces in the csv that match the traces in the h5 file
        train_df = df[df.index.isin(train_traces_ids)]
        val_df = df[df.index.isin(val_traces_ids)]
        test_df = df[df.index.isin(test_traces_ids)]

        # Sort the dataframe in trace order
        train_df = train_df.reindex(train_traces_ids)
        val_df = val_df.reindex(val_traces_ids)
        test_df = test_df.reindex(test_traces_ids)

        # Invert values as to match PTB-XL where a "positive" sex corresponds to female
        # train_labels = 1 - train_labels
        # val_labels = 1 - val_labels

    else:
        raise ValueError("Dataset not recognized.")    
    
    # Depending on prediction target, we need to change the number of classes
    num_classes = 6 if target_variable == 'six' else 1

    if target_variable == 'six':
        train_labels = train_df[['1dAVb','RBBB','LBBB','SB','ST','AF']].values
        val_labels = val_df[['1dAVb','RBBB','LBBB','SB','ST','AF']].values
        test_labels = test_df[['1dAVb','RBBB','LBBB','SB','ST','AF']].values
    elif target_variable == 'AF':
        train_labels = train_df['AF'].values
        val_labels = val_df['AF'].values
        test_labels = test_df['AF'].values
    elif target_variable == 'age':
        train_labels = train_df['age'].values
        val_labels = val_df['age'].values
        test_labels = test_df['age'].values
    else:
        raise ValueError("Target variable not recognized.")

    # Make into torch tensor
    train_labels = torch.tensor(train_labels, dtype=torch.float32).reshape(-1,num_classes)
    val_labels = torch.tensor(val_labels, dtype=torch.float32).reshape(-1,num_classes)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).reshape(-1,num_classes)

    train_dataset = H5Dataset(path_to_train,'tracings',train_labels)
    val_dataset = H5Dataset(path_to_val,'tracings',val_labels)
    test_dataset = H5Dataset(path_to_test,'tracings',test_labels)

    # Make the datasets smaller for testing
    # train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    # val_dataset = torch.utils.data.Subset(val_dataset, range(1000))
    # test_dataset = torch.utils.data.Subset(test_dataset, range(1000))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if target_variable == 'age':
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.BCEWithLogitsLoss()

    # schedule the epsilon values of each epoch
    if num_epochs > adversarial_delay:
        eps_values = np.exp(np.linspace(np.log(start_eps), np.log(end_eps), num_epochs - adversarial_delay))
    
    os.makedirs(output_model_path, exist_ok=True)
    
    model = ResNet1d(input_dim=(12, 4096), n_classes=num_classes, blocks_dim=[(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)], activation_function=nn.GELU())

    if pretrained_model_path is not None:
        tqdm.write("Loading pretrained model")
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.lin.parameters():
                param.requires_grad = True

    model.to(device=device)
    tqdm.write("Model defined")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=patience, min_lr=min_lr)

    tqdm.write("Training model")
    best_loss = np.inf
    # allocation
    train_loss_all, valid_loss_all, adv_valid_loss_all = [], [], []
    if target_variable == 'age':
        mae_all, mse_all = [], []
    else:
        ap_all = []

    # loop over epochs
    for epoch in tqdm(range(1, num_epochs + 1)):
        # training loop
        adversarial = False if epoch <= adversarial_delay else True
        adv_eps = eps_values[epoch - adversarial_delay - 1] if epoch > adversarial_delay else 0

        # train_loss = train_loop(epoch, train_dataloader, model, optimizer, loss_function, device, adversarial=adversarial, adv_eps=adv_eps, adv_alpha=adv_eps/5, adv_steps=adv_steps)
        train_loss = train_loop_apgd(epoch, train_dataloader, model, optimizer, loss_function, device, adversarial=adversarial, adv_eps=adv_eps, adv_iters=10, adv_restarts=1)
        # validation loop
        valid_loss, y_pred, y_true = eval_loop_apgd(epoch, val_dataloader, model, loss_function, device, adversarial=False)
        adv_valid_loss, adv_y_pred, adv_y_true = eval_loop_apgd(epoch, val_dataloader, model, loss_function, device, adversarial=True, adv_eps=0.05, adv_iters=10, adv_restarts=1)

        # update learning rate
        if use_scheduler:
            scheduler.step(valid_loss)

        # collect losses
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)
        adv_valid_loss_all.append(adv_valid_loss)
        
        # compute validation metrics for performance evaluation    

        # compute AP for each class idependently
        # auroc = roc_auc_score(y_true, y_pred)
        if target_variable == 'age':
            mae = np.abs(y_true - y_pred).mean()
            mse = ((y_true - y_pred)**2).mean()

            mae_all.append(mae)
            mse_all.append(mse)

        else:
            # apply sigmoid to y_pred
            y_pred = torch.sigmoid(torch.tensor(y_pred)).numpy()
            ap = average_precision_score(y_true, y_pred, average=None)
            ap_all.append(ap)
        
        # y_pred = np.round(y_pred)
        
        # compute accuracy    
        # accuracy = accuracy_score(y_true, y_pred)
        # f1 = f1_score(y_true, y_pred, average='binary')
        
        # auroc_all.append(auroc)
        # accuracy_all.append(accuracy)
        # f1_all.append(f1)

        # # save best model: here we save the model only for the lowest validation loss
        if valid_loss < best_loss:
            # Save model parameters
            torch.save({'model': model.state_dict()}, output_model_path + '/best.pth') 
            # Update best validation loss
            best_loss = valid_loss
            # statement
            model_save_state = "Best model -> saved"
        else:
            model_save_state = ""
        
        torch.save({'model': model.state_dict()}, output_model_path + '/latest.pth') 

        # Print message
        if target_variable == 'age':
            tqdm.write(
            f'Epoch {epoch:2d}: \t'
            f'Train Loss {train_loss:.6f} \t'
            f'Valid Loss {valid_loss:.6f} \t'
            f'Adversarial Loss {adv_valid_loss:.6f} \t'
            f'MAE {mae:.6f} \t'
            f'MSE {mse:.6f} \t'
            f'{model_save_state}'
            )
        else:
            tqdm.write(
            f'Epoch {epoch:2d}: \t'
            f'Train Loss {train_loss:.6f} \t'
            f'Valid Loss {valid_loss:.6f} \t'
            f'Adversarial Loss {adv_valid_loss:.6f} \t'
            f'Average Precision {ap.mean():.6f} \t'
            f'{model_save_state}'
            )

    # Save the metrics to file together with the hyperparameters
    if target_variable == 'age':
        metrics = {
            'train_loss': train_loss_all,
            'valid_loss': valid_loss_all,
            'adv_valid_loss': adv_valid_loss_all,
            'mae': mae_all,
            'mse': mse_all,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'adversarial_delay': adversarial_delay,
                'adv_eps': adv_eps,
                'adv_steps': adv_steps
            }
        }
    else:
        metrics = {
            'train_loss': train_loss_all,
            'valid_loss': valid_loss_all,
            'adv_valid_loss': adv_valid_loss_all,
            'ap': ap_all,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'adversarial_delay': adversarial_delay,
                'adv_eps': adv_eps,
                'adv_steps': adv_steps
            }
        }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.number):
                return obj.item()
            return super().default(obj)

    with open(output_model_path + '/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
