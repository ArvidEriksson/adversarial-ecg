# Import
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import trange, tqdm
import h5py
from torch.utils.data import TensorDataset, random_split, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from utils import pgd_attack, train_loop, eval_loop, train_loop_apgd, eval_loop_apgd
import utils
from models import ResNet1d
import argparse
from warnings import warn
import json

from dataloader import BatchDataloader

if __name__ == "__main__":

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='maximum number of epochs (default: 20)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='weight decay (default: 1e-2)')
    parser.add_argument('--output', default='./out',
                        help='output folder (default: ./out)')
    parser.add_argument('--dataset', default='ptb-xl',
                        help='dataset to use (default: ptb-xl)')
    parser.add_argument('--dataset_path', default='./ptb-xl',
                        help='path to dataset (default: ./ptb-xl)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='use fine tuning (default: False)')
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
    
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write("Use device: {device:}\n".format(device=device))
    
    # Load the data
    num_epochs = args.epochs
    seed = args.seed
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    output_model_path = args.output
    finetuning = args.fine_tune
    pretrained_model_path = args.pretrained_model
    dataset = args.dataset
    dataset_path = args.dataset_path
    adversarial_delay = args.adv_delay
    start_eps = args.start_eps
    end_eps = args.end_eps
    adv_steps = args.adv_steps

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    path_to_train, path_to_val, path_to_test = dataset_path + '/train.h5', dataset_path + '/val.h5', dataset_path + '/test.h5'

    train_traces, val_traces, test_traces = None, None, None
    train_labels, val_labels, test_labels = None, None, None

    if dataset == 'ptb-xl':
        path_to_csv, path_to_scp = dataset_path + '/ptbxl_database.csv', dataset_path + '/scp_statements.csv'

        # Get labels
        df = pd.read_csv(path_to_csv, index_col='ecg_id')
        # df = df[df['age'] != 300] filter out the 300 when doing age prediction

        validation_fold = 9
        test_fold = 10

        # Load labels
        train = df[(df.strat_fold != validation_fold) & (df.strat_fold != test_fold)]
        val = df[df.strat_fold == validation_fold]
        test = df[df.strat_fold == test_fold]

        # change this for other labels
        train_labels = train['afib'].values
        val_labels = val['afib'].values
        test_labels = test['afib'].values

        # Define traces
        traces_train = h5py.File(path_to_train, 'r')['tracings']
        traces_val = h5py.File(path_to_val, 'r')['tracings']
        traces_test = h5py.File(path_to_test, 'r')['tracings']


    if dataset == 'code':
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

        # Define traces
        train_traces = train_file['tracings']
        val_traces = val_file['tracings']
        test_traces = test_file['tracings']

        # Sort the dataframe in trace order
        train_df = train_df.reindex(train_traces_ids)
        val_df = val_df.reindex(val_traces_ids)
        test_df = test_df.reindex(test_traces_ids)

        # Get labels
        train_labels = train_df['AF'].values
        val_labels = val_df['AF'].values
        test_labels = test_df['AF'].values

        # Invert values as to match PTB-XL where a "positive" sex corresponds to female
        # train_labels = 1 - train_labels
        # val_labels = 1 - val_labels

    else:
        raise ValueError("Dataset not recognized.")

    # Make into torch tensor
    train_labels = torch.tensor(train_labels, dtype=torch.float32).reshape(-1,1)
    val_labels = torch.tensor(val_labels, dtype=torch.float32).reshape(-1,1)
    test_labels = torch.tensor(test_labels, dtype=torch.float32).reshape(-1,1)

    # Define dataloaders
    train_dataloader = BatchDataloader(train_traces, train_labels, batch_size=batch_size)
    val_dataloader = BatchDataloader(val_traces, val_labels, batch_size=batch_size)
    test_dataloader = BatchDataloader(test_traces, test_labels, batch_size=batch_size)

    loss_function = nn.BCELoss()

    # schedule the epsilon values of each epoch
    eps_values = np.exp(np.linspace(np.log(start_eps), np.log(end_eps), num_epochs - adversarial_delay))
    
    os.makedirs(output_model_path, exist_ok=True)
    
    tqdm.write("Defining model...")
    model = ResNet1d(input_dim=(12, 4096), n_classes=1, blocks_dim=[(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)], activation_function=nn.GELU)#, kernel_size=3, dropout_rate=0.8

    if finetuning:
        tqdm.write("Load pretrained model...")
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    model.to(device=device)
    tqdm.write("Done!\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='e

    tqdm.write("Training...")
    best_loss = np.Inf
    # allocation
    train_loss_all, valid_loss_all, adv_valid_loss_all = [], [], []
    auroc_all, ap_all, accuracy_all, f1_all = [], [], [], []

    # loop over epochs
    for epoch in trange(1, num_epochs + 1):
        # training loop
        adversarial = False if epoch <= adversarial_delay else True
        adv_eps = eps_values[epoch - adversarial_delay - 1] if epoch > adversarial_delay else 0
        
        train_loss = train_loop_apgd(epoch, train_dataloader, model, optimizer, loss_function, device, adversarial=adversarial, adv_eps=adv_eps, adv_iters=10, adv_restarts=1)
        # validation loop
        valid_loss, y_pred, y_true = eval_loop(epoch, val_dataloader, model, loss_function, device)
        adv_valid_loss, adv_y_pred, adv_y_true = eval_loop(epoch, val_dataloader, model, loss_function, device, adversarial=True)

        # collect losses
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)
        adv_valid_loss_all.append(adv_valid_loss)
        
        # apply sigmoid to y_pred
        y_pred = torch.sigmoid(torch.tensor(y_pred)).numpy()

        # compute validation metrics for performance evaluation    
        auroc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        
        y_pred = np.round(y_pred)
        
        # compute accuracy    
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        
        auroc_all.append(auroc)
        ap_all.append(ap)
        accuracy_all.append(accuracy)
        f1_all.append(f1)

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
        tqdm.write(
            f'Epoch {epoch:2d}: \t'
            f'Train Loss {train_loss:.6f} \t'
            f'Valid Loss {valid_loss:.6f} \t'
            f'Adversarial Loss {adv_valid_loss:.6f} \t'
            f'AUROC {auroc:.6f} \t'
            f'Accuracy {accuracy:.6f} \t'
            f'F1 {f1:.6f} \t'
            f'Average Precision {ap:.6f} \t'
            f'{model_save_state}'
        )

        # Update learning rate with lr-scheduler
        # if lr_scheduler:
        #     lr_scheduler.step(valid_loss)

    # Save the metrics to file together with the hyperparameters
    metrics = {
        'train_loss': train_loss_all,
        'valid_loss': valid_loss_all,
        'adv_valid_loss': adv_valid_loss_all,
        'auroc': auroc_all,
        'ap': ap_all,
        'accuracy': accuracy_all,
        'f1': f1_all,
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
    with open(output_model_path + '/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)