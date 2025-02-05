# Import
import os
from dataloader import BatchDataloader
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import trange, tqdm
import h5py
from torch.utils.data import TensorDataset, random_split, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import pgd_attack, train_loop, eval_loop, train_loop_apgd, eval_loop_apgd
import utils
import ecg_plot
from models import ResNet1d, ResNet1dGELU
import ast
import argparse
from warnings import warn

if __name__ == "__main__":

    # Arguments that will be saved in config file
    parser = argparse.ArgumentParser(add_help=True,
                                     description='Train model to predict rage from the raw ecg tracing.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='maximum number of epochs (default: 70)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for number generator (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--output', default='ptb_20/',
                        help='output folder (default: ./out)')
    parser.add_argument('--dataset', default='ptb_xl',
                        help='path to dataset (default: ptb_xl)')
    parser.add_argument('--adv_delay', type=int, default=10,
                        help='delay before adversarial training (default: 10)')
    parser.add_argument('--start_eps', type=float, default=0.001,
                        help = 'starting epsilon for adversarial training (default: 0.001)')
    parser.add_argument('--end_eps', type=float, default=0.01,
                        help = 'ending epsilon for adversarial training (default: 0.01)')
    
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write("Use device: {device:}\n".format(device=device))
    
    # Load the data
    dataset_path = args.dataset
    batch_size = args.batch_size

    path_to_csv, path_to_scp = dataset_path + '/ptbxl_database.csv', dataset_path + '/scp_statements.csv'
    path_to_train, path_to_val, path_to_test = dataset_path + '/train_age.h5', dataset_path + '/val_age.h5', dataset_path + '/test_age.h5'

    # Get labels
    df = pd.read_csv(path_to_csv, index_col='ecg_id')
    df = df[df['age'] != 300]

    validation_fold = 9
    test_fold = 10

    # Load labels
    train = df[(df.strat_fold != validation_fold) & (df.strat_fold != test_fold)]
    val = df[df.strat_fold == validation_fold]
    test = df[df.strat_fold == test_fold]

    # change this for other labels
    labels_train = train['age'].values
    labels_val = val['age'].values
    labels_test = test['age'].values

    # Make them torch tensors
    labels_train = torch.tensor(labels_train, dtype=torch.float32).reshape(-1,1)
    labels_val = torch.tensor(labels_val, dtype=torch.float32).reshape(-1,1)
    labels_test = torch.tensor(labels_test, dtype=torch.float32).reshape(-1,1)

    # Define traces
    traces_train = h5py.File(path_to_train, 'r')['tracings']
    traces_val = h5py.File(path_to_val, 'r')['tracings']
    traces_test = h5py.File(path_to_test, 'r')['tracings']

    # Define dataloaders
    train_dataloader = BatchDataloader(traces_train, labels_train, batch_size=batch_size)
    val_dataloader = BatchDataloader(traces_val, labels_val, batch_size=batch_size)
    test_dataloader = BatchDataloader(traces_test, labels_test, batch_size=batch_size)
    
    # hyperparameters
    learning_rate = args.lr
    weight_decay = 1e-2  
    num_epochs = args.epochs
    adversarial_delay = args.adv_delay # Do not perform adversarial training until epoch x
    start_eps = args.start_eps
    end_eps = args.end_eps
    loss_function = nn.MSELoss()

    # for exponential schedule
    eps_values = np.exp(np.linspace(np.log(start_eps), np.log(end_eps), num_epochs - adversarial_delay))
    
    pretrained_model_path = "models/code_model_10/latest.pth"
    output_model_path = args.output
    os.makedirs(output_model_path, exist_ok=True)

    is_finetuning = False
    
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    tqdm.write("Define model...")
    model = ResNet1dGELU(input_dim=(12, 4096), n_classes=1, blocks_dim=[(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)])#, kernel_size=3, dropout_rate=0.8

    if is_finetuning:
        tqdm.write("Load pretrained model...")
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

    model.to(device=device)
    tqdm.write("Done!\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    lr_scheduler = None #torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    tqdm.write("Training...")
    best_loss = np.Inf
    # allocation
    train_loss_all, valid_loss_all, adv_valid_loss_all = [], [], []
    mse_all, mae_all = [], []

    # loop over epochs
    for epoch in trange(1, num_epochs + 1):
        # training loop
        adversarial = False if epoch <= adversarial_delay else True
        adv_eps = eps_values[epoch - adversarial_delay - 1] if epoch > adversarial_delay else 0

        train_loss = train_loop_apgd(epoch, train_dataloader, model, optimizer, loss_function, device, adversarial=adversarial, adv_eps=adv_eps, adv_iters=10, adv_restarts=1)
        # validation loop
        valid_loss, y_pred, y_true = eval_loop_apgd(epoch, val_dataloader, model, loss_function, device)
        adv_valid_loss, adv_y_pred, adv_y_true = eval_loop_apgd(epoch, val_dataloader, model, loss_function, device, adversarial=True, adv_eps=end_eps, adv_iters=10, adv_restarts=1)

        # collect losses
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)
        adv_valid_loss_all.append(adv_valid_loss)

        # compute validation metrics for performance evaluation
        mse = mean_squared_error(y_true, y_pred)
        mse_all.append(mse)

        mae = mean_absolute_error(y_true, y_pred)
        mae_all.append(mae)

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
        tqdm.write('Epoch {epoch:2d}: \t'
                    'Train Loss {train_loss:.6f} \t'
                    'Valid Loss {valid_loss:.6f} \t'
                    'Adversarial Loss {adv_valid_loss:.6f} \t'
                    'MSE {mse:.6f} \t'
                'MAE {mae:.6f} \t'
                    '{model_save}'
                    .format(epoch=epoch,
                            train_loss=train_loss,
                            valid_loss=valid_loss,
                            adv_valid_loss=adv_valid_loss,
                            mse=mse,
                            mae=mae,
                            model_save=model_save_state))

        # Update learning rate with lr-scheduler
        if lr_scheduler:
            lr_scheduler.step(valid_loss)
    
    # Save the metrics to file together with the hyperparameters
    metrics = {'train_loss': train_loss_all,
            'valid_loss': valid_loss_all,
            'mse': mse_all,
            'mae': mae_all,
            'hyperparameters':
                {'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                    'num_epochs': num_epochs,
                    'batch_size': batch_size,
                    'adversarial_delay': adversarial_delay,
                    'adv_eps': adv_eps,
                    'adv_alpha': adv_eps/5,
                    'adv_steps': 10}}

    torch.save(metrics, output_model_path + '/metrics.pth')