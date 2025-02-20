import os
import torch
from tqdm import tqdm
import pandas as pd
import h5py
from dataloader import MultExamH5Dataset
from torch.utils.data import DataLoader
from models import ResNet1d
from torchsummary import summary

device = torch.device(
    'mps'
    if torch.backends.mps.is_available()
    else 'cuda' if torch.cuda.is_available() else 'cpu'
)
tqdm.write(f'\nUse device: {device:}\n')

# hyperparameters
batch_size = 32
dataset_path = '~/Desktop/data'
model_path = '~/Desktop/models'


def load_data(dataset_path, artifact, type):
    '''
    type needs to be 'train', 'val', or 'test'

    artifact needs to be 'tracings' or 'spectrograms'
    '''
    # Load the data
    dataset_path = os.path.expanduser(dataset_path)

    path_to_labels = os.path.join(dataset_path, 'exams.csv')
    path_to_data = os.path.join(dataset_path, f'{type}.h5')

    # Get labels
    df = pd.read_csv(path_to_labels, index_col='exam_id')

    # Get h5 file
    h5file = h5py.File(path_to_data, 'r')
    traces_ids = h5file['exam_id']

    # Only keep the traces in the csv that match the traces in the h5 file
    df = df[df.index.isin(traces_ids)]

    # Sort the dataframe in trace order
    df = df.reindex(traces_ids)

    # Get AF labels and patient_ids
    labels = torch.tensor(df[artifact].astype(int).values, dtype=torch.float32).reshape(
        -1, 1
    )
    patient_ids = torch.tensor(
        df['patient_id'].astype(int).values, dtype=torch.int32
    ).reshape(-1, 1)

    # Define dataloaders
    dataset = MultExamH5Dataset(path_to_data, labels, patient_ids)

    return dataset


test_data = load_data(dataset_path, 'AF', 'test')
tqdm.write(f'Loaded {len(test_data)} samples for testing\n')
loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
tqdm.write(f'Using {len(loader)} batches of size {batch_size} for evaluating\n')

model = ResNet1d(
    input_dim=(12, 4096),
    n_classes=1,
    blocks_dim=[(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)],
)
model_name = 'code15e20adv001/best.pth'
model.load_state_dict(
    torch.load(
        os.path.join(os.path.expanduser(model_path), model_name),
        map_location=lambda storage, loc: storage,
    )['model']
)
tqdm.write('Model loaded:')
summary(model, input_size=(4096, 12))

model.to(device)

predictions = []
tqdm.write('\nEvaluating on test set...')
for X, patient_ids, y in tqdm(loader):
    X = X.to(device)
    with torch.no_grad():
        preds = model(X).cpu()
    batch_pred = pd.DataFrame(
        {
            'truth': y.numpy().flatten(),
            'prediction': preds.numpy().flatten(),
            'patient_id': patient_ids.numpy().flatten(),
        }
    )
    predictions.append(batch_pred)

df = pd.concat(predictions, ignore_index=True)
