import collections
import datetime
import fluidsynth
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple

import pathlib
import pretty_midi

from torch.nn.utils.rnn import pad_sequence
from midi_and_notes import midi_to_notes

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
_SAMPLING_RATE = 16000

data_dir = pathlib.Path('data/maestro-v2.0.0')

def load_to_folder(data_dir):
    if not data_dir.exists():
        tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.', cache_subdir='data',
        )

    filenames = glob.glob(str(data_dir/'**/*.mid*'))
    print('Количество скаченных midi-файлов:', len(filenames))
    return filenames


def plot_piano_roll(notes: pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    _ = plt.title(title)
    
    
def get_array_data(filenames):
    train_data = []

    for i, file in enumerate(filenames):
        sample_file = filenames[i]
        raw_notes = midi_to_notes(sample_file)
        train_data.append(raw_notes)
        
    return train_data


class SongsDataset(Dataset):
    def __init__(self, dataframe_list):
        self.dataframes = dataframe_list

    def __len__(self):
        return len(self.dataframes)

    def __getitem__(self, idx):
        # Получение одной песни в виде DataFrame
        song_df = self.dataframes[idx]

        # Предобработка данных
        #label_encoder = OneHotEncoder()
        #song_df['pitch'] = label_encoder.fit_transform(song_df['pitch'])

        #scaler = MinMaxScaler()
        #song_np = scaler.fit_transform(song_df.values)

        song_df['pitch'] = song_df['pitch'].astype(int)
        song_tensor = torch.tensor(song_df.values, dtype=torch.float32)

        return song_tensor
    

def collate_fn_pad(batch, max_len=100):
    # Дополняем каждый тензор в пакете до длины max_len
    padded_batch = []
    for x in batch:
        if x.shape[0] > max_len:
            x = x[:max_len, :]
        elif x.shape[0] < max_len:
            padding_size = max_len - x.shape[0]
            x = torch.cat([x, torch.zeros(padding_size, x.shape[1])], dim=0)
        padded_batch.append(x)

    batch_padded = torch.stack(padded_batch, dim=0)
    return batch_padded

      