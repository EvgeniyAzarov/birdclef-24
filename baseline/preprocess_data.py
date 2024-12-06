import math
import os

import cv2
import librosa
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy as sci
import torch
from config import Config
from tqdm.auto import tqdm


def ogg2spec_via_scipy(config, audio_data):
    # handles NaNs
    mean_signal = np.nanmean(audio_data)
    audio_data = np.nan_to_num(audio_data, nan=mean_signal) if np.isnan(audio_data).mean() < 1 else np.zeros_like(audio_data)

    # to spec.
    frequencies, times, spec_data = sci.signal.spectrogram(
        audio_data, 
        fs=config.FS, 
        nfft=config.N_FFT, 
        nperseg=config.WIN_SIZE, 
        noverlap=config.WIN_LAP, 
        window='hann'
    )
    
    # Filter frequency range
    valid_freq = (frequencies >= config.MIN_FREQ) & (frequencies <= config.MAX_FREQ)
    spec_data = spec_data[valid_freq, :]
    
    # Log
    spec_data = np.log10(spec_data + 1e-20)
    
    # min/max normalize
    spec_data = spec_data - spec_data.min()
    spec_data = spec_data / spec_data.max()
    
    return spec_data

def main():
    config = Config()

    # labels
    label_list = sorted(os.listdir(os.path.join(config.DATA_ROOT, 'train_audio')))
    label_id_list = list(range(len(label_list)))
    label2id = dict(zip(label_list, label_id_list))
    id2label = dict(zip(label_id_list, label_list))

    metadata_df = pd.read_csv(f'{config.DATA_ROOT}/train_metadata.csv')

    train_df = metadata_df[['primary_label', 'rating', 'filename']].copy()

    # create target
    train_df['target'] = train_df.primary_label.map(label2id)
    # create filepath
    train_df['filepath'] = config.DATA_ROOT + '/train_audio/' + train_df.filename
    # create new sample name
    train_df['samplename'] = train_df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

    print(f'{len(train_df)} samples')


    all_bird_data = dict()
    for i, row_metadata in tqdm(train_df.iterrows(), total=train_df.shape[0]):

        # load ogg
        audio_data, _ = librosa.load(row_metadata.filepath, sr=config.FS)

        # crop
        n_copy = math.ceil(5 * config.FS / len(audio_data))
        if n_copy > 1: audio_data = np.concatenate([audio_data]*n_copy)

        # start_idx = int(len(audio_data) / 2 - 2.5 * config.FS)
        start_idx = 0
        end_idx = int(start_idx + 5.0 * config.FS)
        input_audio = audio_data[start_idx:end_idx]

        # ogg to spec.
        input_spec = ogg2spec_via_scipy(config, input_audio)
        
        input_spec = cv2.resize(input_spec, (256, 256), interpolation=cv2.INTER_AREA)

        all_bird_data[row_metadata.samplename] = input_spec.astype(np.float32)

    # save to file
    np.save(os.path.join(config.OUTPUT_DIR, f'spec_first_5sec_256_256.npy'), all_bird_data)


if __name__ == "__main__":
    main()