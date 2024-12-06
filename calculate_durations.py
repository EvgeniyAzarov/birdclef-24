import pandas as pd
import librosa
from tqdm.auto import tqdm


def main():
    meta_df = pd.read_csv("data/train_metadata.csv")
    filenames= meta_df['filename']
    data_root = "data/train_audio"

    durations = []
    for file in tqdm(filenames):
        audio, sr = librosa.load(f"{data_root}/{file}")
        l = librosa.get_duration(y=audio, sr=sr)
        durations.append(l)
    
    meta_df['duration'] = durations
    meta_df.to_csv("data/train_metadata_stat.csv", index=False)


if __name__ == "__main__":
    main()