import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm.auto import tqdm

from .model import AudioEncoder, EncoderConfig, TextEncoder


def extract_audio_embeddings_with_checkpointing(
    audio_dir: Path,
    encoder: AudioEncoder,
    checkpoint_path: Path,
    checkpoint_freq: int = 100,
    resume: bool = True
) -> Dict[str, np.ndarray]:
    embeddings = {}

    if resume and checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Resumed from checkpoint: {len(embeddings)} embeddings loaded")

    audio_files = sorted(list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.wav")))

    remaining_files = [f for f in audio_files if f.name not in embeddings]

    print(f"Processing {len(remaining_files)} audio files...")

    for idx, audio_file in enumerate(tqdm(remaining_files, desc="Extracting audio embeddings")):
        try:
            embedding = encoder.encode_audio(str(audio_file))
            embeddings[audio_file.name] = embedding

            if (idx + 1) % checkpoint_freq == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                print(f"Checkpoint saved: {len(embeddings)} embeddings")

        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}", file=sys.stderr)
            continue

    # final checkpoint
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings


def process_dataset(
    tsv_path: Path,
    audio_embeddings: Dict[str, np.ndarray],
    text_encoder: TextEncoder,
    output_path: Path,
    batch_size: int = 32
) -> List[Dict]:
    df = pd.read_csv(tsv_path, sep='\t')

    print(f"Processing {len(df)} samples from {tsv_path.name}")

    dataset = []
    # print(f"audio_embeddings: {audio_embeddings.keys()}")

    for start_idx in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[start_idx:start_idx + batch_size]

        # extract text embeddings
        assert "text" in batch_df.columns, "No text column found in tsv"
        texts = batch_df['text'].tolist()
        text_embeddings = text_encoder.encode_text(texts)

        # match with audio
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            # Construct audio filename (adjust based on your file naming)
            audio_filename = row['audio'].split('/')[-1]

            if audio_filename not in audio_embeddings:
                print(f"Skipping {audio_filename} because no embedding found")
                continue

            assert "uniq_id" in row.index, "No uniq_id column found in tsv"

            sample = {
                "uniq_id": row['uniq_id'],
                "text": row['text'],
                "audio_embedding": audio_embeddings[audio_filename],
                "text_embedding": text_embeddings[idx]
            }
            dataset.append(sample)

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"Saved {len(dataset)} samples to {output_path}")

    return dataset


class AudioTextDataset(Dataset):
    def __init__(self, pickle_path: Path):
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)

        print(f"Loaded dataset with {len(self.data)} samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]

        audio_emb = torch.from_numpy(sample['audio_embedding']).float()
        text_emb = torch.from_numpy(sample['text_embedding']).float()

        return audio_emb, text_emb


def get_embedding_dimensions(config: EncoderConfig) -> Tuple[int, int]:
    audio_encoder = AudioEncoder(config.audio_encoder_name)
    text_encoder = TextEncoder(config.text_encoder_name)

    # rofloforward pass to get dims
    dummy_audio = torch.randn(1, 16000).numpy()
    dummy_text = ["whatever"]

    audio_dim = len(audio_encoder.processor(
        audios=dummy_audio,
        sampling_rate=audio_encoder.processor.feature_extractor.sampling_rate,
        return_tensors="pt"
    )['input_features'][0])

    text_dim = text_encoder.encode_text(dummy_text).shape[-1]

    return audio_dim, text_dim


if __name__ == "__main__":
    DATA_ROOT = "../data/audiocaps/"
    CHECKPOINT_FREQ = 50
    BATCH_SIZE = 32
    CHECKPOINT_FILENAME = "audio_embeddings_train.pkl"
    PROCESSED_DATASET_FILENAME = "train_processed.pkl"

    data_root = Path(DATA_ROOT)
    assert data_root.exists(), "No data directory found"

    audio_dir = data_root / "audio" / "train"
    assert audio_dir.exists()

    assert audio_dir.exists() and list(audio_dir.glob("*.flac")), "No audio files found in train directory"

    tsv_path = data_root / "audiocaps_train.tsv"
    assert tsv_path.exists(), "No train.tsv file found"

    config = EncoderConfig(
        audio_encoder_name="laion/clap-htsat-unfused",
        text_encoder_name="openai/clip-vit-base-patch32",
    )

    print("\nConfig:")
    print(config)

    audio_encoder = AudioEncoder(config.audio_encoder_name)
    text_encoder = TextEncoder(config.text_encoder_name)

    print(f"Extracting Audio Embeddings (with checkpointing) from {audio_dir}...")

    checkpoint_path = data_root / CHECKPOINT_FILENAME

    if not checkpoint_path.exists():
        audio_embeddings = extract_audio_embeddings_with_checkpointing(
            audio_dir=audio_dir,
            encoder=audio_encoder,
            checkpoint_path=checkpoint_path,
            checkpoint_freq=CHECKPOINT_FREQ,
            resume=True
        )
        print(f"Extracted {len(audio_embeddings)} audio embeddings")
    else:
        audio_embeddings = pickle.load(open(checkpoint_path, 'rb'))
        print(f"Loaded {len(audio_embeddings)} audio embeddings from checkpoint {checkpoint_path}")

    print("Processing Dataset (merge audio + text)...")

    dataset = process_dataset(
        tsv_path=tsv_path,
        audio_embeddings=audio_embeddings,
        text_encoder=text_encoder,
        output_path=data_root / PROCESSED_DATASET_FILENAME,
        batch_size=BATCH_SIZE
    )

    print(f"Processed dataset with {len(dataset)} samples")

    print("Creating PyTorch Dataset...")

    pytorch_dataset = AudioTextDataset(data_root / PROCESSED_DATASET_FILENAME)

    print("Sample:")
    audio_emb, text_emb = pytorch_dataset[0]
    print(f"\taudio embedding shape: {audio_emb.shape}")
    print(f"\ttext embedding shape: {text_emb.shape}")
    print("Great success")
