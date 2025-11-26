from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torchaudio
from transformers import AutoModel, AutoProcessor, CLIPTextModel, CLIPTokenizer


DEVICE = "cuda:0"
MAX_LENGTH = 77


@dataclass
class EncoderConfig:
    audio_encoder_name: str
    text_encoder_name: str
    audio_sample_rate: int = 48000
    batch_size: int = 32
    checkpoint_freq: int = 100


class AudioEncoder:
    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_audio(self, audio_path: str) -> np.ndarray:
        waveform, sr = torchaudio.load(audio_path)

        # convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.processor.feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(
                sr, self.processor.feature_extractor.sampling_rate
            )
            waveform = resampler(waveform)

        inputs = self.processor(
            audios=waveform.squeeze().numpy(),
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = self.model.get_audio_features(**inputs)
        embedding = outputs.cpu().numpy()

        return embedding.squeeze()

    @torch.no_grad()
    def encode_audio_batch(self, audio_paths: List[str]) -> np.ndarray:
        # probably could be parallelized
        embeddings = []
        for path in tqdm(audio_paths, desc="Encoding audio"):
            emb = self.encode_audio(path)
            embeddings.append(emb)
        return np.stack(embeddings)


class TextEncoder:
    """Wrapper for CLIP text encoder"""

    def __init__(self, model_name: str):
        self.model = CLIPTextModel.from_pretrained(model_name).to(DEVICE)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = self.model(**inputs)

        # CLS token
        embeddings = outputs.pooler_output

        # L2 norm
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()


if __name__ == "__main__":
    config = EncoderConfig(
        audio_encoder_name="laion/clap-htsat-unfused",
        text_encoder_name="openai/clip-vit-base-patch32",
    )

    print("\nConfig:")
    print(config)

    print("\n" + "=" * 80)
    print("Step 1: Initialize Encoders")
    print("=" * 80)

    audio_encoder = AudioEncoder(config.audio_encoder_name)
    text_encoder = TextEncoder(config.text_encoder_name)

    print("Audio encoder loaded and frozen successfully")
    print("Text encoder loaded and frozen successfully")

    print("\n" + "=" * 80)
    print("Step 2: Extract Sample Embeddings")
    print("=" * 80)

    sample_texts = [
        "lalalala",
        "whatever",
        "sample deez nuts"
    ]

    text_embeddings = text_encoder.encode_text(sample_texts)
    print(f"Text embeddings shape: {text_embeddings.shape}")

