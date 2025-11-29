import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm.auto import tqdm


class AudioProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, init_temp: float = 0.07):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)  # Audio -> Text

        # log scaled lernable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    def get_temperature(self) -> float:
        return self.logit_scale.exp().item()


def contrastive_loss(
    audio_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor
) -> torch.Tensor:
    """
    inputs sould be aligned audio[i] -> text[i]
    """
    audio_features = F.normalize(audio_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # similarity matrix / T
    logits = (audio_features @ text_features.t()) * logit_scale.exp()  # (Batch, Dim) @ (Dim, Batch) -> (Batch, Batch)

    # GT on diag
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device, dtype=torch.long)

    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.t(), labels)

    return (loss_a2t + loss_t2a) / 2


def train_epoch(model: AudioProjection, loader: DataLoader, optimizer: optim.AdamW, device: str) -> float:
    model.train()
    total_loss = 0

    for audio_emb, text_emb in tqdm(loader, desc="Training"):
        audio_emb, text_emb = audio_emb.to(device), text_emb.to(device)

        optimizer.zero_grad()
        audio_proj = model(audio_emb)
        loss = contrastive_loss(audio_proj, text_emb, model.logit_scale)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_retrieval(model: AudioProjection, loader: DataLoader, device: str, k_list=[1, 3, 10]) -> tuple[dict, int]:
    model.eval()

    all_audio_proj = []
    all_text = []

    # project audio
    for audio_emb, text_emb in loader:
        audio_emb = audio_emb.to(device)
        all_audio_proj.append(model(audio_emb))
        all_text.append(text_emb.to(device))

    audio_feats = torch.cat(all_audio_proj, dim=0)
    text_feats = torch.cat(all_text, dim=0)

    audio_feats = torch.nn.functional.normalize(audio_feats, dim=-1)
    text_feats = torch.nn.functional.normalize(text_feats, dim=-1)

    n_samples = audio_feats.shape[0]

    # might run out of memory, might have to change to CPU
    sim_matrix = (audio_feats @ text_feats.t()).to(device)  # similarity matrix (N, N)

    targets = torch.arange(n_samples, device=device)

    results = {}

    max_k = max(k_list)
    _, top_indices = sim_matrix.topk(max_k, dim=1)  # (N, max_k)

    for k in k_list:
        # check if target is in first K columns
        # (N, 1) == (N, k) with broadcasting
        hits = (targets.unsqueeze(1) == top_indices[:, :k]).any(dim=1)
        acc = hits.float().mean().item()
        results[f"R@{k}"] = acc

    return results, n_samples
