import torch
import torch.nn as nn
from sahformer.model.config import ModelConfig
from sahformer.model.embedding import InputEmbedding
from sahformer.model.encoder import Encoder
from sahformer.model.heads import PolicyHead, ValueHead, ThinkTimeMDNHead, policy_difficulty
from sahformer.model.temporal import TemporalEncoder, FiLMGenerator

class ClockAwareChessformer(nn.Module):
    def __init__(self, cfg: ModelConfig, use_film: bool = True, use_time_gab: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_film = use_film
        self.use_time_gab = use_time_gab
        self.input_emb = InputEmbedding(cfg)
        self.temporal_enc = TemporalEncoder(cfg)
        self.film_gen = FiLMGenerator(cfg) if use_film else None
        self.encoder = Encoder(cfg, time_conditioned=use_time_gab)
        self.policy = PolicyHead(cfg)
        self.value = ValueHead(cfg)
        self.think = ThinkTimeMDNHead(cfg)
        self.t_to_d = nn.Linear(cfg.t_ctx, cfg.dim_vit)

    def forward(self, batch: dict) -> dict:
        tok = self.input_emb(batch["board"].float(), batch["history"].float(),
                             batch["elo_self"], batch["elo_opp"])
        t = self.temporal_enc(batch["temporal"])
        film = self.film_gen(t) if self.use_film else None
        enc = self.encoder(tok, t=(t if self.use_time_gab else None), film=film)
        pooled = enc.mean(dim=1) + self.t_to_d(t)
        move_logits = self.policy(enc)
        think_in = torch.cat([pooled, policy_difficulty(move_logits)], dim=-1)
        return {
            "move_logits": move_logits,
            "value_logits": self.value(enc),
            "mdn": self.think(think_in),
            "pooled": pooled,            # position summary — used to condition clone adapters
        }
