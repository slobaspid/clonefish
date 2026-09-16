"""Personal 'clone' adapter: a small deviation on top of the frozen all-around model.

The base model already predicts BOTH what a generic human plays (move_logits) and how long
they think (the MDN clock head). A clone is just a per-person residual on those two heads,
conditioned on the base's position summary (`pooled`). Zero-initialized, so an untrained
clone reproduces the base exactly and training only teaches it where THIS person deviates.
"""
import torch
import torch.nn as nn


class CloneAdapter(nn.Module):
    def __init__(self, dim: int, n_moves: int, mdn_k: int, hid: int = 256):
        super().__init__()
        self.dim = dim
        self.n_moves = n_moves
        self.mdn_k = mdn_k
        self.move = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.GELU(),
                                  nn.Linear(hid, n_moves))          # move deviation
        self.time = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hid), nn.GELU(),
                                  nn.Linear(hid, mdn_k * 3))        # clock deviation (dpi,dmu,dsigma)
        for last in (self.move[-1], self.time[-1]):                 # start == base (pure deviation)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def move_residual(self, pooled):
        return self.move(pooled)                                    # [b, n_moves]

    def time_offset(self, pooled):
        o = self.time(pooled)
        k = self.mdn_k
        return o[..., :k], o[..., k:2 * k], o[..., 2 * k:3 * k]     # dpi, dmu, dsigma


def apply_clone(out: dict, adapter: CloneAdapter) -> dict:
    """Deviate a base forward's outputs by the clone. Returns a shallow-copied dict."""
    pooled = out["pooled"]
    pi, mu, sigma = out["mdn"]
    dpi, dmu, dsigma = adapter.time_offset(pooled)
    out = dict(out)
    out["move_logits"] = out["move_logits"] + adapter.move_residual(pooled)
    out["mdn"] = (pi + dpi, mu + dmu, sigma + dsigma)
    return out


def save_clone(path: str, adapter: CloneAdapter, meta: dict):
    torch.save({"state_dict": adapter.state_dict(),
                "dim": adapter.dim, "n_moves": adapter.n_moves,
                "mdn_k": adapter.mdn_k, "meta": meta}, path)


def load_clone(path: str, device: str = "cpu") -> CloneAdapter:
    ck = torch.load(path, map_location=device, weights_only=False)   # our own trusted file
    a = CloneAdapter(ck["dim"], ck["n_moves"], ck["mdn_k"]).to(device)
    a.load_state_dict(ck["state_dict"])
    a.eval()
    a.meta = ck.get("meta", {})          # name, avg_elo, n_moves_seen, ...
    return a


class PooledCloneAdapter(nn.Module):
    """The WINNING clone (+4pp): a shared move-residual head (trained across many players) driven
    by ONE person's low-dim style code. Move deviation only — timing uses the BASE clock (the
    timing residual over-thinks), so time_offset is exactly zero (§0.1 handoff)."""
    def __init__(self, net: nn.Module, code: torch.Tensor):
        super().__init__()
        self.net = net
        self.register_buffer("code", code.detach().clone())

    def move_residual(self, pooled):
        c = self.code.to(pooled.dtype).expand(pooled.shape[0], -1)
        return self.net(torch.cat([pooled, c], dim=-1))

    def time_offset(self, pooled):
        return 0.0, 0.0, 0.0                 # base clock (no timing residual)


def save_pooled_clone(path: str, adapter: PooledCloneAdapter, meta: dict):
    torch.save({"kind": "pooled", "net": adapter.net, "code": adapter.code.cpu(), "meta": meta}, path)


def load_pooled_clone(path: str, device: str = "cpu") -> PooledCloneAdapter:
    ck = torch.load(path, map_location=device, weights_only=False)   # our own trusted file
    a = PooledCloneAdapter(ck["net"], ck["code"]).to(device)
    a.eval()
    a.meta = ck.get("meta", {})
    return a


def load_any_clone(path: str, device: str = "cpu"):
    """Dispatch on the saved format: single-player CloneAdapter or PooledCloneAdapter."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("kind") == "pooled":
        return load_pooled_clone(path, device)
    return load_clone(path, device)
