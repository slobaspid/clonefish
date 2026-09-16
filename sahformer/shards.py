import numpy as np

N_BINS = 22

def elo_bin(elo: int) -> int:
    return int(np.clip((elo - 600) // 100, 0, N_BINS - 1))

def records_to_arrays(records):
    """Stack a list of PositionRecord into a dict of batched numpy arrays."""
    n = len(records)
    out = {
        "board": np.zeros((n, 8, 8, 12), np.int8),
        "history": np.zeros((n, 7, 8, 8, 12), np.int8),
        "stm": np.zeros(n, np.int8),
        "elo_self": np.zeros(n, np.int16),
        "elo_opp": np.zeros(n, np.int16),
        "temporal": np.zeros((n, records[0].temporal.shape[0]), np.float32),
        "move_from": np.zeros(n, np.int8),
        "move_to": np.zeros(n, np.int8),
        "promo": np.zeros(n, np.int8),
        "result": np.zeros(n, np.int8),
        "think_time": np.zeros(n, np.float32),
        "player_id": np.zeros(n, np.int64),   # the mover; lets a person's games be pulled back out
        "date": np.zeros(n, np.int32),        # YYYYMMDD; lets them be ordered -> rating trajectory
        "site": np.zeros(n, np.int8),         # rating scales differ per platform
    }
    for i, r in enumerate(records):
        out["board"][i] = r.board
        out["history"][i] = r.history
        out["stm"][i] = r.stm
        out["elo_self"][i] = r.elo_self
        out["elo_opp"][i] = r.elo_opp
        out["temporal"][i] = r.temporal
        out["move_from"][i] = r.move_from
        out["move_to"][i] = r.move_to
        out["promo"][i] = r.promo
        out["result"][i] = r.result
        out["think_time"][i] = r.think_time
        out["player_id"][i] = r.player_id
        out["date"][i] = r.date
        out["site"][i] = r.site
    return out

def balance_indices(elo_self: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return indices that equalize all present Elo bins to the smallest bin size."""
    rng = np.random.default_rng(seed)
    bins = np.array([elo_bin(int(e)) for e in elo_self])
    present = [b for b in range(N_BINS) if (bins == b).any()]
    min_count = min((bins == b).sum() for b in present)
    keep = []
    for b in present:
        idx = np.where(bins == b)[0]
        rng.shuffle(idx)
        keep.extend(idx[:min_count].tolist())
    keep = np.array(sorted(keep))
    return keep

def save_shard(path: str, arrays: dict):
    np.savez_compressed(path, **arrays)
