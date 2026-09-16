from dataclasses import dataclass
import hashlib
import numpy as np
import chess
from sahformer.encoding import encode_board, encode_move, build_temporal, TEMPORAL_DIM

BASE_SECONDS = 180.0  # 3+0

SITE_CHESSCOM = 0
SITE_LICHESS = 1
SITE_UNKNOWN = -1

# FIXED FOREVER. The hash must be identical across runs, machines and rebuilds or a
# player's games scatter across several ids and nothing can be pulled back out.
_PLAYER_SALT = b"sahformer-player-id-v1"


def player_id_of(username):
    """Stable, salted, one-way id for a username. 0 means unknown.

    Usernames never reach the shards - only this id - while the raw PGN archives keep them
    locally. Chess.com names are case-insensitive, so we fold case (and strip whitespace)
    to keep one person on one id.
    """
    if not username:
        return 0
    norm = str(username).strip().lower()
    if not norm:
        return 0
    digest = hashlib.blake2b(_PLAYER_SALT + norm.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def parse_pgn_date(headers):
    """PGN date -> YYYYMMDD int, 0 if absent or malformed (PGN uses '????.??.??').

    Prefer UTCDate: chess.com writes both, and Date is local-time so it can straddle a day
    boundary and mis-order a player's games.
    """
    for key in ("UTCDate", "Date"):
        raw = headers.get(key)
        if not raw:
            continue
        parts = str(raw).strip().split(".")
        if len(parts) != 3:
            continue
        try:
            y, m, d = (int(p) for p in parts)
        except ValueError:
            continue
        if 1 <= m <= 12 and 1 <= d <= 31 and y > 0:
            return y * 10000 + m * 100 + d
    return 0


def site_of(headers):
    """Which platform's rating scale this game is on - 1800 does not mean the same on both."""
    raw = str(headers.get("Site", "")).lower()
    if "chess.com" in raw:
        return SITE_CHESSCOM
    if "lichess" in raw:
        return SITE_LICHESS
    return SITE_UNKNOWN


@dataclass
class PositionRecord:
    board: np.ndarray        # int8[8,8,12]
    history: np.ndarray      # int8[7,8,8,12]
    stm: int
    elo_self: int
    elo_opp: int
    temporal: np.ndarray     # float32[TEMPORAL_DIM]
    move_from: int
    move_to: int
    promo: int
    result: int              # stm-relative: 0 loss, 1 draw, 2 win
    think_time: float
    player_id: int = 0       # int64, the MOVER (matches elo_self); 0 = unknown
    date: int = 0            # int32 YYYYMMDD; 0 = unknown
    site: int = SITE_UNKNOWN # int8

_RESULT_WHITE = {"1-0": 2, "0-1": 0, "1/2-1/2": 1}

def _result_for_stm(result_str: str, white_to_move: bool) -> int:
    w = _RESULT_WHITE.get(result_str, 1)
    if white_to_move:
        return w
    return {0: 2, 1: 1, 2: 0}[w]  # mirror for black

def game_to_records(game):
    """Yield a PositionRecord per ply that has a clock annotation."""
    result_str = game.headers.get("Result", "1/2-1/2")
    white_elo = int(game.headers.get("WhiteElo", 0) or 0)
    black_elo = int(game.headers.get("BlackElo", 0) or 0)
    white_id = player_id_of(game.headers.get("White", ""))
    black_id = player_id_of(game.headers.get("Black", ""))
    game_date = parse_pgn_date(game.headers)
    game_site = site_of(game.headers)

    board = game.board()
    prev_clock = {chess.WHITE: BASE_SECONDS, chess.BLACK: BASE_SECONDS}
    think_hist = {chess.WHITE: [], chess.BLACK: []}  # most-recent first
    plane_hist = []  # list of int8[8,8,12], newest last

    node = game
    ply = 0
    while node.variations:
        node = node.variation(0)
        move = node.move
        mover = board.turn                      # who is about to move
        clock_after = node.clock()              # seconds left AFTER this move
        if clock_after is None:
            board.push(move)
            continue
        think = max(prev_clock[mover] - clock_after, 0.0)

        my_clock = prev_clock[mover]
        opp_clock = prev_clock[not mover]
        temporal = build_temporal(
            my_clock=my_clock, opp_clock=opp_clock,
            own_think_history=think_hist[mover], ply=ply,
        )

        cur = encode_board(board)
        hist = _stack_history(plane_hist, cur)
        frm, to, promo = encode_move(board, move)

        yield PositionRecord(
            board=cur, history=hist, stm=0 if mover == chess.WHITE else 1,
            elo_self=white_elo if mover == chess.WHITE else black_elo,
            elo_opp=black_elo if mover == chess.WHITE else white_elo,
            temporal=temporal, move_from=frm, move_to=to, promo=promo,
            result=_result_for_stm(result_str, mover == chess.WHITE),
            think_time=think,
            player_id=white_id if mover == chess.WHITE else black_id,
            date=game_date,
            site=game_site,
        )

        # advance bookkeeping
        prev_clock[mover] = clock_after
        think_hist[mover] = [think] + think_hist[mover]
        plane_hist.append(cur)
        board.push(move)
        ply += 1

def _stack_history(plane_hist, current):
    """Return int8[7,8,8,12]: the 7 plies before `current`, newest first,
    earliest repeated if fewer than 7 exist."""
    out = np.zeros((7, 8, 8, 12), dtype=np.int8)
    prev = list(reversed(plane_hist[-7:]))  # newest first
    for i in range(7):
        if i < len(prev):
            out[i] = prev[i]
        elif prev:
            out[i] = prev[-1]  # repeat earliest available
        else:
            out[i] = current   # very first ply: repeat current
    return out
