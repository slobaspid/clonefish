"""player_id / date / site on PositionRecord (design 2026-09-02 s4).

Without these a person's games cannot be pulled back out of the shards, cannot be ordered
in time (so no rating trajectory), and chess.com Elo cannot be told apart from Lichess Elo.
The information is in the PGN headers and was being dropped at record-build time.
"""
import io

import chess.pgn
import numpy as np
import pytest

from sahformer.records import (
    SITE_CHESSCOM,
    SITE_LICHESS,
    SITE_UNKNOWN,
    game_to_records,
    parse_pgn_date,
    player_id_of,
    site_of,
)
from sahformer.shards import records_to_arrays


PGN = """[Event "Live Chess"]
[Site "Chess.com"]
[White "AliceP"]
[Black "bobQ"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1450"]
[TimeControl "180"]
[UTCDate "2024.03.15"]

1. e4 {[%clk 0:02:58]} e5 {[%clk 0:02:57]} 2. Nf3 {[%clk 0:02:55]} 1-0
"""


def _game(pgn=PGN):
    return chess.pgn.read_game(io.StringIO(pgn))


# --- player_id ------------------------------------------------------------------

def test_player_id_is_stable_and_case_insensitive():
    assert player_id_of("AliceP") == player_id_of("AliceP")
    assert player_id_of("AliceP") == player_id_of("alicep")     # chess.com names are case-insensitive
    assert player_id_of("  AliceP ") == player_id_of("alicep")  # and stray whitespace must not fork an id
    assert player_id_of("AliceP") != player_id_of("bobQ")


def test_player_id_is_zero_for_missing_name():
    assert player_id_of("") == 0
    assert player_id_of(None) == 0


def test_player_id_fits_int64():
    for name in ("a", "AliceP", "x" * 200, "üñïçø∂é"):
        v = player_id_of(name)
        assert -(2 ** 63) <= v < 2 ** 63
        assert np.int64(v) == v          # round-trips through the shard dtype


# --- date -----------------------------------------------------------------------

def test_parse_pgn_date_prefers_utcdate():
    assert parse_pgn_date({"UTCDate": "2024.03.15", "Date": "2024.03.14"}) == 20240315
    assert parse_pgn_date({"Date": "2021.12.01"}) == 20211201


def test_parse_pgn_date_zero_when_absent_or_malformed():
    assert parse_pgn_date({}) == 0
    assert parse_pgn_date({"Date": "????.??.??"}) == 0
    assert parse_pgn_date({"Date": "not a date"}) == 0


def test_dates_sort_chronologically():
    days = [parse_pgn_date({"UTCDate": d}) for d in
            ("2023.01.02", "2023.01.10", "2023.02.01", "2024.01.01")]
    assert days == sorted(days)


# --- site -----------------------------------------------------------------------

def test_site_detection():
    assert site_of({"Site": "Chess.com"}) == SITE_CHESSCOM
    assert site_of({"Site": "https://lichess.org/abcd1234"}) == SITE_LICHESS
    assert site_of({}) == SITE_UNKNOWN


# --- wired into game_to_records -------------------------------------------------

def test_records_carry_the_movers_identity():
    recs = list(game_to_records(_game()))
    assert len(recs) == 3                       # e4 (W), e5 (B), Nf3 (W)
    alice, bob = player_id_of("AliceP"), player_id_of("bobQ")
    # player_id must follow the MOVER, matching elo_self
    assert [r.player_id for r in recs] == [alice, bob, alice]
    assert [r.elo_self for r in recs] == [1500, 1450, 1500]
    assert [r.elo_opp for r in recs] == [1450, 1500, 1450]


def test_records_carry_date_and_site():
    recs = list(game_to_records(_game()))
    assert all(r.date == 20240315 for r in recs)
    assert all(r.site == SITE_CHESSCOM for r in recs)


def test_arrays_carry_new_fields_with_shard_dtypes():
    arr = records_to_arrays(list(game_to_records(_game())))
    assert arr["player_id"].dtype == np.int64
    assert arr["date"].dtype == np.int32
    assert arr["site"].dtype == np.int8
    assert arr["player_id"].tolist() == [player_id_of("AliceP"),
                                         player_id_of("bobQ"),
                                         player_id_of("AliceP")]
    assert arr["date"].tolist() == [20240315] * 3


def test_one_players_positions_are_recoverable_from_a_shard_array():
    """The whole point: select a person's rows back out and get their games in order."""
    arr = records_to_arrays(list(game_to_records(_game())))
    mine = arr["player_id"] == player_id_of("alicep")     # looked up by name, any case
    assert mine.sum() == 2
    assert set(arr["elo_self"][mine].tolist()) == {1500}
