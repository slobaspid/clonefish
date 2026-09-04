"""Per-band position budgeting with whole-game admission (design 2026-09-02 s3)."""
from dataclasses import dataclass

from sahformer.bandcap import BANDS, BandBudget, band_of, bands_touched, in_range


@dataclass
class FakeRec:
    elo_self: int


def game(elo_a, elo_b, plies=10):
    """A game's records: half the positions are each player's moves."""
    return [FakeRec(elo_a)] * (plies // 2) + [FakeRec(elo_b)] * (plies // 2)


def test_band_layout():
    assert BANDS[0] == 800 and BANDS[-1] == 2900 and len(BANDS) == 22
    assert band_of(800) == 800 and band_of(899) == 800 and band_of(900) == 900
    assert in_range(800) and in_range(2900)
    assert not in_range(700) and not in_range(3000)


def test_peer_matched_game_touches_one_band_mismatch_touches_two():
    # a <100 gap cannot straddle a 100-wide band
    assert bands_touched(1520, 1560) == {1500}
    assert bands_touched(1300, 1890) == {1300, 1800}
    # out-of-range sides are dropped, in-range side still counts
    assert bands_touched(700, 1500) == {1500}
    assert bands_touched(700, 3200) == set()


def test_admits_until_budget_then_stops_wanting():
    b = BandBudget(budget=100)
    for _ in range(10):
        g = game(1500, 1520, plies=10)          # 10 positions, all band 1500
        assert b.wants(bands_touched(1500, 1520))
        b.admit(g)
    assert b.counts[1500] == 100
    assert not b.wants(bands_touched(1500, 1520))


def test_whole_game_admission_overshoots_rather_than_splitting():
    """The core rule: never truncate a game, so a band lands slightly OVER budget."""
    b = BandBudget(budget=95)
    for _ in range(10):
        if b.wants(bands_touched(1500, 1520)):
            b.admit(game(1500, 1520, plies=10))
    assert b.counts[1500] == 100        # 95 budget, admitted whole games -> 100, not 95
    assert b.counts[1500] >= 95


def test_mismatch_game_splits_positions_across_two_bands():
    b = BandBudget(budget=1000)
    b.admit(game(1300, 1890, plies=10))
    assert b.counts[1300] == 5
    assert b.counts[1800] == 5


def test_spill_counted_when_a_mismatch_overshoots_a_full_band():
    b = BandBudget(budget=10)
    b.admit(game(1300, 1300, plies=20))      # fills band 1300 to 20 (over its budget of 10)
    assert b.counts[1300] == 20
    assert b.spill == 10                      # 10 of those landed past the budget
    spill_before = b.spill
    # band 1800 still wants games; the 1300 half spills further
    assert b.wants(bands_touched(1300, 1890))
    b.admit(game(1300, 1890, plies=10))
    assert b.counts[1800] == 5
    assert b.spill == spill_before + 5


def test_full_requires_every_band():
    b = BandBudget(budget=10, bands=(800, 900))
    b.admit([FakeRec(850)] * 10)
    assert not b.full()
    assert b.shortfall() == 10
    b.admit([FakeRec(950)] * 10)
    assert b.full()
    assert b.shortfall() == 0


def test_out_of_range_positions_are_not_counted():
    b = BandBudget(budget=100)
    b.admit([FakeRec(700)] * 5 + [FakeRec(1500)] * 5)   # 700 is below the floor
    assert b.counts[700] == 0
    assert b.counts[1500] == 5
    assert b.positions == 5


def test_report_mentions_shortfall_only_when_short():
    b = BandBudget(budget=10, bands=(800,))
    assert "SHORTFALL" in b.report()
    b.admit([FakeRec(850)] * 10)
    assert "SHORTFALL" not in b.report()
