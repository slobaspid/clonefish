"""Per-band position budgeting with whole-game admission.

Two rules, and the second one is the subtle one.

**Even bands.** Bands must hold equal numbers of POSITIONS, because positions are what the
model sees. An unbalanced corpus is what lets Elo conditioning go lazy - if the model sees
twice as much 2600 play as 800 play it can ignore the Elo input and default to the majority,
which is exactly the "Elo dial became a confidence knob" failure in HANDOFF-08-16 s3.

**Whole games only.** A position is NOT independent of the rest of its game: its temporal
vector carries that player's own last-5 think-times, the <2s / <5s / <10s low-clock flags,
and ply/80 (`encoding.build_temporal`). Truncating a game to hit a budget therefore deletes
its BACK HALF - the flag scramble, the signal we deliberately keep where Maia-3 discards it -
and deletes more of it from bands that needed more trimming. So a game is admitted whole or
not at all, and bands land even to within one game (~73 positions), which is noise.

A mismatched game straddles two bands and so spills positions into the second one even when
only the first wanted it. At ~5% mismatch that is small; it is counted and reported rather
than prevented, since preventing it would mean splitting games.
"""
from collections import Counter

BAND_LO = 800
BAND_HI = 3000
BAND_W = 100
BANDS = tuple(range(BAND_LO, BAND_HI, BAND_W))


def band_of(elo):
    return (int(elo) // BAND_W) * BAND_W


def in_range(band):
    return BAND_LO <= band < BAND_HI


def bands_touched(white_elo, black_elo):
    """The in-range bands a game contributes positions to - one per side, by that side's OWN
    rating. Peer-matched sides share a band (a <100 gap cannot straddle a 100-wide band)."""
    return {b for b in (band_of(white_elo), band_of(black_elo)) if in_range(b)}


class BandBudget:
    """Tracks positions per band and decides whole-game admission."""

    def __init__(self, budget, bands=BANDS):
        self.budget = int(budget)
        self.bands = tuple(bands)
        self.counts = Counter()
        self.spill = 0          # positions landing in bands already at budget
        self.games = 0
        self.positions = 0

    def wants(self, touched):
        """Admit if ANY band this game touches still has room."""
        return any(self.counts[b] < self.budget for b in touched if in_range(b))

    def full(self):
        return all(self.counts[b] >= self.budget for b in self.bands)

    def admit(self, records):
        """Count an admitted game's positions, each into its mover's own band."""
        self.games += 1
        for r in records:
            b = band_of(r.elo_self)
            if not in_range(b):
                continue
            if self.counts[b] >= self.budget:
                self.spill += 1
            self.counts[b] += 1
            self.positions += 1

    def shortfall(self):
        return sum(max(0, self.budget - self.counts[b]) for b in self.bands)

    def report(self):
        lines = [f"{'band':>10} {'positions':>12} {'vs budget':>10}"]
        for b in self.bands:
            n = self.counts[b]
            lines.append(f"{b:>6}-{b + BAND_W - 1} {n:>12,} {n / self.budget:>9.2f}x"
                         f"{'' if n >= self.budget else '  SHORT'}")
        lines.append("")
        lines.append(f"games admitted: {self.games:,} | positions: {self.positions:,} | "
                     f"budget/band: {self.budget:,}")
        lines.append(f"cross-band spill (mismatch games overshooting a full band): "
                     f"{self.spill:,} ({100 * self.spill / max(self.positions, 1):.2f}%)")
        if self.shortfall():
            lines.append(f"SHORTFALL: {self.shortfall():,} positions - some band ran out of games")
        return "\n".join(lines)
