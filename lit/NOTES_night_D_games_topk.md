# Night D: non-chess player-cloning + top-k losses

Verified from arXiv/primary abstract pages only. Chess papers already in BRIEF/lit/ not repeated.

| Paper | ID | Verified claim | Maps to us |
|---|---|---|---|
| Berrada, Zisserman, Kumar, Smooth Loss Functions for Deep Top-k Classification | 1802.07595 (ICLR'18) | Abstract: top-k-specific loss "can bring significant improvements" **"in the context of limited and noisy data"**; "more robust to noise and overfitting than cross-entropy"; built for k=5. No numeric deltas in abstract. | Our per-player fine-tune (5k games, noisy human labels) is exactly this small/noisy regime — unlike bulk pretrain. |
| Lapin, Hein, Schiele, Loss Functions for Top-k Error: Analysis and Insights | 1512.00486 | Abstract, verbatim: **"the softmax loss yields competitive top-k performance for all k simultaneously."** Their new losses give further, unquantified gains. | We already train with softmax/cross-entropy. This says CE is close to top-k-optimal already — matches our observed ~85% top-3 ceiling; a swapped loss on the frozen/pretrain base is unlikely to move much. |
| Fire Emblem "Mirror Mode" | 2512.11902 | Trains per-*individual participant* on that person's own demonstrations (closest non-chess analogue to our clone). Verified: good defensive imitation, weak offensive imitation; participants "recognized their own retreating tactics," higher satisfaction — evaluated by human survey, not move-accuracy numbers. | Only non-chess system found that clones a *named individual*, not a population/persona. Confirms human recognition test is a legit metric when automated judges fail (matches our failed judge-reward attempts). |
| KataGo HumanSL (prior dive, re-cited) | — | Conditions only on rank/era/rank-pair, never a specific player. | Reconfirms: no Go/poker/StarCraft/Dota/Rocket League system found (8 searches) clones one named individual's move-for-move decisions with reported numbers — chess (Maia-individual etc.) is still the frontier. |

Poker, StarCraft II, Dota 2, Rocket League: searched directly; found only population/class-level opponent modeling (Bayes' Bluff, evolved-RNN poker predictors ~80% action accuracy on HU Limit) or generic style-diversity RL (SCC, TStarBot-X, Necto) — none clone one specific player with a verified match-rate number.

## Top 2 ideas

1. **Try Berrada smooth-top-k (or CE+top-k blend) only in the per-player fine-tune stage**, not pretrain — the paper's stated regime (limited, noisy data) is our regime, whereas Lapin shows plain CE is already near-optimal in the bulk-data regime we pretrain on.
   Falsify: fine-tune the same player twice (CE vs smooth-top-k@3), held-out post-opening top-3. If the delta is inside our measured 0.13pp run-to-run noise, kill it.
2. **Add a blind human-recognition survey** (Mirror-Mode style: "is this your play?") as an evaluation metric, since our 3 automated judge-reward attempts already failed.
   Falsify: if judges can't tell clone-vs-self above chance on held-out games, the clone isn't personal regardless of move-match numbers.
