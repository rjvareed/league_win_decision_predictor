# League of Legends win predictor

Predicts the probability that blue side / red side wins a ranked solo queue game, before it
starts. Inputs are the ten champions and their sides (optionally roles) and, when available,
each player's recent match history. Trained on Diamond+ games from the newest patches, on a
single consumer GPU.

```
$ lolwp-predict --blue Aatrox "Lee Sin" Ahri Jinx Thresh --red Garen Vi Syndra Caitlyn Lux
Blue (Aatrox, Lee Sin, Ahri, Jinx, Thresh): 51.3%      <- illustrative output
Red  (Garen, Vi, Syndra, Caitlyn, Lux): 48.7%
```

## What to expect (read this first)

The draft alone says little about who wins a solo queue game; player skill and execution
dominate. Honest draft-only models land in the **low-to-mid 50s % accuracy** against a ~50-52%
"always pick blue" baseline. Published results far above that use extra information - player
champion mastery and match history (e.g. Do et al. 2021 report ~75% with player-champion
experience), or in-game statistics. The first version of this project used none of that and
measured exactly this: 53.5% accuracy, with the neural network tied with plain logistic
regression (see "Results so far"). Player history is therefore now an optional second input.

So the useful output is a *calibrated probability*, typically somewhere in 40-60%, not a
confident verdict. The training report therefore leads with log loss, Brier score and
calibration error, and always prints two baselines next to the model so you can see how much
the neural network actually adds. If it does not beat logistic regression on your data, the
report will show that.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
# PyTorch build matching your GPU driver; cu124 suits NVIDIA driver 550+. See pytorch.org.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
pytest            # ~1 min, includes an end-to-end run on synthetic data
```

### Riot API key (free)

Match data comes from the official Riot API. You do not need to buy anything:

1. Sign in at <https://developer.riotgames.com> with a Riot account.
2. Copy the **development key** from the dashboard (valid 24 hours, regenerate as needed), or
   register a **personal API key** product there (also free, does not expire, same rate limits).
3. `cp .env.example .env` and paste the key. `.env` is git-ignored.

Rate limit for both key types: 20 requests/s and 100 requests/2 min *per routing host*.

## Usage

Run all commands from the project root.

```bash
lolwp-collect            # crawl Diamond+ solo queue matches into data/matches.sqlite
lolwp-tune --trials 30   # optional: hyperparameter search, writes models/best_config.json
lolwp-train              # train + evaluate, writes models/draft_model.pt
lolwp-predict --blue A B C D E --red F G H I J
lolwp-draft pick --side blue --role mid     # pick / ban advice, see below
```

### 1. Collect

`lolwp-collect` lists every Challenger, Grandmaster, Master and Diamond solo queue player on
the chosen platforms with their ladder position, then walks their recent ranked games **from
the top of the ladder down**, revisiting each player every 24 hours.

* Default platforms `kr euw1 na1 vn2` sit in the four API regions, each with its own rate
  budget, so they are crawled in parallel: up to ~200 requests/minute in total. It runs until
  stopped; let it run for days. Revisits of high players outrank first visits of lower ones,
  so the crawl settles at the ladder depth the budget can keep fresh.
* Why top-down: the model's player features are built from each player's earlier games *in
  this database*. High-elo players mostly meet each other, so crawling the top first gives
  dense histories; a random crawl gives one known player and nine strangers per game.
  `--order random` restores the old behaviour.
* Safe to stop (Ctrl-C) and rerun at any time; it resumes, and never fetches a match twice.
  With a development key, regenerate the key each day and rerun.
* Only the two newest patches are collected by default (`--patches`); older patches already
  in the database are kept and used for pretraining and player histories.
* Stored per match: patch, champions by side and role, winner, the ten player ids (PUUIDs),
  bans, end-of-game team totals (gold, kills, towers, ...) and per-player numbers
  (`lolwp.riot.PSTAT_FIELDS`): kills / deaths / assists, gold, damage, CS, vision, lane
  leads, plus what is fixed before the game starts (account level, summoner spells, keystone
  rune, and whether the player got their primary role, secondary role or was autofilled).
  Everything that is an outcome reaches the model only through *earlier* games of a player.
* Every ladder snapshot is kept (`ladder_history`), so that rank can later be used as it was
  before a game rather than after it.
* Databases from older versions are upgraded in place. Matches that lack player ids or
  per-player numbers are re-fetched newest first with a share of the request budget
  (`--backfill-share`, default 0.2, `1` = only backfill). One request per match: about
  50 matches a minute per region, so 400k matches take roughly two days at
  `--backfill-share 0.9`. Until then those games train with the inputs they have.

"Diamond+" means every game contains at least one Diamond+ player (the one whose history
surfaced it). Matchmaking keeps lobbies tight, but games near the Emerald/Diamond border
include some Emerald players. With top-down crawling the data is weighted towards Master+.

**Selection bias.** The ladder list is a snapshot. Someone who is Diamond+ at the snapshot
has, on average, been winning shortly before it, so games *older* than the snapshot that were
found through that player lean slightly towards their team - and the player features reveal
who the crawled players are. Games played after the snapshot are unaffected. The crawler
therefore only follows players who join the list later forward in time, and `lolwp-train`
warns when its evaluation games predate the first snapshot. Trust player-feature results only
when that warning is absent.

### 2. Train

`lolwp-train` uses the newest patch plus older ones while fewer than `--min-matches` (400k)
games are available, at most `--max-patches` (4). Use `--patches 16.22` to force a set.
The split is chronological: oldest 80% train, next 10% validation (early stopping and
calibration), newest 10% test (reported once).

Each ensemble member is trained in three steps:

1. **Start from logistic regression.** A regression on champion and champion-in-role one-hots
   is fitted and frozen inside the network as an offset. The network's output starts at zero,
   so it begins exactly at that baseline and learns a correction. Optional (`--lr-offset`):
   it helped on draft-only data and stopped helping once player inputs were added.
2. **Pretrain on all selected patches**, older ones down-weighted by `--patch-decay` (0.8).
3. **Fine-tune on the newest patch** at a lower learning rate; the result is kept only if the
   validation loss improves (`--no-finetune`).

Two things are learned alongside the win/loss label:

* **Player inputs** (only if the database has player ids), `lolwp.data.PLAYER_FEATURES`.
  For each of the ten players:
  * *Record*, from games that started before the one being predicted: games and win rate on
    the picked champion, overall and in today's role, streak and recent form, gold / kill /
    tower margins of their teams, rest since the last game, games in the last 3 and 24
    hours, games shared with today's teammates, champion pool, days since they last played
    the champion.
  * *Performance* in those earlier games, always compared with the opponent in the same
    role: gold, damage, CS, level, vision, KDA, share of team deaths, kill participation,
    objective damage, time dead, lane leads, solo kills, plates, AFK rate; gold and KDA
    also on today's champion; how often they get their primary role or are autofilled.
    The main comparisons are repeated for the last 10 games, for today's role, with the
    typical winner / loser value removed, and against today's lane opponent's champion.
  * *Fixed before the game*: account level, primary / secondary / autofilled role, summoner
    spells, keystone rune.

  These inputs are randomly hidden during training, so the same model also predicts from the
  draft alone.
* **Auxiliary targets**: gold, kill and tower margins and game length. They give the network
  more feedback per game than a single bit and are never inputs (`--aux-weight 0` turns
  them off).

The report compares, on the same test games:

```
prior (blue winrate)
logistic regression, draft
logistic regression, draft + players       (team totals of the player features)
transformer, draft (single, uncalibrated)
transformer ensemble, draft
  same, roles hidden
transformer ensemble, draft + players
```

followed by paired bootstrap confidence intervals for the differences that matter.
`lolwp-tune` runs a random hyperparameter search scored on the validation set and writes
`models/best_config.json` for `lolwp-train --config`. Training five ensemble members on
200k games takes a few minutes on an RTX 2060. For a final model that also learns from the
newest games, rerun with `--test-frac 0` (metrics are then on the validation set and
slightly optimistic).

### 3. Predict

Champions are given in role order **top, jungle, mid, bot, support**; add `--no-roles` if the
order is arbitrary. Names are matched loosely (`leesin`, `"Lee Sin"`, `MonkeyKing`, `Wukong`).

```python
from lolwp.predict import Predictor
p = Predictor.load("models/draft_model.pt")
p_blue = p.predict(["Aatrox", "Lee Sin", "Ahri", "Jinx", "Thresh"],
                   ["Garen", "Vi", "Syndra", "Caitlyn", "Lux"])
```

`lolwp-predict` and `lolwp-draft` currently predict from the draft alone, also with a model
trained on player features. `Predictor.predict(..., player_feats=...)` accepts the features,
but fetching ten players' histories live is not built yet.

Retrain after every patch; the model reports which patch it was trained on.

### 4. Pick and ban advice

`lolwp-draft` turns the win model into recommendations for solo queue champion select.

```bash
# what should I pick mid? (allies as role=Champion; enemy roles are inferred if omitted)
lolwp-draft pick --side blue --role mid --ally top=Aatrox jg=Vi --enemy Ahri Jinx --bans Zed
lolwp-draft pick --side blue --role mid --pool Syndra Orianna Viktor     # only champions I play

# what should I ban? (--ally may hold intended picks, including your own)
lolwp-draft ban --side red --ally mid=Syndra
```

The win model only scores finished 5v5 drafts, so an unfinished draft is valued by sampling:
open slots are filled the way Diamond+ players actually pick in each role (pick shares from
the training games) and the predicted win rate is averaged over 256 such completions.

* **pick** ranks every champion commonly played in your role by that expected win rate, shows
  how far it is above the average pick (`vs field`), how much the ensemble members disagree
  (`+-model`), and for the top few the enemy reply that would hurt most (a one-step lookahead;
  Chen et al. 2018 run a full tree search over a win predictor in the same spirit).
* **ban** scores each champion by `gain = P(enemy picks it) x (win rate without it - win rate
  when the enemy has it)`. `net` also subtracts what your own team loses by not having it,
  which is near zero for champions that are simply strong for whoever gets them; sort by it
  with `--sort net` to find bans that hurt the enemy more than they hurt you.

Caveats: the advice is only as good as the win model, whose edge over a coin flip is small, so
most gaps between neighbouring options are within the `+-model` noise. Ban rates are not used
yet (bans are stored for new matches), so heavily banned champions look less picked (and less
worth banning) than they are. Champions
played in a role in under 0.5% of games are not suggested. Your own skill on a champion
matters far more than these few percentage points.

## Model

A draft is a set of ten tokens: champion embedding + role embedding + team embedding. A small
pre-norm transformer encoder (1 layer, width 64) lets every champion attend
to allies and enemies, so synergies and counters are learned rather than hand-engineered.

Design choices, and where they come from:

| Choice | Why |
|---|---|
| Self-attention over the 10 picks | Same core idea as DraftRec (Lee et al., WWW 2022) and the Set Transformer (Lee et al., ICML 2019); replaces explicit pairwise synergy/counter features (Semenov et al. 2016, Chen et al. 2018). |
| Antisymmetric scoring: `logit = s(blue\|red) - s(red\|blue) + side terms` | "Draft A beats draft B" must flip when sides swap. Sharing one scorer for both teams lets each game teach the network about both drafts; only an explicit bias and per-champion term model the blue/red asymmetry. `--no-antisymmetric` disables it for comparison. |
| Role embeddings with role dropout | Roles come from Riot's `teamPosition`. Randomly hiding them during training makes one model usable with or without roles. |
| Chronological split, newest patch, recency weights | The meta drifts; random splits leak the future and overstate accuracy. |
| Small model, dropout, strong weight decay, early stopping | Roughly 170 champions and a weak signal: overfitting is the main risk. |
| Seed ensemble + Platt scaling on validation data | Cheap variance reduction (Lakshminarayanan et al. 2017) and calibrated probabilities (Guo et al. 2017). |
| Network as a correction on top of logistic regression | On real data the first version only tied the regression. Starting from it means the network cannot be worse at initialisation and spends its capacity on what the linear model misses (the same idea as boosting from a base score). |
| Pretrain on several patches, fine-tune on the newest | Most champion relationships survive a patch, so older games are useful data; fine-tuning then adapts to the current balance. |
| Player-history tokens with dropout | The largest known signal beyond the draft (Do et al. 2021; DraftRec). Computed only from earlier games in the crawl, so no future information leaks in. |
| Auxiliary margin targets | Gold/kill/tower margins say *how* a game was won; multi-task learning with them regularises the shared encoder. |
| Baselines in every report | Logistic regression on champion one-hots is the classic draft model (Conley & Perry 2013) and is hard to beat by much. |

### Results so far

513k Diamond+ games (patches 16.19-16.22, kr/euw1/na1/vn2; 341k with per-player
statistics), tested on the newest 48,956 games, which no model or hyperparameter choice
ever saw:

| Model | Log loss | Accuracy |
|---|---|---|
| Always the blue-side win rate | 0.6928 | 51.4% |
| Logistic regression, draft | 0.6883 | 54.0% |
| Transformer ensemble, draft | 0.6863 | 54.6% |
| Logistic regression, draft + players | 0.6623 | 60.0% |
| Transformer ensemble, draft + players | **0.6577** | **60.7%** |

On the 45,499 test games played after the first ladder snapshot of their platform, which are
free of the selection bias described above, the result is the same: 0.6579 / 60.7%.

How the player inputs got here, same test protocol:

| Player inputs | Accuracy |
|---|---|
| none (draft only) | 54-55% |
| games and win rate on the champion and overall, role share, streak, rest | 56.2% |
| + recent form, team margins of past games, role win rate, session, duo, champion pool | 57.8% |
| + performance vs lane opponent in past games, and inputs fixed before the game | 60.3% |
| + 35k more matches with statistics | 60.6% |
| + recent / in-role / result-adjusted / matchup performance, retuned | 60.7% |

The last step is within noise for accuracy (log loss 0.6584 -> 0.6577): more views of the
same statistics add little, so the curve has flattened. Shuffling one group of inputs
between games shows what the model rests on: the performance history costs 0.024 nats when
shuffled, the whole pre-game block (role assignment, spells, rune, account level) only
0.003. Accuracy barely depends on how many earlier games of a player are stored once there
are five or so.

Things that look better than they are:

* **Current rank / LP.** With the ladder position stored by the crawler, "the team with the
  higher average LP wins" is right in 62-64% of games played *before* the snapshot, because
  the LP already contains their results, and in 52.6% (+-1.6) of the games played after it.
  Rank is therefore not a model input. Published pre-game accuracies of 65-75% usually rest
  on this kind of feature (rank, mastery points or season win rate read after the game).
* **Draft only.** About 54%, and no better with twice the data.

### Prior work

* Conley & Perry, *How Does He Saw Me? A Recommendation Engine for Picking Heroes in Dota 2* (2013) - logistic regression / kNN on hero one-hots.
* Semenov et al., *Performance of Machine Learning Algorithms in Predicting Game Outcome from Drafts in Dota 2* (AIST 2016) - factorization machines for pairwise hero interactions.
* Chen et al., *The Art of Drafting: A Team-Oriented Hero Recommendation System for MOBA Games* (RecSys 2018) - win-rate predictor inside Monte Carlo tree search.
* Do et al., *Using Machine Learning to Predict Game Outcomes Based on Player-Champion Experience in League of Legends* (FDG 2021, arXiv:2108.02799) - shows how much of the predictable signal sits in the players, not the draft.
* Lee et al., *DraftRec: Personalized Draft Recommendation for Winning in MOBA Games* (WWW 2022, arXiv:2204.12750) - transformer over the draft plus player histories.
* Lee et al., *Set Transformer* (ICML 2019); Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017); Lakshminarayanan et al., *Deep Ensembles* (NeurIPS 2017).

## Limitations

* No ranks, bans, pick order, runes or summoner spells as inputs.
* Player features only see games inside the crawl window (about a month) and only for players
  the crawl has covered; a first-time opponent's long-term mastery is invisible.
* Live prediction with player features is not built: the command line tools are draft-only.
* Rarely played champions have poorly estimated effects; `lolwp-predict` flags champions with
  fewer than 50 training games.
* Regions are pooled; regional metas differ somewhat.
* Trained on solo queue; it does not transfer to pro play or lower elo.

## Layout

```
src/lolwp/riot.py       API client, rate limiter, match parsing
src/lolwp/collect.py    crawler (lolwp-collect)
src/lolwp/db.py         SQLite schema
src/lolwp/data.py       patch selection, player-history features, chronological split
src/lolwp/model.py      the transformer
src/lolwp/baselines.py  prior + logistic regression
src/lolwp/train.py      training, calibration, report (lolwp-train)
src/lolwp/tune.py       hyperparameter search (lolwp-tune)
src/lolwp/predict.py    CLI + Python API (lolwp-predict)
src/lolwp/draft.py      pick / ban recommendations (lolwp-draft)
src/lolwp/synth.py      synthetic data for tests / dry runs
```

## Legal

This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot
Games or anyone officially involved in producing or managing Riot Games properties. Riot Games
and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

Use of the Riot API is subject to Riot's developer terms and policies. Do not commit your API
key or redistribute the collected match database.
