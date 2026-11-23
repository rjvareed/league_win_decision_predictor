# League of Legends win predictor and champion-select assistant

Two things built on one model:

* **A win predictor.** The probability that blue side / red side wins a ranked solo queue
  game, before it starts. Inputs are the ten champions and their sides (optionally roles)
  and, when available, each player's recent match history. Trained on Diamond+ games from
  the newest patches, on a single consumer GPU.
* **A champion-select assistant** (`lolwp-app`, in the browser, or `lolwp-draft` on the
  command line). Enter the bans and picks you can see and it ranks the picks for your role
  and the bans worth making, and shows the win chance of the draft as it stands. Give it
  your Riot ID and the ranking becomes personal: it knows which champions you have been
  playing and how that went.

```
$ lolwp-app
champion select assistant: http://127.0.0.1:8765  (model: patch 16.23, 16.22, 16.21; Ctrl-C to stop)

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

For someone in champion select this has a practical consequence. Which champion you pick
moves the model's win chance by two or three points between a good and an average pick -
and *whether you have been playing it* moves the real win rate by more: in these games a
player on a champion they have not touched in the last month wins 46.9% of the time, one
with more than 20 recent games on it 52.1% ("Results"). The assistant therefore ranks picks
with your own record when you let it, and is best used to break ties between champions you
already play.

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
lolwp-app                # champion-select assistant in the browser, see below
lolwp-draft pick --side blue --role mid     # the same advice on the command line
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

Each ensemble member is trained in these steps:

1. **Start from logistic regression.** A regression on champion and champion-in-role one-hots
   is fitted and frozen inside the network as an offset. The network's output starts at zero,
   so it begins exactly at that baseline and learns a correction. Optional (`--lr-offset`):
   it helped on draft-only data and stopped helping once player inputs were added.
2. **Pretrain on all selected patches**, older ones down-weighted by `--patch-decay` (0.8).
3. **Fine-tune on the newest patch** at a lower learning rate; the result is kept only if the
   validation loss improves (`--no-finetune`).
4. **Refit with the validation games** (`--no-refit` to skip): a fresh model is trained on
   training + validation games for exactly the number of epochs step 2 and 3 settled on.
   The validation games are the newest ones before the test period, and more games is the
   one thing that reliably makes this model better ("Results"). Nothing is selected on them
   in this step, and the calibration still comes from the models that never saw them.

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

  These inputs are randomly hidden during training - for all ten players in some games
  (`--player-dropout`), for a random one to nine of them in others (`--player-partial`) - so
  the same model also predicts from the draft alone, and from the draft plus the one player
  the assistant can know in champion select: you.
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
transformer ensemble, draft + one known player   (what the assistant sees with your Riot ID)
transformer ensemble, draft + players
```

followed by paired bootstrap confidence intervals for the differences that matter.
`lolwp-tune` runs a random hyperparameter search scored on the validation set and writes
`models/best_config.json` for `lolwp-train --config`. Training five ensemble members on
500k games takes about an hour on an RTX 2060 (nearly half of it the refit). For a model
to *use* rather than to report on, rerun with `--test-frac 0`: it then also learns from the
newest games, which are the ones played on the current patch (metrics are then on the
validation set, from the models before the refit, and slightly optimistic).

### 3. Predict

Champions are given in role order **top, jungle, mid, bot, support**; add `--no-roles` if the
order is arbitrary. Names are matched loosely (`leesin`, `"Lee Sin"`, `MonkeyKing`, `Wukong`).

```python
from lolwp.predict import Predictor
p = Predictor.load("models/draft_model.pt")
p_blue = p.predict(["Aatrox", "Lee Sin", "Ahri", "Jinx", "Thresh"],
                   ["Garen", "Vi", "Syndra", "Caitlyn", "Lux"])
```

`lolwp-predict` predicts from the draft alone, also with a model trained on player features.
`Predictor.predict(..., player_feats=...)` accepts the features; `lolwp.live` computes them
for one player from their Riot ID (this is what the assistant uses for you). Fetching all ten
players of a lobby is not built: solo queue hides the other names in champion select anyway.

Retrain after every patch; the model reports which patch it was trained on.

### 4. Champion-select assistant

```bash
lolwp-app                 # then open http://127.0.0.1:8765
```

A local web page for champion select. Set your side and role, then click champions into the
ban row, your team and the enemy team as they appear (enemy roles are guessed and can be
corrected). After every change the page shows:

* **the win chance of the draft as it stands**, open slots filled the way Diamond+ players
  pick;
* **the best picks for your role**, with how far each is above the average pick, how much the
  ensemble members disagree (`±`), and the enemy reply that would hurt it most. Click a row to
  lock it in. "Only my pool" restricts the list to the champions you play;
* **the best bans**: `gain = P(an enemy picks it if you do not ban it) x (how much that costs
  you)`. Put the champion you intend to play into your slot first and the bans protect that
  pick.

**Your match history (optional).** Enter your Riot ID and server and press "Load my games".
The assistant fetches your ranked solo games of the last 30 days (at most 40; one API request
per game the first time, a few seconds, then cached in `data/live_cache.sqlite`) and computes
for every candidate the same inputs the model was trained with: your games and results on
the champion, how you did against lane opponents, form, role, and so on. Picks are then ranked
for *you*; the `Draft only` column shows the same pick for an anonymous player, so you can see
what your record changes. Champions you have not played recently lose a few points, as they
do in real games. This needs a Riot API key in `.env` (the same one the crawler uses; with a
development key, renew it daily) and a model trained with player features.

The page talks only to the server on your machine. Champion icons come from Riot's Data
Dragon CDN; your Riot ID is sent to the Riot API and nowhere else. The server listens on
localhost unless you pass `--host`.

**Command line.** The same engine without the page:

```bash
# what should I pick mid? (allies as role=Champion; enemy roles are inferred if omitted)
lolwp-draft pick --side blue --role mid --ally top=Aatrox jg=Vi --enemy Ahri Jinx --bans Zed
lolwp-draft pick --side blue --role mid --pool Syndra Orianna Viktor     # only champions I play
lolwp-draft pick --side blue --role mid --me "Name#TAG" --platform euw1  # ranked for me

# what should I ban? (--ally may hold intended picks, including your own)
lolwp-draft ban --side red --ally mid=Syndra
```

**How it decides.** The win model only scores finished 5v5 drafts, so an unfinished draft is
valued by sampling: open slots are filled the way Diamond+ players actually pick in each role
and the predicted win rate is averaged over those completions (256 on a GPU, 96 on a CPU; the
same completions for every candidate, so candidates are compared on identical games).

* **pick** ranks every champion commonly played in your role (plus, with your history, the
  ones you play) by that expected win rate. For the top few it also searches the enemy's open
  roles for the reply that would hurt most (a one-step lookahead; Chen et al. 2018 run a full
  tree search over a win predictor in the same spirit).
* **ban**: `net` also subtracts what your own team loses by not having the champion, which is
  near zero for champions that are simply strong for whoever gets them (`--sort net`, or "Net
  of own loss" on the page).
* **Ban rates.** A champion that is banned in 40% of games is picked more often than its raw
  pick share says in the games where it gets through. The training games' bans are counted,
  and once the bans of the current draft are known (six or more entered) open slots are
  sampled from pick shares *given availability*. Before that, raw shares are the right ones:
  they already contain the chance that somebody else bans the champion.

Caveats: the advice is only as good as the win model, whose edge over a coin flip is small, so
most gaps between neighbouring options are within the `±` noise, and the gain of any single
ban is a fraction of a point. Champions played in a role in under 0.5% of games are not
suggested unless you play them. The model knows what you have played in the last month, not
how good you are on a champion you last touched a season ago.

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
| Hiding a random subset of players, not only all ten | The first versions only ever saw all ten players or none, so "I know one player" was a situation the model had never been trained on - exactly the one the assistant puts it in. |
| Refit on training + validation games | The validation games are 11% of the usable data and the most recent; the learning curve is still rising (see below). Epoch counts come from the run that held them out. |
| Auxiliary margin targets | Gold/kill/tower margins say *how* a game was won; multi-task learning with them regularises the shared encoder. |
| Baselines in every report | Logistic regression on champion one-hots is the classic draft model (Conley & Perry 2013) and is hard to beat by much. |

### Results so far

522,271 Diamond+ games (patches 16.21-16.23, kr/euw1/na1/vn2), tested on the newest 52,227:

| Model | Log loss | Accuracy |
|---|---|---|
| Always the more frequent winner (red side) | 0.6927 | 51.6% |
| Logistic regression, draft | 0.6876 | 54.3% |
| Transformer ensemble, draft | 0.6853 | 54.8% |
| Transformer ensemble, draft + one known player | 0.6838 | 55.4% |
| Logistic regression, draft + players | 0.6613 | 60.4% |
| Transformer ensemble, draft + players | **0.6549** | **61.3%** |

"Draft + one known player" is what the champion-select assistant has when you give it your
Riot ID. Practically all of these test games (52,190) were played after the first ladder
snapshot of their platform, so they are free of the selection bias described above.

**Games nobody had seen.** The choices of this round (further down) were compared on the
test period of an earlier snapshot, so that comparison is not blind. The crawler kept
running afterwards and 20,242 games of patch 16.23 arrived that no model, comparison or
person had looked at - a patch both models below had hardly any training games of.

| On those 20,242 games (log loss / accuracy) | Draft | Draft + one player | Draft + players |
|---|---|---|---|
| Previous recipe, trained on the earlier snapshot | 0.6865 / 54.6% | 0.6838 / 55.4% | 0.6558 / 61.3% |
| Current recipe (the model of the table above) | 0.6864 / 54.3% | 0.6839 / 55.3% | 0.6554 / 61.4% |
| Difference in log loss, 95% interval | -0.0004 .. +0.0004 | -0.0004 .. +0.0005 | -0.0010 .. +0.0002 |

Read this soberly: on a new patch the current recipe is **not measurably better** than the
previous one. The gains listed further down were measured on one snapshot; here the point
estimate is about a quarter of them and its interval includes zero. About 61% is what this information supports;
see "what did not" below for the things that were tried to get past it.

The probabilities can be taken at face value. On the unseen games, by how sure the full
model was:

| Model gives the favourite | Share of games | Favourite won |
|---|---|---|
| 50-55% | 25% | 52.6% |
| 55-60% | 23% | 58.2% |
| 60-65% | 19% | 61.6% |
| over 65% | 33% | 70.0% |

**Do the pick ratings mean anything?** For 4,000 of the unseen games, each of the ten picks
was rated the way the assistant rates it: the model's win chance with the actual pick minus
the pick-share weighted average over the usual picks for that role, the other nine champions
as they were, draft only.

| Assistant's rating of the pick | Share of picks | Team won |
|---|---|---|
| 2 points or more below the field | 16% | 45.8% |
| 1 to 2 below | 14% | 48.0% |
| 0 to 1 below | 18% | 49.4% |
| 0 to 1 above | 18% | 50.2% |
| 1 to 2 above | 15% | 52.3% |
| 2 points or more above the field | 20% | 53.4% |

One point of rated value was followed by 1.1 points of actual win rate (each row +-1.2), so
the ratings are about as large as they claim. They are also small: nine picks in ten are
rated within 3.4 points below to 3.9 above the field.

**Does knowing who you are change the estimate for good reason?** On 20,000 of the unseen
games the model was given what the assistant gets: the draft and the history of one of the
ten players, nothing fixed before the game (runes, spells) filled in. Games are grouped by how
far that moved the estimate for the player's team away from the draft-only one:

| Adjustment for the known player | Share of games | Model then says | Team won |
|---|---|---|---|
| 4 points or more down | 15% | 44.3% | 45.2% |
| 2 to 4 down | 16% | 47.1% | 46.3% |
| 0 to 2 down | 21% | 48.8% | 48.6% |
| 0 to 2 up | 22% | 51.0% | 51.4% |
| 2 to 4 up | 15% | 53.1% | 53.5% |
| 4 points or more up | 11% | 56.1% | 55.2% |

(each row +-1.5 to 2; the draft-only estimate is 50% in every row.) One player's record moves
the estimate by more than the whole draft does, and the outcomes follow it.

**Play what you have been playing.** The newest 102,190 games of the earlier snapshot, by
how often a player had been on that champion in the month before (players with at least 20
games in the crawl):

| Games on the champion, last month | Share | Player's team won |
|---|---|---|
| 0 | 13% | 46.9% |
| 1-2 | 15% | 48.4% |
| 3-5 | 13% | 49.3% |
| 6-10 | 13% | 50.0% |
| 11-20 | 14% | 51.2% |
| more than 20 | 33% | 52.1% |

Five points from first to last row, more than the draft offers; this is what the personal
ranking in the assistant feeds on.

**What moved the model this round**, each compared on the same 51,095 test games of the
511k-game snapshot (two-model ensembles, log loss / accuracy):

| Recipe | Draft + players | Draft | Draft + one player |
|---|---|---|---|
| Previous: 40 epochs, all ten players hidden in 35% of training games | 0.6555 / 61.1% | 0.6857 / 54.6% | - |
| + refit on training + validation games | 0.6547 / 61.2% | 0.6854 / 54.9% | 0.6838 / 55.4% |
| 60 epochs instead of 40 | 0.6549 / 61.3% | 0.6854 / 54.8% | 0.6836 / 55.2% |
| 60 epochs, players hidden for all (20%) or for a random part (20%) | 0.6548 / 61.2% | 0.6853 / 54.9% | 0.6831 / 55.6% |

The current recipe is the last row plus the refit.

and what did not:

* **Weighting recent games** (weight halving every five days): 0.6566. The newest games are
  not worth more than the older ones; there are just too few of any.
* **Dropping the oldest 30% of training games** (whose player histories are thinnest):
  0.6571. Together with the refit this is the clearest finding: the model is limited by the
  *number* of games, and the curve is still rising. Let the crawler run.
* **Deeper player histories** are not what it is short of: test accuracy is flat once the
  ten players have five or so earlier games each (61.6% at 5-15 games for the third-thinnest
  history of a lobby, 60.8% at 60+).
* **Hiding a random part of the players in 25% of games without showing the full lobby more
  often**: one known player improves to 0.6831, the full model drops to 0.6562. The shares
  above keep both.
* **A wider network** (64 instead of 32; one seed, 0.0008 worse), and **gradient-boosted trees** on lane-by-lane
  differences of the player inputs (0.6625, level with logistic regression at 0.6616): the
  transformer is not the bottleneck either.

In the previous round the full model scored 0.6577 / 60.7% on its own 48,956 test games; the
test games differ between rounds, so compare within a table, not across.

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

* No ranks or pick order as inputs; bans only shape the assistant's pick shares.
* Player features only see games inside the crawl window (about a month) and only for players
  the crawl has covered; a first-time opponent's long-term mastery is invisible.
* The assistant knows one player (you). It cannot see who your teammates and opponents are.
* The assistant does not read the League client; the draft is entered by hand.
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
src/lolwp/live.py       one player's recent games -> model inputs (Riot ID lookup, cache)
src/lolwp/app.py        champion-select assistant, local web server (lolwp-app)
src/lolwp/static/       its page
src/lolwp/synth.py      synthetic data for tests / dry runs
```

## Legal

This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot
Games or anyone officially involved in producing or managing Riot Games properties. Riot Games
and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

Use of the Riot API is subject to Riot's developer terms and policies. Do not commit your API
key or redistribute the collected match database.
