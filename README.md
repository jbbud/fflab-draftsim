# fflab

`fflab` is a Python CLI package for retroactive fantasy football drafts. It loads a historical NFL season, scores every player week by week under custom league settings, runs a snake draft against heuristic AI opponents, then evaluates rosters with optimal weekly lineups.

## Quick Start

```powershell
python -m pip install -e .[dev]
fflab draft --season 2019 --config examples/league.json
```

## Local GUI

```powershell
python run_gui.py --port 8765
```

Then open `http://127.0.0.1:8765`. The GUI runs with demo data by default, and can use live `nflreadpy` data once optional runtime dependencies are installed.

## Draft Optimizer

Train a fast weighted draft policy with evolutionary self-play:

```powershell
fflab train --season 2019 --config examples/league.json --episodes 200 --population 32 --output models/draft_policy.json
```

Evaluate it:

```powershell
fflab evaluate-policy --season 2019 --config examples/league.json --policy models/draft_policy.json --drafts 100
```

Use it in `ai_policies` as `trained:models/draft_policy.json`. Missing trained-policy files fall back to a safe weighted default.

Train a weekly-vector neural policy:

```powershell
fflab train-neural --season 2019 --config examples/league.json --samples 5000 --epochs 20 --output models/neural_policy.pt
fflab evaluate-neural --season 2019 --config examples/league.json --policy models/neural_policy.pt --drafts 100
```

Add `--profile` to neural training, improvement, and benchmark commands to show Rich progress bars plus phase timings. Use `--rollout-budget` and `--candidate-pool-size` to keep rollout labeling and lookahead bounded during long runs:

```powershell
fflab train-neural --season 2019 --config examples/league.json --samples 5000 --epochs 20 --rollout-budget 8 --candidate-pool-size 12 --profile --output models/neural_policy.pt
```

Improve a neural policy against the exact configured league mix:

```powershell
fflab improve-neural --season 2019 --config examples/league.json --base-policy models/neural_policy.pt --generations 5 --samples-per-generation 6000 --epochs-per-generation 12 --validation-drafts 10 --rollout-budget 8 --candidate-pool-size 12 --profile --output models/neural_policy_champion.pt
fflab benchmark-neural --season 2019 --config examples/league.json --policy models/neural_policy_champion.pt --drafts 100 --profile
```

Check or train for broader league-size robustness:

```powershell
fflab benchmark-neural-variants --season 2019 --config examples/league.json --policy models/neural_policy_champion.pt --league-sizes 4,8,10,12 --drafts 100 --profile
fflab improve-neural --season 2019 --config examples/league.json --base-policy models/neural_policy_champion.pt --robust --league-sizes 4,8,10,12 --generations 5 --samples-per-generation 6000 --epochs-per-generation 12 --validation-drafts 12 --rollout-budget 8 --candidate-pool-size 12 --profile --output models/neural_policy_robust.pt
```

Run an overnight stochastic multi-season curriculum:

```powershell
fflab improve-neural --seasons 2021,2022,2023,2024,2025 --target-season 2025 --config examples/league.json --base-policy models/nn25r.pt --robust --league-sizes 4,10,12,14 --generations 12 --samples-per-generation 30000 --epochs-per-generation 8 --rollouts-per-candidate 3 --behavior-epsilon 0.35 --opponent-temperature 0.18 --candidate-noise-std 0.12 --policy-mix-jitter 0.25 --candidate-pool-size 16 --rollout-budget 12 --validation-drafts 40 --accept-4team-threshold 2876.20 --output models/nn25g.pt --profile
```

Benchmarks use fast pure neural scoring with roster guardrails by default. Add `--lookahead` when you want slower benchmark behavior that more closely matches live Bot picks. Use neural policies in `ai_policies` as `neural:models/neural_policy.pt`. Neural training and benchmarking use the configured `ai_policies` cycle for exact-league rollouts.

Latest local champion policy:

```json
{
  "ai_policies": [
    "best_available",
    "scarcity",
    "balanced",
    "neural:models/nn25g.pt"
  ]
}
```

Use forward slashes in JSON policy paths. The `models/` directory is ignored by git, so trained `.pt` artifacts stay local unless you intentionally publish them somewhere else.

For test or offline runs, pass a fixture directory with `players.csv` and `weekly_scores.csv`:

```powershell
fflab draft --season 2019 --config examples/league.json --fixture-dir path\to\fixture --auto
```

## Data

The live adapter uses `nflreadpy` lazily, so importing and testing the engine does not require network access. Offensive players are loaded from weekly player stats, kickers are loaded from kicking stats when available or derived from play-by-play, and team defenses are synthetic `DEF_*` players derived from schedules and play-by-play.
