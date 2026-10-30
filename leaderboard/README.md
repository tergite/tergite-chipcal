# Submitting to the leaderboard

The leaderboard is [LEADERBOARD.md](../LEADERBOARD.md), generated from the folders in [submissions/](submissions).
An entry is a pull request that adds one folder. CI validates it, recomputes every number from the submitted episodes
and replays a few of them; a maintainer merges it.

## 1. Run on the frozen chips

Evaluate your model on all 100 frozen chips of a tier with the standard settings:

```bash
python scripts/evaluate.py --policy model --model <model> --base-url <endpoint> --tier 2q1c --n 100 --out results/my-model
```

Keep discovery on and the tier's turn cap; do not edit the environment. A strategy prompt (`--prompt-addendum`)
puts the entry in the **guided** track, and its sha256 goes in `meta.json`. Without one, the entry is in the **raw**
track. Report the sampling settings you used (temperature, thinking budget, vision or `--text-only`).

## 2. Build the submission folder

```bash
python scripts/leaderboard.py new --from results/my-model --kind model --model <model> --name "My model" \
    --track raw --submitter "My lab" --settings '{"temperature": 0.7, "thinking_tokens": 4096, "visual": true}' \
    --out leaderboard/submissions/2026-11-01-my-model
```

This writes `meta.json`, `episodes.jsonl` (the evaluation rows) and `actions.jsonl` (every episode's actions, from
the transcripts under `results/my-model/episodes/`). Transcripts with the model's reasoning are welcome but not
required: link them in `meta.json` (`link`).

## 3. Check it and open a pull request

```bash
python scripts/leaderboard.py validate
python scripts/leaderboard.py verify leaderboard/submissions/2026-11-01-my-model --chips 2
python scripts/leaderboard.py render
```

`render` rewrites `LEADERBOARD.md` and the chart at the top of the main README (`docs/leaderboard-2q1c-*.svg`).
Commit the folder and the regenerated files, and open a pull request.

## How entries are checked

- Every number on the board is recomputed from `episodes.jsonl`; a summary you write yourself is not used.
- The simulator is deterministic given a chip and an action sequence. `verify` replays the submitted actions of
  chips chosen from the folder name and checks that they reach the recorded scores, so a score cannot be edited
  without an action sequence that earns it. CI replays 2 chips; maintainers may replay all of them before merging.
- A code baseline (`--kind code`) is checked by rerunning its policy on those chips.
- Entries on fewer than all frozen chips are listed as partial and not ranked.

Rules: use only what the environment shows the agent (no hidden device parameters, no oracle), the frozen chips of
the stated benchmark version, and the standard environment settings.
