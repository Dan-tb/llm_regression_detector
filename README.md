# Model Regression Detection System

This repo runs a fixed golden dataset of support emails through our email-classifier prompt every time a file in `prompts/` changes. It compares the result with the previous accepted run, flags regressions (cases that used to pass and now fail), grades the outcome as pass / warn / critical against configurable thresholds, posts a summary on the PR, alerts Slack, and blocks the merge on critical regressions. It also tracks a rolling average so slow, per-PR-invisible degradation still raises a flag.

## Layout

| Path | Purpose |
|---|---|
| `prompts/*.yaml` | Versioned prompts (version, timestamp, model, system prompt, few-shot). This is what CI tests. |
| `data/golden_v1.json` | Golden dataset, itself versioned via `dataset_version`. |
| `regression/` | Pipeline: `runner` -> `compare` -> `report` / `alerts`, persisted by `store`. |
| `results/` | `runs.db` (SQLite history), `report.html`, `summary.md`. Git-ignored; cached in CI. |
| `.github/workflows/eval.yml` | PR gate. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q

# Offline, no API key, deterministic (good for trying the system):
python -m regression.cli --prompt prompts/v1.yaml --provider mock   # records the baseline
python -m regression.cli --prompt prompts/v2.yaml --provider mock   # simulated bad prompt -> critical, exit 1

# Real run (default provider is Gemini; --model overrides the YAML, e.g. --model gemini-3.8-flash):
export GEMINI_API_KEY=...
python -m regression.cli --prompt prompts/v1.yaml
```

Open `results/report.html` for the diff. The exit code is `1` only for a critical regression.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | - | Required for `--provider gemini` (the default). Get one at https://aistudio.google.com/apikey. |
| `SLACK_WEBHOOK_URL` | unset | If set, a structured alert is posted. Alert failure never changes the exit code. |
| `REPORT_URL` | unset | Link included in the Slack message (CI sets it to the workflow run). |
| `WARN_THRESHOLD` | `0.03` | Absolute pass-rate drop that warns. |
| `CRITICAL_THRESHOLD` | `0.08` | Absolute pass-rate drop that fails the build. |
| `DRIFT_WINDOW` | `7` | Number of runs in the moving average. |
| `DRIFT_FLOOR` | `0.90` | Moving average below this raises a slow-drift warning. |

The CLI automatically loads variables from the project `.env` file. Values already exported in the environment take precedence.

## Adding test cases

1. Add an object to `cases` in the dataset file with a stable, unique `id` (e.g. `bill-007`), `input`, `expected_category`, `expected_summary`, `expected_difficulty` (`easy|medium|hard`) and a `notes` line saying why the case matters.
2. Write cases and labels by hand. Do not generate them with an LLM: the dataset is the ground truth, and the eval can only be as good as it is.
3. Bump `dataset_version`. The baseline lookup is scoped to the current `dataset_version`, so a changed eval bar never gets diffed against an older one; the first run after a bump becomes the new baseline.
4. When a real production failure is found, add it as a case. This is how the set grows beyond the seed.

The shipped `golden_v1.json` is a 24-case seed (including short, typo-ridden, mixed-language, sarcastic and ambiguous emails). The target is 50-100; treat it as a starting point and expand it from your own emails.

## Adjusting thresholds

Set `WARN_THRESHOLD` / `CRITICAL_THRESHOLD` in the workflow `env` or the container. With 24 cases one flip is about 4.2 points, so the defaults make a single regression a warning and two a critical failure. That is intentional on a small set but noisy; raise the thresholds or grow the dataset rather than ignoring alerts. Compare thresholds against dataset size before changing them.

## Architecture decisions

- **Prompt as a parameter.** `classify_email(email, config, provider)` takes a `PromptConfig`, so CI swaps prompts without touching code and the feature is testable with a mock provider.
- **Pass/fail is category match only.** It is deterministic. Summary quality (LLM judge, 1-5), latency and tokens are recorded and diffed but do not gate, because judge scores are noisy. Promote them to gates once you have measured their variance.
- **Mock provider.** A keyword classifier and word-overlap judge let the whole pipeline (including CI unit tests and demos) run offline. Its numbers say nothing about real model quality. `mock_profile` in the prompt YAML exists only to simulate a bad prompt.
- **SQLite for history, JSON for data.** No infrastructure, one portable file, easy to cache in CI. Concurrent writers are not supported; one pipeline per database.
- **Bounded async concurrency.** A semaphore (default 8) keeps within provider rate limits while cutting wall-clock time.
- **Slow-drift check.** Per-run diffs only compare against the last run, so a series of small drops can pass every time. The moving average over the last `DRIFT_WINDOW` runs catches that.
- **Invalid model output counts as a failure**, with the parsing error recorded, rather than crashing the run.

## CI behaviour and caveats

- The workflow triggers on PRs touching `prompts/**`, evaluates the first changed prompt file, comments the summary, uploads `results/`, and fails the job on a critical regression. Make the `eval` job a required status check in branch protection to block merges.
- The baseline is the previous run in `results/runs.db`, restored from the Actions cache. A PR run is saved to history even if the PR is later abandoned; if that matters, run the eval with `--no-save` on PRs and save only on merge to main.
- Secrets (`GEMINI_API_KEY`, `SLACK_WEBHOOK_URL`) are not available to PRs from forks.

## Docker

```bash
docker build -t llm-regression .
docker run --rm -v "$PWD/results:/app/results" \
  -e GEMINI_API_KEY -e SLACK_WEBHOOK_URL -e CRITICAL_THRESHOLD=0.08 \
  llm-regression --prompt prompts/v2.yaml
```
