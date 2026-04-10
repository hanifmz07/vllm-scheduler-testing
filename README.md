# vllm-scheduler-testing

Scheduler-focused benchmarking for vLLM CPU inference.

This repository compares queue behavior and latency under different scheduling strategies, with a custom longest-prompt-first scheduler implemented on top of vLLM AsyncScheduler.

## What this repo does

- Runs async request workloads against vLLM using configurable scheduler settings.
- Measures per-request latency metrics:
	- TTFT (time to first token)
	- TPOT (time per output token)
	- E2E latency
- Supports built-in and custom schedulers:
	- fcfs
	- priority
	- longest-first (custom queue policy from prompt length)

## Repository layout

- [main_custom.py](main_custom.py): Main benchmark entrypoint with CLI arguments.
- [custom_scheduler.py](custom_scheduler.py): Prompt-length request queue and custom AsyncScheduler classes.
- [scripts/run_scheduler_test.sh](scripts/run_scheduler_test.sh): Grid-search style runner for multiple test cases and params.
- [test_cases](test_cases): Prompt datasets used for experiments.
- [results](results): Generated CSV outputs (created/populated by runs).
- [experiment_notes](experiment_notes): Notes and historical observations.
- [setup.sh](setup.sh): Environment bootstrap commands.

## Prerequisites

- Linux host with Python 3.12
- CPU build support for vLLM
- numactl (recommended for consistent CPU benchmark runs)
- uv package manager

## Setup

1. Create and activate virtual environment.
2. Install vLLM CPU wheel and dependencies.
3. Configure environment variables.

Quick start:

```bash
uv venv --python 3.12
source .venv/bin/activate

bash setup.sh
```

Environment file example:

```env
CACHE_DIR=/path/to/model/cache
PARENT_DIR=/path/to/parent
```

You can copy values from [.env.example](.env.example) and place them in .env.

## Running a single benchmark

Basic command:

```bash
source .venv/bin/activate
.venv/bin/python main_custom.py \
	--scheduler-type longest-first \
	--max-num-seqs 5 \
	--test-case-path test_cases/prompts_20.json \
	--max-tokens-generated 50
```

Save CSV output:

```bash
.venv/bin/python main_custom.py \
	--scheduler-type fcfs \
	--max-num-seqs 10 \
	--test-case-path test_cases/prompts_100.json \
	--max-tokens-generated 100 \
	--save-results \
	--output-dir results
```

## CLI arguments

- --scheduler-type: fcfs | priority | longest-first
- --max-num-seqs: Maximum number of concurrent sequences.
- --test-case-path: JSON file path containing prompt inputs.
- --max-tokens-generated: Max output tokens per request.
- --save-results: Flag to enable CSV export.
- --output-dir: Base directory for CSV results (used only with --save-results).

## Batch experiments

Use the provided script to run multiple combinations of:

- test case size
- max_num_seqs
- max tokens per sequence
- scheduler type

Run:

```bash
chmod +x scripts/run_scheduler_test.sh
./scripts/run_scheduler_test.sh
```

The script uses numactl to pin compute and memory for more stable CPU benchmark numbers.

## Test case format

[main_custom.py](main_custom.py) accepts either:

1. A JSON list of strings
2. A JSON list of objects with a prompt field

Example:

```json
[
	"Write a haiku about city rain.",
	{"prompt": "Explain blockchain to a teenager using sports analogies."},
    ...
]
```

## Output format

When enabled, results are saved as CSV under a parameterized path:

`results/<test_case_name>/max_num_seqs_<N>/max_tokens_<M>/<scheduler>_metrics.csv`

Columns:

- id: Original prompt index
- length: Input token length
- ttft: Time to first token (seconds)
- tpot: Time per output token (seconds)
- e2e: End-to-end latency (seconds)
- started: Timestamp of first generated token (used for ordering)

## Custom scheduler notes

- The custom longest-first policy is implemented in [custom_scheduler.py](custom_scheduler.py).
- It uses a heap-backed queue keyed by prompt length, then arrival time, then request id.
- Removal is lazy (active-set based), with stale heap cleanup before peek/pop.

## Related notes

- Experiment writeups: [experiment_notes](experiment_notes)
- Extra scheduler behavior notes: [notes.md](experiment_notes/notes.md)
- WARNING: Current results maybe not reliable, because it is ran in a shared machine with user(s) running multiple processes at the same time.

