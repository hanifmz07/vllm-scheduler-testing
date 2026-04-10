# vllm-scheduler-testing
Experiments on vLLM scheduler


### Test cases
- Tokens length x to y
- y >= x
- Must handle where tokens length are the same

### Results

#### Test Cases
```
[{'estimated_tokens': 6, 'prompt': 'Write a haiku about city rain.'},
 {'estimated_tokens': 13,
  'prompt': 'Describe your ideal breakfast in three vivid sentences. breakfast in three vivid sentences.'},
 {'estimated_tokens': 3, 'prompt': 'Summarize this week'},
 {'estimated_tokens': 22,
  'prompt': 'Explain blockchain to a teenager using sports analogies. blockchain to a teenager using sports analogies. blockchain to a teenager using sports analogies.'},
 {'estimated_tokens': 27,
  'prompt': "Draft a polite message asking to reschedule tomorrow's meeting. Draft a polite message asking to reschedule tomorrow's meeting. Draft a polite message asking to reschedule tomorrow's meeting."},
 {'estimated_tokens': 12,
  'prompt': 'Give five quick tips to improve sleep quality tonight. sleep quality tonight.'},
 {'estimated_tokens': 9,
  'prompt': 'Invent a product name for smart reusable water bottles.'},
 {'estimated_tokens': 16,
  'prompt': 'Plan a two-hour study session for calculus practice. Plan a two-hour study session for calculus practice.'},
 {'estimated_tokens': 12,
  'prompt': 'Write a short dialogue between a robot and gardener. robot and gardener.'},
 {'estimated_tokens': 5, 'prompt': 'Create a travel checklist for'}]
```

#### Longest Prompt First
Scheduler: `spt_scheduler.LongestPromptFirstScheduler`

| Rank | Original ID | Tokens Length |
|---|---:|---:|
| 1 | 0 | 8 |
| 2 | 4 | 36 |
| 3 | 3 | 29 |
| 4 | 7 | 20 |
| 5 | 8 | 16 |
| 6 | 1 | 15 |
| 7 | 5 | 14 |
| 8 | 6 | 11 |
| 9 | 2 | 5 |
| 10 | 9 | 5 |


### Notes
- Scheduler states
```py
WAITING = enum.auto()
WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
WAITING_FOR_REMOTE_KVS = enum.auto()
WAITING_FOR_STREAMING_REQ = enum.auto()
RUNNING = enum.auto()
PREEMPTED = enum.auto()
# Note: anything after PREEMPTED will be considered
# as a finished status.
FINISHED_STOPPED = enum.auto()
FINISHED_LENGTH_CAPPED = enum.auto()
FINISHED_ABORTED = enum.auto()
FINISHED_IGNORED = enum.auto()
FINISHED_ERROR = enum.auto()
FINISHED_REPETITION = enum.auto()
```
- How does vLLM scheduler works (oversimplified)
    1. New inserted requests becomes WAITING
    2. Loops RUNNING requests first (preempts if allocation fails)
    3. Promotes WAITING requests to RUNNING
    4. Speculative decode branch during schedule
    5. Built the output object
    6. Stop condition and finish
    7. Next iteration
- `vllm serve` is the cli version of running `AsyncLLM`(or `AsyncLLMEngine`, both are alias to each other). So I decided to keep using `AsyncLLMEngine` instead of `vllm serve` for easier testing.
- `LLM` class used for offline inference, unsuitable for testing the scheduler (?) 
- Do we need to patch the `schedule()` method for this case? Because we could just use existing priority queue implementation, but change the priority metric based on its length. (`main.py`)
    - Implementation in `main.py`
- Patched version in `main_patch.py` -> Still buggy for some reason, the order is not changing


