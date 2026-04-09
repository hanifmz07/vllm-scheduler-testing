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


#### Shortest Prompt First
Scheduler: `spt_scheduler.ShortestPromptFirstScheduler`

| Rank | Original ID | Tokens Length |
|---|---:|---:|
| 1 | 0 | 8 |
| 2 | 2 | 5 |
| 3 | 9 | 5 |
| 4 | 6 | 11 |
| 5 | 5 | 14 |
| 6 | 1 | 15 |
| 7 | 8 | 16 |
| 8 | 7 | 20 |
| 9 | 3 | 29 |
| 10 | 4 | 36 |


