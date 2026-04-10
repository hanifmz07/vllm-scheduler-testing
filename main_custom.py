from dotenv import load_dotenv
import os
import asyncio
import time
import argparse
import json
from pathlib import Path
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import pandas as pd

load_dotenv()
CACHE_DIR = os.getenv("CACHE_DIR")
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "8"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run vLLM scheduler tests with configurable options.")
    parser.add_argument(
        "--scheduler-type",
        choices=["fcfs", "priority", "longest-first"],
        required=True,
        help="Scheduler type to run: fcfs, priority, or longest-first.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        required=True,
        help="Maximum number of concurrent sequences.",
    )
    parser.add_argument(
        "--test-case-path",
        type=Path,
        required=True,
        help="Path to a JSON test case file containing prompts.",
    )
    parser.add_argument(
        "--max-tokens-generated",
        type=int,
        required=True,
        help="Maximum number of generated tokens per prompt.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save metrics as CSV when set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for CSV files. Used only when --save-results is set.",
    )
    return parser.parse_args()

def _load_prompts(test_case_path: Path) -> list[str]:
    with test_case_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("Test case JSON must be a list of prompts or prompt objects.")

    prompts: list[str] = []
    for item in payload:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            prompts.append(item["prompt"])
        else:
            raise ValueError("Each test case item must be a string or an object with a string 'prompt' field.")

    if not prompts:
        raise ValueError("No prompts found in test case file.")

    return prompts


async def test_scheduler(args: argparse.Namespace):

    # Configure scheduler based on selected type
    scheduler_kwargs: dict[str, str] = {}
    if args.scheduler_type == "longest-first":
        scheduler_kwargs["scheduler_cls"] = "custom_scheduler.LongestPromptFirstScheduler"
        scheduler_kwargs["scheduling_policy"] = "fcfs"
    else:
        scheduler_kwargs["scheduling_policy"] = args.scheduler_type

    # Initialize the engine with the specified scheduler configuration
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2-0.5B",
        download_dir=CACHE_DIR,
        max_model_len=1024,
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=False, # To observe clearer scheduling effects without caching optimizations
        **scheduler_kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = engine.get_tokenizer()
    prompts = _load_prompts(args.test_case_path)

    # Warmup to initialize model and memory, ensuring more accurate performance metrics for the actual test prompts
    print("Running warmup prompt to initialize PyTorch/vLLM memory...")
    warmup_input = engine.renderer.render_cmpl([{"prompt": "Warmup"}])[0]
    warmup_gen = engine.generate(warmup_input, SamplingParams(max_tokens=1), request_id="warmup")
    async for _ in warmup_gen:
        pass
    print("Warmup complete. Starting benchmark.")

    # Tracker for performance metrics
    tracker = []

    # Define the async function to run each prompt and collect metrics
    async def run_prompt(prompt_id, prompt_text, is_blocker=False):
        token_length = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        engine_input = engine.renderer.render_cmpl([{"prompt": prompt_text}])[0]
        
        s_params = SamplingParams(
            temperature=0.7,
            max_tokens=20 if is_blocker else args.max_tokens_generated,
        )
        
        submission_time = time.time()
        
        results_generator = engine.generate(engine_input, s_params, request_id=str(prompt_id))
        
        first_token_time = None
        output_tokens = 0
        
        async for request_output in results_generator:
            if first_token_time is None:
                first_token_time = time.time() 
            output_tokens = len(request_output.outputs[0].token_ids)
                
        finish_time = time.time()
        
        if not is_blocker:
            ttft = first_token_time - submission_time
            e2e_latency = finish_time - submission_time
            generation_time = finish_time - first_token_time
            
            tpot = (generation_time / (output_tokens - 1)) if output_tokens > 1 else generation_time

            tracker.append({
                "id": prompt_id,
                "length": token_length,
                "ttft": ttft,
                "tpot": tpot,
                "e2e": e2e_latency,
                "started": first_token_time # Still keeping this for sorting
            })

    print("Submitting prompts...")
    tasks = [run_prompt(i, p) for i, p in enumerate(prompts)]
    
    # Await all prompt tasks to complete
    await asyncio.gather(*tasks)
    
    # Print the execution order based on actual start times
    print("\n========== Scheduler Performance Metrics ==========")
    print(f"{'Order':<6} | {'ID':<3} | {'Tokens':<6} | {'TTFT (s)':<9} | {'TPOT (s)':<9} | {'E2E (s)':<9}")
    print("-" * 55)
    
    # Sort the tracker by actual start time to show the order of execution
    sorted_tracker = sorted(tracker, key=lambda x: x["started"])
    
    # Print the metrics for each prompt in the order they started generating tokens
    for i, t in enumerate(sorted_tracker):
        print(f"{i+1:<6} | {t['id']:<3} | {t['length']:<6} | {t['ttft']:<9.4f} | {t['tpot']:<9.4f} | {t['e2e']:<9.4f}")
    
    # Save the metrics to CSV if the flag is set
    if args.save_results:
        output_path = os.path.join(
            args.output_dir,
            os.path.basename(args.test_case_path).replace(".json", ""),
            f"max_num_seqs_{args.max_num_seqs}",
            f"max_tokens_{args.max_tokens_generated}",
            f"{args.scheduler_type}_metrics.csv"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(tracker)
        df.to_csv(output_path, index=False)
        print(f"\nMetrics saved to {output_path}")

def main():
    args = parse_args()
    asyncio.run(test_scheduler(args))

if __name__ == "__main__":
    main()