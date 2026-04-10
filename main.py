from dotenv import load_dotenv
import os
import asyncio
import time
import multiprocessing as mp
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

load_dotenv()

CACHE_DIR = os.getenv("CACHE_DIR")
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "2"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

async def test_scheduler():
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2-0.5B",
        download_dir=CACHE_DIR,
        max_model_len=1024,
        enforce_eager=True,
        max_num_seqs=1, # Force 1-by-1 processing to test queue order
        scheduling_policy="priority", # Enable native priority queue
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = engine.get_tokenizer()
    
    prompts = [
        "Write a haiku about city rain.",
        "Describe your ideal breakfast in three vivid sentences. breakfast in three vivid sentences.",
        "Summarize this week",
        "Explain blockchain to a teenager using sports analogies. blockchain to a teenager using sports analogies. blockchain to a teenager using sports analogies.",
        "Draft a polite message asking to reschedule tomorrow's meeting. Draft a polite message asking to reschedule tomorrow's meeting. Draft a polite message asking to reschedule tomorrow's meeting.",
        "Give five quick tips to improve sleep quality tonight. sleep quality tonight.",
        "Invent a product name for smart reusable water bottles.",
        "Plan a two-hour study session for calculus practice. Plan a two-hour study session for calculus practice.",
        "Write a short dialogue between a robot and gardener. robot and gardener.",
        "Create a travel checklist for",
    ]
    
    tracker = []

    async def run_prompt(prompt_id, prompt_text, is_blocker=False):
        token_length = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        engine_input = engine.renderer.render_cmpl([{"prompt": prompt_text}])[0]
        
        # LPF Logic: Lower number = higher priority in Python's heapq. 
        req_priority = 0 if is_blocker else -token_length
        # req_priority = -token_length
        
        s_params = SamplingParams(temperature=0.7, max_tokens=20 if is_blocker else 1)
        # s_params = SamplingParams(temperature=0.7, max_tokens=20)
        
        # Pass the priority to the engine
        results_generator = engine.generate(
            engine_input, 
            s_params, 
            request_id=str(prompt_id),
            priority=req_priority 
        )
        
        first_token_time = None
        
        async for request_output in results_generator:
            if first_token_time is None:
                first_token_time = time.time() 
                
        finish_time = time.time()
        
        if not is_blocker:
            tracker.append({
                "id": prompt_id,
                "length": token_length,
                "started": first_token_time,
                "finished": finish_time
            })

    blocker_task = asyncio.create_task(
        run_prompt("BLOCKER", "Write", is_blocker=True)
    )
    
    await asyncio.sleep(1)
    
    print("Submitting LPF prompts to the waiting queue...")
    tasks = [run_prompt(i, p) for i, p in enumerate(prompts)]
    
    await asyncio.gather(*tasks)
    # await blocker_task
    
    print("\n===== Actual Scheduler Execution Order =====")
    sorted_tracker = sorted(tracker, key=lambda x: x["started"])
    
    for i, t in enumerate(sorted_tracker):
        print(f"Order: {i+1:<2} | Original ID: {t['id']:<2} | Tokens Length: {t['length']:<3}")

def main():
    asyncio.run(test_scheduler())

if __name__ == "__main__":
    mp.freeze_support()
    main()