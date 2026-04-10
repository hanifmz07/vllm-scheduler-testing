from dotenv import load_dotenv
import os
import asyncio
import time
import multiprocessing as mp
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
load_dotenv()

CACHE_DIR = os.getenv("CACHE_DIR")
os.environ["VLLM_CPU_KVCACHE_SPACE"] = "2"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"

def _apply_lpf_patch() -> None:
    """Apply LPF patch to the AsyncScheduler."""
    if getattr(AsyncScheduler, "_lpf_patch_applied", False):
        return

    orig_init = AsyncScheduler.__init__
    orig_add_request = AsyncScheduler.add_request

    def _lpf_prompt_len(req):
        n = getattr(req, "num_prompt_tokens", None)
        if n is None:
            pt = getattr(req, "prompt_token_ids", None) or []
            n = len(pt)
        return int(n)

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self.policy = SchedulingPolicy.PRIORITY
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)

        print("LPF Patch Applied: Scheduler now uses PRIORITY queue and add_request is patched to set priority based on prompt length.")

    def patched_add_request(self, request):
        # The monkey patch applies the priority here!
        request.priority = -_lpf_prompt_len(request)
        return orig_add_request(self, request)

    AsyncScheduler.__init__ = patched_init
    AsyncScheduler.add_request = patched_add_request
    AsyncScheduler._lpf_patch_applied = True

async def test_scheduler():
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2-0.5B",
        download_dir=CACHE_DIR,
        max_model_len=1024,
        enforce_eager=True,
        max_num_seqs=20,
        # scheduling_policy="priority", # Enable native priority queue 
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
        
        s_params = SamplingParams(temperature=0.7, max_tokens=20 if is_blocker else 1)
        
        results_generator = engine.generate(engine_input, s_params, request_id=str(prompt_id))
        
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


    # Create the task list with the test prompts
    tasks = [run_prompt(i, p) for i, p in enumerate(prompts)]
    
    print("Submitting ALL prompts simultaneously so the queue builds up...")
    
    await asyncio.gather(*tasks)
    
    print("\n===== Actual Scheduler Execution Order =====")
    sorted_tracker = sorted(tracker, key=lambda x: x["started"])
    
    for i, t in enumerate(sorted_tracker):
        print(f"Order: {i+1:<2} | Original ID: {t['id']:<2} | Tokens Length: {t['length']:<3}")

def main():
    _apply_lpf_patch()
    asyncio.run(test_scheduler())

if __name__ == "__main__":
    mp.freeze_support()
    main()