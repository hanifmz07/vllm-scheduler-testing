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
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

async def test_scheduler():
	engine_args = AsyncEngineArgs(
		model="Qwen/Qwen2-0.5B",
		download_dir=CACHE_DIR,
		max_model_len=1024,
		enforce_eager=True,
		max_num_seqs=1, # Force 1-by-1 processing to test queue order
		scheduling_policy="fcfs",
		scheduler_cls="custom_scheduler.LongestPromptFirstScheduler",
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
		"Create a travel checklist for ",
	]
	sampling_params = SamplingParams(temperature=0.7, max_tokens=3)
	
	# Store our custom timing metrics
	tracker = []

	async def run_prompt(prompt_id, prompt_text):
		token_length = len(tokenizer.encode(prompt_text, add_special_tokens=False))
		engine_input = engine.renderer.render_cmpl([{"prompt": prompt_text}])[0]
		results_generator = engine.generate(engine_input, sampling_params, request_id=str(prompt_id))
		
		first_token_time = None
		
		# Stream the results
		async for request_output in results_generator:
			if first_token_time is None:
				# The moment the first token yields, the scheduler has processed it
				first_token_time = time.time() 
				
		finish_time = time.time()
		tracker.append({
			"id": prompt_id,
			"length": token_length,
			"started": first_token_time,
			"finished": finish_time
		})

	print("Submitting prompts to Async Engine...")
	tasks = [run_prompt(i, p) for i, p in enumerate(prompts)]
	await asyncio.gather(*tasks)
	
	# 3. Sort and print by the exact time the scheduler started them
	print("\n===== Actual Scheduler Execution Order =====")
	sorted_tracker = sorted(tracker, key=lambda x: x["started"])
	
	for i, t in enumerate(sorted_tracker):
		duration = t["finished"] - t["started"]
		print(f"Order: {i+1} | Original ID: {t['id']} | Tokens Length: {t['length']:<3} | Duration: {duration:.4f}s")

def main():
	asyncio.run(test_scheduler())

if __name__ == "__main__":
	mp.freeze_support()
	main()