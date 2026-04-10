source .venv/bin/activate

# Restrict PyTorch math to the 64 physical cores of a single CPU
export OMP_NUM_THREADS=64

# Prevent PyTorch threads from jumping between cores
export OMP_PROC_BIND=true


TEST_CASES=(
    "test_cases/prompts_10.json"
    "test_cases/prompts_20.json"
    "test_cases/prompts_50.json"
    "test_cases/prompts_100.json"
)

MAX_NUMS_SEQS=(1 5 10 20)
MAX_TOKENS_PER_SEQ=(1 10 50 100)
OUTPUT_DIR="results"
SCHEDULER_TYPES=("fcfs" "longest-first")

echo "Pre-loading model weights into RAM cache..."
# Use numactl to lock the process to NUMA Node 0 (Socket 0)
# Warmup with a single prompt to ensure model weights are loaded into RAM cache before starting the grid search
numactl --cpunodebind=0 --membind=0 python main_custom.py \
    --test-case-path "test_cases/prompts_10.json" \
    --max-num-seqs 1 \
    --max-tokens-generated 1 \
    --scheduler-type "fcfs"
echo "Model cached. Starting grid search."

for TEST_CASE in "${TEST_CASES[@]}"; do
    for MAX_NUM_SEQS in "${MAX_NUMS_SEQS[@]}"; do
        for MAX_TOKENS in "${MAX_TOKENS_PER_SEQ[@]}"; do
            for SCHEDULER in "${SCHEDULER_TYPES[@]}"; do
                echo "=========================================="

                echo "Running test with ${TEST_CASE}, max_num_seqs=${MAX_NUM_SEQS}, max_tokens_per_seq=${MAX_TOKENS}, scheduler=${SCHEDULER}"
                # Use numactl to lock the process to NUMA Node 0 (Socket 0) for consistent performance metrics
                numactl --cpunodebind=0 --membind=0 python main_custom.py \
                    --test-case-path "${TEST_CASE}" \
                    --max-num-seqs "${MAX_NUM_SEQS}" \
                    --max-tokens-generated "${MAX_TOKENS}" \
                    --scheduler-type "${SCHEDULER}" \
                    --save-results \
                    --output-dir "${OUTPUT_DIR}"
                    
                sleep 5 
                echo "=========================================="
            done
        done
    done
done

