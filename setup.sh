export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')

# use uv
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl --torch-backend cpu

CURRENT_USER=$(whoami)

find /home/${CURRENT_USER} -iname *libtcmalloc_minimal.so.4
find /home/${CURRENT_USER} -iname *libiomp5.so

# Automatically find the path of libtcmalloc_minimal.so.4 and libiomp5.so and set the environment variables
# Only set if the path includes '/home/${CURRENT_USER}/{current_repo_name}/.venv/lib'
TC_PATH=$(find /home/${CURRENT_USER} -iname *libtcmalloc_minimal.so.4 | grep "/home/${CURRENT_USER}/$(basename $(pwd))/.venv/lib" || true)
IOMP_PATH=$(find /home/${CURRENT_USER} -iname *libiomp5.so | grep "/home/${CURRENT_USER}/$(basename $(pwd))/.venv/lib" || true)

# Add to LD_PRELOAD if TC_PATH and IOMP_PATH are found
if [ -n "$TC_PATH" ] && [ -n "$IOMP_PATH" ]; then
    export LD_PRELOAD="$TC_PATH:$IOMP_PATH:$LD_PRELOAD"
    echo "Set LD_PRELOAD to: $LD_PRELOAD"
else
    echo "Could not find both libtcmalloc_minimal.so.4 and libiomp5.so in the expected paths."
fi

# Install other dependencies
uv pip install dotenv
uv pip install pandas