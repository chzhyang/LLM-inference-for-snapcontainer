# Chatglm2

In this directory, you will find examples on how you could start LLM server, apply IPEX on OPT models. For illustration purposes, we utilize the [facebook/OPT-1.3b](https://huggingface.co/facebook/opt-1.3b) as a reference OPT model.

## opt-1.3b

### 1. Prepare model

Download opt-1.3b to `$MODEL_REPOSITORY/opt-1.3b`, refer to [prepare model](../../../doc/prepare_model.md)

### 2. Start LLM server

Config and start LLM server on docker, load `opt-1.3b`

- Option 1: Run container diractely

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY/opt-1.3b:/models/opt-1.3b \
    -e OMP_NUM_THREADS=36 \
    -e MODEL_NAME=opt-1.3b \
    -e MODEL_PATH=/models/opt-1.3b \
    -e FRAMEWORK=ipex \
    -e MODEL_DTYPE=bf16 \
    -e SVC_PORT=8000 \
    -p 8000:8000 \
    ccr-registry.caas.intel.com/cnbench/snapcontainer-opt-1.3b-bf16-model:latest
    ```

- Option 2: Run container interactively and manually starting the server, use command flags to config LLM server

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY/opt-1.3b:/models/opt-1.3b \
    -p 8000:8000 \
    -it \
    ccr-registry.caas.intel.com/cnbench/snapcontainer-opt-1.3b-bf16-model:latest \
    /bin/bash
    ```

    Config and start LLM server in contianer

    ```shell
    conda init bash && exec bash -l
    conda activate llmenv
    OMP_NUM_THREADS=36 numactl -C 0-35 -m 0 python3 api.py \
    -m opt-1.3b \
    -d /models/opt-1.3b \
    -f ipex \
    -t bf16 \
    -p 8000
    ```

### Send prompt request to server

- Option 1: use `client.py`

    ```shell
    python3 client.py -t completion -p 'What is AI?'
    ```

- Option 2: use `curl`

    ```shell
    curl -X POST "http://127.0.0.1:8000/v2/completions" \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "What is AI?", "max_length":32, "history": []}'
    ```
