# Llama2

In this directory, you will find examples on how you could start LLM server, apply HF/transformers on Llama2 models. For illustration purposes, we utilize the [Meta/Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf) as a reference Llama2 model.

## llama2-7b

### 1. Prepare model

Download llama2-7b to `$MODEL_REPOSITORY/Llama-2-7b-hf`, refer to [prepare model](../../../doc/prepare_model.md)

### 2. Start LLM server

Config and start LLM server on docker, load `Llama2-7b(FP16)` by `huggingface/transformers`

- Option 1: Run container diractely

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY:/models \
    -e OMP_NUM_THREADS=36 \
    -e MODEL_REPOSITORY=/models
    -e MODEL_NAME=llama2-7b \
    -e MODEL_PATH=/models/Llama-2-7b-hf \
    -e FRAMEWORK=transformers \
    -e MODEL_DTYPE=fp16 \
    -e SVC_PORT=8000 \
    -p 8000:8000 \
    chzhyang/llm-server:v2
    ```

- Option 2: Run container interactively and manually starting the server, use enviroment to config LLM server

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY:/models \
    -e OMP_NUM_THREADS=36 \
    -e MODEL_REPOSITORY=/models
    -e MODEL_NAME=llama2-7b \
    -e MODEL_PATH=/models/Llama-2-7b-hf \
    -e FRAMEWORK=transformers \
    -e MODEL_DTYPE=fp16 \
    -e SVC_PORT=8000 \
    -p 8000:8000 \
    -it \
    chzhyang/llm-server:v2 \
    /bin/bash
    ```

    Start LLM server in contianer

    ```shell
    conda init bash && exec bash -l
    conda activate llmenv
    OMP_NUM_THREADS=36 numactl -C 0-35 -m 0 python api.py
    ```

- Option 3: Run container interactively and manually starting the server, use command flags to config LLM server

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY:/models \
    -p 8000:8000 \
    -it \
    chzhyang/llm-server:v2 \
    /bin/bash
    ```

    Config and start LLM server in contianer

    ```shell
    conda init bash && exec bash -l
    conda activate llmenv
    OMP_NUM_THREADS=36 numactl -C 0-35 -m 0 python api.py \
    -m llama2-7b \
    -d /models/Llama-2-7b-hf \
    -f transformers \
    -t fp16 \
    -p 8000
    ```

### 3. Send prompt request to server

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

