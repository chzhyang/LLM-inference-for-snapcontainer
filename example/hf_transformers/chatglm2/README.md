# Chatglm2

In this directory, you will find examples on how you could start LLM server, apply HF/transformers on CHatglm2 models. For illustration purposes, we utilize the [THUDM/Chatglm2-7b](https://huggingface.co/THUDM/chatglm2-6b) as a reference Chatglm2 model.

## Chatglm2-6b

### 1. Prepare model

Download chatglm2-6b to `$MODEL_REPOSITORY/chatglm2-6b`, refer to [prepare model](../../../doc/prepare_model.md)

### 2. Start LLM server

Config and start LLM server on docker, load `chatglm2-6b` by `huggingface/transformers`

- Option 1: Run container diractely

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY:/models \
    -e OMP_NUM_THREADS=36 \
    -e MODEL_NAME=chatglm2-6b \
    -e MODEL_PATH=/models/chatglm2-6b \
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
    -e MODEL_NAME=chatglm2-6b \
    -e MODEL_PATH=/models/chatglm2-6b \
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
    OMP_NUM_THREADS=36 numactl -C 0-35 -m 0 python3 api.py \
    -m chatglm2-6b \
    -d /models/chatglm2-6b \
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
