# Llama2

In this directory, you will find examples on how you could start LLM inference server, apply OpenVINO on Llama2 models.
- Support model datatype: int8, INT8
- Intel/OpenVINO accelerates LLM inference on Xeon platform, such as IR model, oneDNN(AMX, AVX512, VNNI), etc.

## llama2-7b

### 1. Prepare model and image

Download model to '$MODEL_REPOSITORY/llama2-7b-ov-int8'(need 6GB+ disk)

docker pull chzhyang/llm-server:v2-snap-2023.12.21

### 2. Start LLM server

Config and start LLM server on docker, load `llama2-7b-ov-int8`(at least 15GB+ RAM)

- Option 1: Run container diractely

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY/llama2-7b-ov-int8:/models/llama2-7b-ov-int8 \
    -e OMP_NUM_THREADS=30 \
    -e MODEL_REPOSITORY=/models
    -e MODEL_NAME=llama2-7b \
    -e MODEL_PATH=/models/llama2-7b-ov-int8 \
    -e FRAMEWORK=openvino \
    -e MODEL_DTYPE=int8 \
    -e SVC_PORT=8000 \
    -p 8000:8000 \
    chzhyang/llm-server:v2-snap-2023.12.21
    ```

- Option 2: Run container interactively and manually starting the server, use enviroment to config LLM server

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY:/models \
     -e OMP_NUM_THREADS=30 \
    -e MODEL_REPOSITORY=/models \
    -e MODEL_NAME=llama2-7b \
    -e MODEL_PATH=/models/llama2-7b-ov-int8 \
    -e FRAMEWORK=openvino \
    -e MODEL_DTYPE=int8 \
    -e SVC_PORT=8000 \
    -p 8000:8000 \
    chzhyang/llm-server:v2-snap-2023.12.21 -it \
    /bin/bash
    ```

    Start LLM server in contianer

    ```shell
    conda init bash && exec bash -l
    conda activate llmenv
    OMP_NUM_THREADS=30 numactl -C 0-29 -m 0 python api.py
    ```

- Option 3: Run container interactively and manually starting the server, use command flags to config LLM server

    ```shell
    docker run --privileged \
    -v $MODEL_REPOSITORY:/models \
    -p 8000:8000 \
    -it \
    chzhyang/llm-server:v2-snap-2023.12.21 \
    /bin/bash
    ```

    Config and start LLM server in contianer

    ```shell
    conda init bash && exec bash -l
    conda activate llmenv
    OMP_NUM_THREADS=30 numactl -C 0-29 -m 0 python api.py \
    -m llama2-7b \
    -d /models/llama2-7b-ov-int8 \
    -f openvino \
    -t int8 \
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
