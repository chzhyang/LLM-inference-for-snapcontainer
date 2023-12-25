# Docker Image

| Image       | Size|
| ----------- | --- |
| chzhyang/llm-server:v2-slim | 547MB |
| chzhyang/llm-server:v2 | 6.37GB |
| chzhyang/llm-server:v1 | 5.56GB |
|  |  |  |

To compress image, we extract python lib to conda env, build `chzhyang/llm-server:v2-slim` by `Dockerfile:v2-slim`, image size is significantly reduced.

How to use `chzhyang/llm-server:v2-slim` with conda?

Option 1. Simply download conda and env we've already prepared

Download conda from [here](TODO) to `~/miniconda3-docker`, then mount `~/miniconda3-docker` to `/opt/miniconda3` in container, refer to [here](#run-container-and-start-llm-server)

Option 2. Create conda env locally

[Install conda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) to `~/miniconda3-docker`

```shell
conda create -n llmenv python=3.9
conda activate llmenv
pip install -r llm-server/requirements-all.txt
```

Mount `~/miniconda3` to `/opt/miniconda3` in container, refer to [here](#run-container-and-start-llm-server)

## Usage

### Prepate image

Option1. Use image in docker hub

```shell
docker pull chzhyang/llm-server:v2-slim
```

Option 2. Build image locally

```shell
docker build -f docker/Dockerfile.v2-slim -t chzhyang/llm-server:v2-slim . 
```

### Prepare model

Refer to [here](prepare_model.md)

### Run container and start LLM server

Option 1. Run container diractely, use enviroment to config LLM server

```shell
docker run --privileged --rm \
-v /home/sdp/miniconda3-docker:/opt/miniconda3 \
-v $MODEL_REPOSITORY:/models \
-e OMP_NUM_THREADS=36 \
-e CPP_THREADS=36 \
-e MODEL_NAME=llama2-7b \
-e MODEL_PATH=/models/bigdl_llm_llama_q4_0.bin \
-e FRAMEWORK=bigdl-llm-cpp \
-e MODEL_DTYPE=int4 \
-e SVC_PORT=8000 \
-p 8000:8000 \
chzhyang/llm-server:v2-slim
```

Option 2. Run container interactively and manually starting the server

Run container

```shell
docker run --privileged --rm \
-v /home/sdp/miniconda3-docker:/opt/miniconda3 \
-v $MODEL_REPOSITORY:/models \
-p 8000:8000 -it  \
chzhyang/llm-server:v2-slim /bin/bash
```

Config and start LLM server in contianer

```shell
conda init bash && exec bash -l
conda activate llmenv
OMP_NUM_THREADS=36 numactl -C 0-35 -m 0 python api.py \
--model-name llama2-7b \
--model-path /models/bigdl_llm_llama_q4_0.bin \
--framework bigdl-llm-cpp \
--model-dtype int4 \
--cpp-threads 36 \
--port 8000
```

### Send request to server

- Option 1: use `client.py`

    ```shell
    python3 llm-server/client.py -t completion -p 'What is AI?'
    ```

- Option 2: use `curl`

    ```shell
    curl -X POST "http://127.0.0.1:8000/v2/completions" \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "What is AI?", "max_length":32, "history": []}'
    ```

More api refer to [here](../README.md#http-api)