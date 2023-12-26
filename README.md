# LLM inference server

Large language models(LLM) inference server on Intel/Xeon platform.

- Support popular LLMs(Llama2, Chatglm2, etc.)
- Support popular AI inference frameworks/libraries(Transformers, PyTorch, Intel/IPEX, Intel/Openvino, Intel/BigDl-LLM, Intel/xFastertransformer, etc.)
- Support multiple datatype(FP32, FP16, BF16, INT8, INT4)
- Provide web service based on fastapi(HTTP)

| AI library  | Model       | Data Type  | Example |
| ----------- | -------------------- | ------ | ---- |
| Intel/bigdl-llm-cpp  |  Llama2  | INT4 | [Llama2-7b](./example/bigdl_llm_cpp/README.md)
| Intel/bigdl-llm-cpp-transformers| Llama2, Chatglm2  | INT8, INT4  | [Llama2-7b](./example/bigdl_llm_transformers/llama2/README.md), [Chatglm2-6b](./example/bigdl_llm_transformers/chatglm2/README.md)
| Huggingface/transformers-pytorch | Chatglm2 | FP16  | [Llama2-7b](./example/hf_transformers/llama2/README.md), [Chatglm2-6b](./example/hf_transformers/chatglm2/README.md) |
| Intel/Openvino  |     Llama2     |    BF16, INT8   |   [Llama2-7b](./example/openvino/README.md)     |
|             |                      |               |        |

## Docker image

| Image       | Size| Description | 
| ----------- | --- | --- |
| ccr-registry.caas.intel.com/cnbench/llm-server:v2-snap-model-2023.12.26 | 14.3GB | contains llama2-7b-ov-int8, for snapcontainer |
| ccr-registry.caas.intel.com/cnbench/llm-server:v2-snap-2023.12.21 | 3.36GB | use shared model, for snapcontainer |
| chzhyang/llm-server:v2-slim | 547MB | Slimed image using shared packages |
| chzhyang/llm-server:v2 | 6.37GB | |
| chzhyang/llm-server:v1 | 5.56GB | |
|  |  |  | |


## HTTP API

We provide a [client script](client.py) to easily access LLM server, you can access `localhost:8000/docs` or `localhost:8000/redoc` to get API in OpenAPI format after LLM server started, or simply refer to [api.json](./doc/api.json)


## Example

We provide detailed examples for different AI libraries, models and data types, see [here](./example/README.md)

## System support

Hardware:

- Intel® Xeon® processors

