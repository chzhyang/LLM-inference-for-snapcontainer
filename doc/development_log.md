# Develop records

## Otc 16, 2023

LLM server:v2

- Abstract supported AI libs, models, datatypes into class to improve scalability
  - ModelWorker class
  - Adapter class
    - TransformersAdapter
    - BigdlLLMTransformersAdapter
    - BigdlLLMCPPAdapter
- Integrate bigdl-llm/cpp into LLM server
  - support llama2-7b int4
  - chatglm2-6b still waiting for bigdl-llm ready
- Develop a reintit service to support hot update model, AI lib or data type
- Develop to collect performance data of Transformers, BigdlLLMTransformers, BigdlLLMCPP
  - Transformers and BigdlLLMTransformers should modify transformers code in miniconda3/envs/chatglm/lib/python3.9/site-packages/transformers/generation/utils.py, utils.py should be instead of ./extension/transformers/ubtils.py
  - BigdlLLMCPP should modify bigdl-llm code in miniconda3/envs/chatglm/lib/python3.9/site-packages/bigdl/llm/ggml/model/llama, llama.py and llama_types.py should be instead of ./extension/bigdl-llm-cpp-extension

## Otc 18, 2023

- Update doc and add examples for different AI lib/model/datatype

## Otc 20, 2023

- Add dockerfile for LLM server v2, use conda to manage python libs

## Otc 25, 2023

- Update dockerfile to Dockerfile.v2-slim using conda env volume to compress image, build new image `chzhyang/llm-server:v2-slim`, image size is reduced from 6GB to 500MB , refer to [here](../docker/README.md#docker-image)