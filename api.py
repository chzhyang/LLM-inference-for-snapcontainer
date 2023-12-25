from fastapi import FastAPI, HTTPException, Request
import uvicorn
import time
import argparse
import os
from model_worker import ModelWorker
from utils import CompletionRequestModel, CompletionResponseModel, SupportListResponseModel, ReinitRequestModel, ResponseModel, log
import warnings
# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

DEFAULT_SVC_PORT = 8000

app = FastAPI()


@app.get("/v2/support_list", response_model=SupportListResponseModel, status_code=200)
async def support_list():
    resp = {
        "support_list": get_support_list(),
        "status": 200,
    }
    return resp

@app.post("/v2/completions", response_model=CompletionResponseModel, status_code=200)
async def completion(request: Request, request_data: CompletionRequestModel):
    global worker

    prompt = request_data.prompt
    max_length = request_data.max_length
    top_p = request_data.top_p
    temperature = request_data.temperature
    num_beams = request_data.num_beams
    do_sample = request_data.do_sample

    if not prompt:
        raise HTTPException(status_code=400, detail='Need a prompt')
    log.info(f'Get a prompt: {prompt}')

    st = time.perf_counter()
    result = worker.adapter.create_completion(prompt=prompt,
                                              max_new_tokens=max_length if max_length is not None else 1024,
                                              num_beams=num_beams if num_beams is not None else 1,
                                              do_sample=do_sample if do_sample is not None else False,
                                              top_p=top_p if top_p is not None else 0.7,
                                              temperature=temperature if temperature is not None else 0.95,
                                              stream=False)
    end = time.perf_counter()

    response_data = {
        "status": 200,
        "prompt": prompt,
        "completion": result["completion"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_dur_s": end - st,
        "total_token_latency_s": result["total_token_latency_s"],
        "first_token_latency_ms": result["first_token_latency_ms"],
        "next_token_latency_ms": result["next_token_latency_ms"],
        "avg_token_latency_ms": result["avg_token_latency_ms"]
    }

    return response_data


def get_support_list():
    """
    Returns a dict of supported AI lib, LLM and datatype
    """
    from utils import support_list
    return support_list


def get_model_family(model_name):
    if "llama" in model_name:
        model_family = "llama"
    elif "chatglm" in model_name:
        model_family = "chatglm"
    else:
        raise ValueError(
            f'Unsupported model {model_name}')
    return model_family

def check_in_support_list(framework, model_name, model_dtype):
    support_list = get_support_list()
    if framework not in support_list:
        raise ValueError(f'{framework} is not supported')
    if model_name not in support_list[framework]:
        raise ValueError(
            f'{model_name} is not supported on {framework}')
    if model_dtype not in support_list[framework][model_name]["datatype"]:
        raise ValueError(
            f'{model_dtype} is not supported for {model_name} on {framework}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM inference service')
    parser.add_argument(
        '--device', type=str, choices=["CPU", "GPU"], default="CPU", help='device for inference')
    parser.add_argument('-m', '--model-name', type=str,
                        choices=["llama2-7b", "chatglm2-6b"], help='LLM to load')
    parser.add_argument('-o', '--model-online',
                        action="store_true", help='Load model online')
    parser.add_argument('-d', '--model-path', type=str,
                        default="", help='Local model path')
    parser.add_argument('-p', '--port', type=int, help='Service port')
    parser.add_argument('-e', '--metric_port', type=int,
                        default=8090, help='Metric port')
    parser.add_argument('-f', '--framework', type=str,
                        choices=["transformers", "bigdl-llm-cpp",
                                 "bigdl-llm-transformers", "openvino"],
                        help='Inference framework')
    parser.add_argument('-t', '--model-dtype', type=str,
                        choices=["fp32", "bf16", "int8", "int4"], help='Model data type')
    parser.add_argument('-i', '--ipex', action="store_true",
                        help='Use IPEX(intel-extension-for-pytorch)')
    parser.add_argument('-n', '--n-threads', type=int,
                        help='number of thread')
    parser.add_argument('-w', '--warmup', action="store_true", default=True,help='warmup model')
    args = parser.parse_args()

    # get env
    device = os.environ.get('LLM_SERVER_DEVICE')
    model_name = os.environ.get('MODEL_NAME')
    model_path = os.environ.get('MODEL_PATH')
    model_dtype = os.environ.get('MODEL_DTYPE')
    framework = os.environ.get('FRAMEWORK')
    ipex = os.environ.get('IPEX')
    warmup = os.environ.get('LLM_SERVER_WARMUP')

    if device is not None:
        device = device.lower()
    if ipex is not None:
        ipex = ipex.lower() == "true"
    else:
        ipex = False
    model_online = os.environ.get('MODEL_ONLINE')
    if model_online is not None:
        model_online = model_online.lower() == "true"
    else:
        model_online = False
    n_threads = os.environ.get('N_THREADS')
    if n_threads is not None:
        n_threads = int(n_threads)
    svc_port = os.environ.get('SVC_PORT')
    if svc_port is not None:
        svc_port = int(svc_port)
    metric_port = os.environ.get('METRIC_PORT')
    if metric_port is not None:
        metric_port = int(metric_port)
    if warmup is not None:
        warmup = warmup.lower() == "true"
    else:
        warmup = False

    # get args
    # priority: command line args > env variables
    if device is None or args.device != "CPU":
        device = args.device
    if args.model_name:
        model_name = args.model_name
    if args.model_path:
        model_path = args.model_path
    if args.model_dtype:
        model_dtype = args.model_dtype
    if args.framework:
        framework = args.framework
    if args.ipex:
        ipex = args.ipex
    if args.model_online:
        model_online = args.model_online
    if args.n_threads:
        n_threads = args.n_threads
    if args.port:
        svc_port = args.port
    if args.metric_port:
        metric_port = args.metric_port
    if args.warmup:
        warmup = args.warmup

    if framework is None:
        raise ValueError(f'framework is required')
    if model_name is None:
        raise ValueError(f'model_name is required')
    if model_dtype is None:
        raise ValueError(f'model_dtype is required')
    if model_path is None:
        raise ValueError(f'model_path is required')
    if svc_port is None:
        svc_port = DEFAULT_SVC_PORT

    check_in_support_list(framework, model_name, model_dtype)

    model_family = get_model_family(model_name)

    log.info(f'Initializing model worker')
    st = time.perf_counter()
    worker = ModelWorker(
        framework=framework,
        model_name=model_name,
        model_dtype=model_dtype,
        model_path=model_path,
        model_family=model_family,
        n_threads=n_threads,
        device=device,
        warmup=warmup
    )
    end = time.perf_counter()

    uvicorn.run(app, host='0.0.0.0', port=svc_port, workers=1)
