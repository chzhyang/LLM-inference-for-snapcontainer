from adapter import TransformersAdapter, OpenvinoAdapter
from pydantic import BaseModel
import logging as log
import sys
from typing import Optional

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                    level=log.INFO, stream=sys.stdout)

framework_to_adapter = {
    "transformers": TransformersAdapter,
    "openvino": OpenvinoAdapter,
}


class ReinitRequestModel(BaseModel):
    model_name: str = ""
    model_path: str = ""
    framework: str = ""
    model_dtype: str = ""
    device: str = ""
    n_threads: Optional[int] = None
    warmup: bool = True


class CompletionRequestModel(BaseModel):
    prompt: str
    max_length: int = 1024
    top_p: float = 0.7
    temperature: float = 0.95
    num_beams: int = 1
    do_sample: bool = False
    history: list = []


class ResponseModel(BaseModel):
    status: int
    message: str


class SupportListResponseModel(BaseModel):
    status: int
    support_list: dict


class CompletionResponseModel(BaseModel):
    status: int
    prompt: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_dur_s: float
    total_token_latency_s: float
    first_token_latency_ms: float
    next_token_latency_ms: float
    avg_token_latency_ms: float


support_list = {
    "transformers": {
        "llama2-7b": {
            "datatype": ["fp32", "bf16"],
        },
        "chatglm2-6b": {
            "datatype": ["fp32", "bf16", "int4"],
        }
    },
    "openvino": {
        "llama2-7b": {
            "datatype": ["bf16", "int8"],
        }
    },
}
