from adapter import TransformersAdapter
from pydantic import BaseModel, ConfigDict
import logging as log
import sys
from typing import Optional

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                    level=log.INFO, stream=sys.stdout)

framework_to_adapter = {
    "transformers": TransformersAdapter,
}

class ReinitRequestModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
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

support_list = {
    "transformers": {
        "opt-1.3b": {
            "datatype": ["fp16"],
        }
    },
}
