import time
import torch
import logging as log
import sys

WARMUP_PROMPT = "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun"

log.basicConfig(format='[ %(levelname)s ] %(message)s',
                    level=log.INFO, stream=sys.stdout)

class BaseAdapter():
    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup):
        self.model_name = model_name
        self.model_dtype = model_dtype
        self.model_path = model_path
        self.model_family = model_family
        self.n_threads = n_threads
        self.device = device
        self.warmup = warmup

        if self.warmup is None:
            self.warmup = True


    def get_variables(self):
        log.info(f'\tdevice: {self.device}')
        log.info(f'\tmodel_name: {self.model_name}')
        log.info(f'\tmodel_dtype: {self.model_dtype}')
        log.info(f'\tmodel_path: {self.model_path}')
        log.info(f'\tmodel_family: {self.model_family}')
        log.info(f'\tn_threads: {self.n_threads}')
        log.info(f'\twarmup: {self.warmup}')

class TransformersAdapter(BaseAdapter):
    """
    Huggingface Transformers Adapter

    Support model: models for huggdingface transformers
    Support model datatype: fp32, bf16, int8, int4
    """

    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup, **kwargs):
        super().__init__(
            model_name = model_name,
            model_dtype = model_dtype,
            model_path = model_path,
            model_family = model_family,
            n_threads = n_threads,
            device = device,
            warmup = warmup,
        )
        if model_family == 'llama':
            from transformers import LlamaTokenizer, AutoModelForCausalLM
            log.info(" Loading tokenizer")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
            log.info(" Loading model")
            if model_dtype == "fp32":
                # self.model = AutoModelForCausalLM.from_pretrained(
                #     model_path,
                #     device_map="cpu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path)
            elif model_dtype == "bf16":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16)
        else:
            from transformers import AutoTokenizer
            log.info("Loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)
            from transformers import AutoModel, AutoModelForCausalLM
            log.info("Loading model")
            if model_dtype == "fp32":
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True).float()
            elif model_dtype == "bf16":
                self.model = AutoModel.from_pretrained(
                    model_path, device_map="cpu",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True)
            elif model_dtype == "int4":
                self.model = AutoModel.from_pretrained(model_path,
                                                       trust_remote_code=True).float()
        if self.model:
            self.model = self.model.eval()
        if self.warmup:
                self._warmup(WARMUP_PROMPT)

    def _warmup(self, prompt):
        with torch.inference_mode():
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            _ = self.model.generate(input_ids, max_new_tokens=50)

    def create_completion(
            self,
            prompt,
            max_new_tokens,
            top_p, temperature,
            num_beams,
            do_sample,
            stream=False
    ):
        if not stream:
            with torch.inference_mode():
                perf = {"latency": []}
                log.info(f'Start generating')
                st = time.time()
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    perf=perf
                )
                prompt_tokens = input_ids.shape[1]
                completion_ids = output_ids[0].tolist()[prompt_tokens:]
                completion = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True)
                end = time.time()

                latency = perf["latency"]
                resp = {
                    "completion": completion,
                    "prompt_tokens": prompt_tokens,
                    "total_dur_s": end-st,  # total time, include tokeninzer.encode+decode, tokens generation
                    "completion_tokens": len(completion_ids),
                    # total tokens completion latency, except tokenizer.decode time
                    "total_token_latency_s": sum(latency),
                    # first token completion latency
                    "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
                    # next token completion latency
                    "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
                    # average token completion latency
                    "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
                }
                return resp
        else:
            # TODO stream
            pass

import openvino as ov
from openvino.runtime import Core, Tensor
from typing import Optional
import numpy as np

class OpenvinoAdapter(BaseAdapter):
    def __init__(self, model_name, model_dtype, model_path, model_family, n_threads, device, warmup, **kwargs) -> None:
        super().__init__(
            model_name = model_name,
            model_dtype = model_dtype,
            model_path = model_path,
            model_family = model_family,
            n_threads = n_threads,
            device = device,
            warmup = warmup,
        )
        log.info("Loading tokenizer")
        if model_family == 'llama':
            from transformers import LlamaTokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_path)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True)

        if model_family == 'llama' and model_dtype in ['bf16', 'int8']:
            log.info("Loading model")
            from pathlib import Path
            ir_model = Path(model_path) / "openvino_model.xml"
            core = Core()
            self.model = core.read_model(ir_model)
            self.input_names = {
                key.get_any_name(): idx
                for idx, key in enumerate(self.model.inputs)
            }
            self.output_names = {
                key.get_any_name(): idx
                for idx, key in enumerate(self.model.outputs)
            }
            self.key_value_input_names = [
                key for key in self.input_names if "key_values" in key
            ]
            self.key_value_output_names = [
                key for key in self.output_names if "present" in key
            ]
            ov_config = {
                ov.properties.cache_dir(): "",
                'PERFORMANCE_HINT': 'LATENCY',
            }
            
            # n_threads must be less than or equal to the number of cpu affinity
            import os
            num_cpus = len(os.sched_getaffinity(0))
            if self.n_threads is not None and self.n_threads > num_cpus:
                self.n_threads = num_cpus
                ov_config[ov.properties.inference_num_threads()] = self.n_threads
                
            compiled_model = core.compile_model(
                model=self.model, 
                device_name=device,
                config=ov_config,
            )

            self.request = compiled_model.create_infer_request()
            if self.warmup:
                self._warmup(WARMUP_PROMPT)
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        summation = e_x.sum(axis=-1, keepdims=True)
        return e_x / summation

    def process_logits(self, cur_length, scores, eos_token_id, min_length=0):
        if cur_length < min_length:
            scores[:, eos_token_id] = -float("inf")
        return scores

    def get_top_k_logits(self, scores, top_k):
        filter_value = -float("inf")
        top_k = min(max(top_k, 1), scores.shape[-1])
        top_k_scores = -np.sort(-scores)[:, :top_k]
        indices_to_remove = scores < np.min(top_k_scores)
        filtred_scores = np.ma.array(scores,
                                    mask=indices_to_remove,
                                    fill_value=filter_value).filled()
        return filtred_scores

    def generate_sequence(self, sampling, input_ids, attention_mask, eos_token_id,
                        max_sequence_length, perf: Optional[dict]=None):
        if perf is None:
            perf = {"latency":[]}
        latency = perf["latency"]
        st = time.perf_counter()

        past_key_values = None
        prompt_len = len(input_ids[0])

        while True:
            try:
                inputs = {}
                if past_key_values is not None:
                    inputs = dict(zip(self.key_value_input_names, past_key_values))
                    inputs["input_ids"] = input_ids[:, -1:]
                    cur_input_len = 1
                else:
                    inputs["input_ids"] = input_ids
                    shape_input_ids = input_ids.shape
                    num_attention_heads = 1
                    for input_name in self.key_value_input_names:
                        model_inputs = self.model.input(input_name)
                        shape = model_inputs.get_partial_shape()
                        shape[0] = shape_input_ids[0] * num_attention_heads
                        if shape[2].is_dynamic:
                            shape[2] = 0
                        if shape[1].is_dynamic:
                            shape[1] = 0
                        inputs[input_name] = Tensor(model_inputs.get_element_type(),
                                                    shape.get_shape())
                cur_input_len = len(inputs["input_ids"][0])
                if "attention_mask" in self.input_names and attention_mask is not None:
                    inputs["attention_mask"] = attention_mask
                self.request.start_async(inputs, share_inputs=True)
                self.request.wait()
                logits = self.request.get_tensor("logits").data
                past_key_values = tuple(
                    self.request.get_tensor(key).data for key in self.key_value_output_names)
                next_token_logits = logits[:, cur_input_len - 1, :]
                # pre-process distribution
                next_token_scores = self.process_logits(len(input_ids[0]),
                                                next_token_logits, eos_token_id)
                top_k = 20
                next_token_scores = self.get_top_k_logits(next_token_scores, top_k)
                # get next token id
                if sampling:
                    probs = self.softmax(next_token_scores)
                    next_tokens = np.random.choice(probs.shape[-1],
                                                1,
                                                p=probs[0],
                                                replace=True)
                else:
                    next_tokens = np.argmax(next_token_scores, axis=-1) # greedy search
                # break the loop if max length or end of text token is reached
                if (len(input_ids[0]) - prompt_len
                        ) == max_sequence_length or next_tokens == eos_token_id:
                    # end = time.perf_counter()
                    # latency.append(end - st)
                    break
                else:
                    input_ids = np.concatenate((input_ids, [next_tokens]), axis=-1)
                    attention_mask = np.concatenate(
                        (attention_mask, [[1] * len(next_tokens)]), axis=-1)

            finally:
                end = time.perf_counter()
                latency.append(end - st)
                st = end

        return input_ids
    
    def _warmup(self, prompt):
        log.info(f'Warm up')
        inputs = self.tokenizer(prompt, return_tensors="np")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        _ = self.generate_sequence(
            sampling=False,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_sequence_length=50,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def create_completion(self, prompt, max_new_tokens, top_p, temperature, num_beams, do_sample, stream=False):
        if not stream:
            perf = {"latency": [], }
            log.info(f'Start generating')
            st = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="np")
            input_ids = inputs["input_ids"]
            attention_mask = attention_mask = inputs["attention_mask"]
            output_ids = self.generate_sequence(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_sequence_length=max_new_tokens if max_new_tokens else 2048,
                # top_p=top_p if top_p else 0.7,
                # temperature=temperature if temperature else 0.95,
                eos_token_id=self.tokenizer.eos_token_id,
                sampling=do_sample,
                perf=perf)
            prompt_tokens = input_ids.shape[1]
            completion_ids = output_ids[0].tolist()[prompt_tokens:]
            completion = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True)
            end = time.perf_counter()

            latency = perf["latency"]
            resp = {
                "completion": completion,
                "prompt_tokens": prompt_tokens,
                "total_dur_s": end-st,  # total time, include tokeninzer.encode+decode, tokens generation
                "completion_tokens": len(completion_ids),
                # total tokens completion latency, except tokenizer.decode time
                "total_token_latency_s": sum(latency),
                # first token completion latency
                "first_token_latency_ms": latency[0]*1000 if len(latency) > 0 else 0,
                # next token completion latency
                "next_token_latency_ms": sum(latency[1:])*1000 / len(latency[1:]) if len(latency) > 1 else 0,
                # average token completion latency
                "avg_token_latency_ms": sum(latency)*1000 / len(latency) if len(latency) > 0 else 0,
            }
            return resp
        else:
            # TODO stream
            pass
        return
    
