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
        from transformers import AutoTokenizer, OPTForCausalLM
        # import intel_extension_for_pytorch as ipex
        log.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        log.info("Loading model")
        if model_family=='opt':
            self.model = OPTForCausalLM.from_pretrained(model_path).eval()
            # self.model = ipex.optimize_transformers(self.model)
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
    ):
        log.info(f'Start generating')
        st = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids=inputs['input_ids']
        output_ids = self.model.generate(
            inputs=input_ids,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            num_beams=1,
            do_sample=False,
            attention_mask = inputs["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        prompt_tokens = input_ids.shape[1]
        completion_ids = output_ids[0].tolist()[prompt_tokens:]
        completion = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True)
        end = time.time()

        resp = {
            "completion": completion,
            "prompt_tokens": prompt_tokens,
            "total_dur_s": end-st,
            "completion_tokens": len(completion_ids),
        }
        return resp