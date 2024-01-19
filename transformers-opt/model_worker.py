from utils import framework_to_adapter, log
import time

class ModelWorker():
    def __init__(self, framework, model_name, model_dtype, model_path, model_family, n_threads, device="CPU", warmup=True):
        self.framework = framework
        self.model_name = model_name
        self.model_dtype = model_dtype
        self.model_path = model_path
        self.model_family = model_family
        self.n_threads = n_threads
        self.device = device
        self.warmup = warmup
        if framework in framework_to_adapter:
            st = time.perf_counter()
            self.adapter = framework_to_adapter[framework](
                model_name=model_name,
                model_dtype=model_dtype,
                model_path=model_path,
                model_family=model_family,
                n_threads=n_threads,
                device=device,
                warmup=warmup
            )
            end = time.perf_counter()
            log.info(
                f'Model worker initialization costs {end-st:.2f} s')
            self.adapter.get_variables()
        else:
            raise ValueError(f"Unknown framework: {framework}")

    def get_variables(self):
        ret = "\n"
        for name, value in vars(self).items():
            if name == "adapter":
                value = type(self.adapter).__name__
            ret += f"{name}: {value}\n"
        return ret
