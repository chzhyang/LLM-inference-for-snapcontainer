FROM chzhyang/llm-server-env:latest

ENV PATH /opt/miniconda3/bin:$PATH

WORKDIR /app
COPY llama2-7b-ov-int8 /models/llama2-7b-ov-int8
COPY api.py adapter.py model_worker.py utils.py ./

ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=1
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV OMP_NUM_THREADS=30

CMD ["conda", "run", "--no-capture-output", "-n", "llmenv", "python", "api.py"]
