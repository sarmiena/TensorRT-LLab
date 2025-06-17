#!/bin/bash
docker run --gpus all \
	-it --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
	-v ./0.19.1-scripts:/scripts \
	-v ./meta-llama_Llama-3.1-8B-Instruct:/app/examples/llama/meta-llama_Llama-3.1-8B-Instruct \
	-v ./engine-0.19.1:/engine \
	--net=host \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864	\
	tensorrt_llm/release
