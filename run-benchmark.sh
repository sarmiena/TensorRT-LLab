#!/bin/bash
docker run --network host \
	-e HF_TOKEN=$HF_TOKEN \
	inference_benchmarker_latest inference-benchmarker --no-console   \
		--url http://localhost:8000/v1   \
		--max-vus 800 \
		--duration 120s \
		--warmup 15s \
		--benchmark-kind rate \
		--rates 100   \
		--prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
		--decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
		--model-name "meta-llama/Llama-3.1-8B-Instruct" \
		--tokenizer-name "meta-llama/Llama-3.1-8B-Instruct"
