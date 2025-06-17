#!/bin/bash

rm -rf /ckpt
rm -rf /engine/config.json
rm -rf /engine/rank0.engine

python3 /app/tensorrt_llm/examples/quantization/quantize.py \
--model_dir /app/examples/llama/meta-llama_Llama-3.1-8B-Instruct/ \
--dtype bfloat16   \
--qformat fp8   \
--kv_cache_dtype fp8   \
--output_dir /ckpt \
--calib_size 512

trtllm-build --checkpoint_dir /ckpt \
	--output_dir /engine   \
	--remove_input_padding enable \
	--kv_cache_type paged   \
	--max_batch_size 2048   \
	--max_num_tokens 65536   \
	--max_seq_len 65536    \
	--use_paged_context_fmha enable   \
	--use_fp8_context_fmha enable   \
	--gemm_plugin disable   \
	--multiple_profiles enable
