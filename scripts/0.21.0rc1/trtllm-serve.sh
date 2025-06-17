#!/bin/bash
export KV_CACHE_FREE_GPU_MEM_FRACTION=0.9 && \
export ENGINE_DIR=/engine && \
export TOKENIZER_DIR=/model

trtllm-serve ${ENGINE_DIR} --tokenizer ${TOKENIZER_DIR}
