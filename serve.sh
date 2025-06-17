https://github.com/NVIDIA/TensorRT-LLM/issues/4275#issuecomment-2941155249


Since the requirements.txt is updated and torch>=2.7.0a0,<=2.7.0 now includes the Blackwell stable version 2.7.0, too, I'll close this issue.

edit: still need to figure out how to build TensorRT-LLM version 0.19.0 successfully.. dependency problems everywhere.

There is no need to build it yourself actually.
Just use the new nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3 image because according to the Release notes this includes TensorRT-LLM version release/0.19.0:
# Docker base image used:
export HF_TOKEN=ENTER_YOUR_HUGGINGFACE_TOKEN
sudo docker run --gpus all -it --rm \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN=$HF_TOKEN --net host nvcr.io/nvidia/tritonserver:25.05-trtllm-python-py3
Check TensorRT-LLM version within the docker:

# python3 -c "import tensorrt_llm.version; print(tensorrt_llm.version.__version__)"
[TensorRT-LLM] TensorRT-LLM version: 0.19.0
0.19.0
Now test it by either running the example from https://nvidia.github.io/TensorRT-LLM/0.18.2/installation/linux.html#installing-on-linux or build a TensorRT engine yourself. Here is an example of a high throughput Engine for my RTX 5090, even including FP8 K/V cache:

# my examples to build and run tensorrt-llm:
cd /app/examples/llama && \
nano download.py

# BEGIN download.py
import os
from huggingface_hub import snapshot_download
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN environment variable not set")
repo_id = "meta-llama/Llama-3.1-8B-Instruct"
local_dir = "meta-llama_Llama-3.1-8B-Instruct"
download_root = snapshot_download(
	repo_id=repo_id,
	local_dir=local_dir,
	token=hf_token,
	ignore_patterns=["*/*"]
)
# END download.py

cd /app/examples/llama && \
python3 download.py && \
cd /app/examples/llama && \
python3 /app/examples/quantization/quantize.py \
  --model_dir /app/examples/llama/meta-llama_Llama-3.1-8B-Instruct/ \
  --dtype bfloat16 \
  --qformat fp8 \
  --kv_cache_dtype fp8 \
  --output_dir /ckpt \
  --calib_size 512

# note: Note: FP8 gemm plugin is an experimental feature aimed to improve performance in small-batch-size cases(e.g. BS<=4). Although inputs with batch size larger than 4 can be correctly inferenced, the performance may decrease as batch size grows.
# https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/models/core/llama#fp8-post-training-quantization
# I do not use --gemm_plugin auto or --gemm_plugin fp8 because https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html suggests "On FP8, it’s recommended to be disabled."
trtllm-build --checkpoint_dir /ckpt --output_dir /engine \
  --remove_input_padding enable \
  --kv_cache_type paged \
  --max_batch_size 2048 \
  --max_input_len 1024 \
  --max_num_tokens 1024 \
  --max_seq_len 2048 \
  --use_paged_context_fmha enable \
  --use_fp8_context_fmha enable \
  --gemm_plugin disable \
  --multiple_profiles enable && \
mkdir /triton_model_repo && cp -r /app/all_models/inflight_batcher_llm/* /triton_model_repo/

# if you don't have enough GPU mem and get OOM, use a lower value than 0.9, e.g. 0.5 or 0.2, but on my RTX 5090 it worked:
export KV_CACHE_FREE_GPU_MEM_FRACTION=0.9 && \
export ENGINE_DIR=/engine && \
export TOKENIZER_DIR=/app/examples/llama/meta-llama_Llama-3.1-8B-Instruct/ && \
export MODEL_FOLDER=/triton_model_repo && \
export TRITON_MAX_BATCH_SIZE=1 && \
export INSTANCE_COUNT=1 && \
export MAX_QUEUE_DELAY_MS=0 && \
export MAX_QUEUE_SIZE=0 && \
export FILL_TEMPLATE_SCRIPT=/app/tools/fill_template.py && \
export DECOUPLED_MODE=false && \
export LOGITS_DATATYPE=TYPE_FP32

python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},logits_datatype:${LOGITS_DATATYPE} && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT} && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:inflight_fused_batching,max_queue_size:${MAX_QUEUE_SIZE},encoder_input_features_data_type:TYPE_FP16,logits_datatype:${LOGITS_DATATYPE},kv_cache_free_gpu_mem_fraction:${KV_CACHE_FREE_GPU_MEM_FRACTION},exclude_input_in_output:True && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT} && \
python3 ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${TRITON_MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},logits_datatype:${LOGITS_DATATYPE}

# now you can either run tritonserver which does NOT run an openAI compatible API...
cd /opt/tritonserver/python/openai && \
python3 openai_frontend/main.py --model-repository $MODEL_FOLDER --tokenizer $TOKENIZER_DIR

# ...or you better use trtllm-serve if you need an openAI compatible API:
trtllm-serve ${ENGINE_DIR} --tokenizer ${TOKENIZER_DIR}
Output:

[TensorRT-LLM][INFO] Memory usage when calculating max tokens in paged kv cache: total: 31.84 GiB, available: 17.83 GiB
[TensorRT-LLM][INFO] Number of blocks in KV cache primary pool: 8219
[TensorRT-LLM][INFO] Number of blocks in KV cache secondary pool: 0, onboard blocks to primary memory before reuse: true
[TensorRT-LLM][INFO] Number of tokens per block: 32.
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 16.05 GiB for max tokens in paged KV cache (263008)

Results with Huggingface inference-benchmarker and 200 token prompts: 7435.27 tokens/sec throughput at high QPS. 👍

# Run:
sudo docker run --network host -e HF_TOKEN=$HF_TOKEN \
  inference_benchmarker_latest inference-benchmarker --no-console \
  --url http://localhost:8000/v1 \
  --max-vus 800 --duration 120s --warmup 30s --benchmark-kind rate \
  --rates 1.0 --rates 10.0 --rates 30.0 --rates 100.0 \
  --prompt-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --decode-options "num_tokens=200,max_tokens=220,min_tokens=180,variance=10" \
  --model-name "meta-llama/Llama-3.1-8B-Instruct" \
  --tokenizer-name "meta-llama/Llama-3.1-8B-Instruct"
# Output:

┌─────────────────┬────────────────────────────────────────────────────────────────┐
│ Parameter       │ Value                                                          │
├─────────────────┼────────────────────────────────────────────────────────────────┤
│ Max VUs         │ 800                                                            │
│ Duration        │ 120                                                            │
│ Warmup Duration │ 30                                                             │
│ Benchmark Kind  │ Rate                                                           │
│ Rates           │ [1.0, 10.0, 30.0, 100.0]                                       │
│ Num Rates       │ 10                                                             │
│ Prompt Options  │ num_tokens=Some(200),min_tokens=180,max_tokens=220,variance=10 │
│ Decode Options  │ num_tokens=Some(200),min_tokens=180,max_tokens=220,variance=10 │
│ Tokenizer       │ meta-llama/Llama-3.1-8B-Instruct                               │
│ Extra Metadata  │ N/A                                                            │
└─────────────────┴────────────────────────────────────────────────────────────────┘
┌──────────────────────┬─────────────┬───────────────────┬─────────────┬───────────┬────────────────────┬────────────┬─────────────────────┬─────────────────────────────┬──────────────────────────────┐
│ Benchmark            │ QPS         │ E2E Latency (avg) │ TTFT (avg)  │ ITL (avg) │ Throughput         │ Error Rate │ Successful Requests │ Prompt tokens per req (avg) │ Decoded tokens per req (avg) │
├──────────────────────┼─────────────┼───────────────────┼─────────────┼───────────┼────────────────────┼────────────┼─────────────────────┼─────────────────────────────┼──────────────────────────────┤
│ warmup               │ 0.74 req/s  │ 1.35 sec          │ 22.55 ms    │ 7.70 ms   │ 128.74 tokens/sec  │ 0.00%      │ 23/23               │ 200.00                      │ 173.52                       │
│ constant@1.00req/s   │ 0.99 req/s  │ 1.39 sec          │ 19.75 ms    │ 7.66 ms   │ 178.23 tokens/sec  │ 0.00%      │ 118/118             │ 200.00                      │ 179.15                       │
│ constant@10.00req/s  │ 9.88 req/s  │ 1.59 sec          │ 22.64 ms    │ 8.55 ms   │ 1812.53 tokens/sec │ 0.00%      │ 1184/1184           │ 200.00                      │ 183.53                       │
│ constant@30.00req/s  │ 29.87 req/s │ 2.45 sec          │ 41.49 ms    │ 13.38 ms  │ 5326.72 tokens/sec │ 0.00%      │ 3540/3540           │ 200.00                      │ 178.30                       │
│ constant@100.00req/s │ 41.51 req/s │ 17.09 sec         │ 14852.59 ms │ 11.65 ms  │ 7435.27 tokens/sec │ 0.00%      │ 4981/4981           │ 200.00                      │ 179.12                       │
└──────────────────────┴─────────────┴───────────────────┴─────────────┴───────────┴────────────────────┴────────────┴─────────────────────┴─────────────────────────────┴──────────────────────────────┘
