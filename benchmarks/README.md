# Benchmark archive
When running a benchmark in the tensorrt-llm container, your files will be saved here

Example:

```
./trt-llab bash --model nvidia--Llama-3.1-70B-Instruct-FP8

root@prodooser:/app/tensorrt_llm# /scripts/trtllm-bench-pytorch
```

Would produce something like:

```
benchmarks/1.0.0rc0/nvidia--Llama-3.1-70B-Instruct-FP8-pytorch-bench-07-02-2025-08-48-13.txt
```
