## extra_llm_api_options
This is where you want to put your yml files that you can use when doing trtllm-serve with the --extra_llm_api_options flag.

When running something like:

```
./trt-llab trtllm-serve --model Llama-3_3-Nemotron-Super-49B-v1-FP8 --tag default --serve-config my_config
```

It will load engines/Llama-3_3-Nemotron-Super-49B-v1-FP8/default/serve-args.json

Example serve-args.json
```
{
  "default": {
    "notes": "Defaults created from build configuration on 2025-06-21T01:57:28+00:00",
    "args": {
      "--max_seq_len": "16355",
      "--max_num_tokens": "16355",
      "--max_batch_size": "512"
    }
  },
  "my_config": {
    "notes": "From developers at NVIDIA support saying serve it from the downloaded HF repo",
    "args": {
      "--backend": "pytorch",
      "--port": "8000",
      "--max_batch_size": 128,
      "--tp_size": 2,
      "--max_num_tokens": 2048,
      "--extra_llm_api_options": "pytorch-small-batch.yml"
    }
  }
}

```

Notice the line:

```
    "--extra_llm_api_options": "pytorch-small-batch.yml"
```

That will look for pytorch-small-batch.yml in side the extra_llm_api_options/ directory of this repo for the given filename
