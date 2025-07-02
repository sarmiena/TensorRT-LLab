## Arguments for trtllm-serve
When using build_engine, models will automatically create a file here with {model_name}.default.json
Here is an example structure for one of these files:

```
{
  "notes": "Defaults created from build configuration on 2025-06-21T01:57:28+00:00",
  "args": {
    "--max_seq_len": "16355",
    "--max_num_tokens": "16355",
    "--max_batch_size": "512"
  }
}
```

When running a command like the following for "meta-llama_Llama-3.1-8B-Instruct" model:

```
./trt-llab trtllm-serve --tag mytag --serve-config my-custom-config 
```

It will try to find a config in the following order:

1. model_serve_args/meta-llama_Llama-3.1-8B-Instruct.my-custom-config.json
2. model_serve_args/my-custom-config.json


If you don't specify --serve-config, it will act as if you entered --serve-config default, and search:
1. model_serve_args/meta-llama_Llama-3.1-8B-Instruct.default.json
2. model_serve_args/default.json

The following options are valid args, as they are hard-coded in the allow list of script/build_engine's TRTLLM_SERVE_ARGS array:

```
--config_file
--metadata_server_config_file
--server_start_timeout
--request_timeout
--log_level
--tokenizer
--host
--port
--backend
--max_beam_width
--max_batch_size
--max_num_tokens
--max_seq_len
--tp_size
--pp_size
--ep_size
--cluster_size
--gpus_per_node
--kv_cache_free_gpu_memory_fraction
--num_postprocess_workers
```
