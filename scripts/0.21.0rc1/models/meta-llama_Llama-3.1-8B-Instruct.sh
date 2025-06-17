#!/bin/bash
# Model configuration for meta-llama_Llama-3.1-8B-Instruct

build_model() {
    # Default quantize.py arguments for this model
    quantize_base_args=(
        "--model_dir" "/model"
        "--dtype" "bfloat16"
        "--qformat" "fp8"
        "--kv_cache_dtype" "fp8"
        "--output_dir" "/ckpt"
        "--pp_size" "2"
        "--calib_size" "512"
    )
    
    # Build and execute quantize command
    quantize_cmd=$(build_quantize_cmd "${quantize_base_args[@]}")
    echo "Executing: $quantize_cmd"
    eval "$quantize_cmd"
    
    # Default trtllm-build arguments for this model
    trtllm_base_args=(
        "--checkpoint_dir" "/ckpt"
        "--output_dir" "$MODEL_OUTPUT_DIR"
        "--remove_input_padding" "enable"
        "--kv_cache_type" "paged"
        "--max_batch_size" "512"
        "--max_num_tokens" "16355"
        "--max_seq_len" "16355"
        "--use_paged_context_fmha" "enable"
        "--use_fp8_context_fmha" "enable"
        "--gemm_plugin" "disable"
        "--multiple_profiles" "enable"
    )
    
    # Build and execute trtllm-build command
    trtllm_cmd=$(build_trtllm_cmd "${trtllm_base_args[@]}")
    echo "Executing: $trtllm_cmd"
    eval "$trtllm_cmd"
    
    # Save build command details
    save_build_command "$MODEL_OUTPUT_DIR" "$quantize_cmd" "$trtllm_cmd"
}
