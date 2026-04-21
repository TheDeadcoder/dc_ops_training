Unsloth: AMD currently is not stable with 4bit bitsandbytes. Disabling for now.
INFO 04-21 12:32:08 [vllm_utils.py:724] Unsloth: Patching vLLM v1 graph capture
==((====))==  Unsloth 2026.4.4: Fast Qwen2 patching. Transformers: 4.54.1. vLLM: 0.19.1+rocm721.
   \\   /|    AMD Instinct MI300X VF. Num GPUs = 1. Max memory: 191.688 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+rocm7.1. ROCm Toolkit: 7.1.25424. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = True]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Unsloth: vLLM loading unsloth/Qwen2.5-7B-Instruct with actual GPU utilization = 64.88%
Unsloth: Your GPU has CUDA compute capability 9.4 with VRAM = 191.69 GB.
Unsloth: Using conservativeness = 1.0. Chunked prefill tokens = 3072. Num Sequences = 256.
Unsloth: vLLM's KV Cache can use up to 109.96 GB. Also swap space = 6 GB.
Unsloth: Not an error, but `level` is not supported in vLLM.config.CompilationConfig. Skipping.
Unsloth: Not an error, but `use_cudagraph` is not supported in vLLM.config.CompilationConfig. Skipping.
Unsloth: Not an error, but `use_inductor` is not supported in vLLM.config.CompilationConfig. Skipping.
Unsloth: Not an error, but `swap_space` is not supported in vLLM. Skipping.
Unsloth: Not an error, but `device` is not supported in vLLM. Skipping.
INFO 04-21 12:32:10 [utils.py:233] non-default args: {'dtype': torch.bfloat16, 'max_model_len': 3072, 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.6488409887512226, 'max_num_batched_tokens': 8192, 'max_num_seqs': 256, 'max_logprobs': 0, 'disable_log_stats': True, 'enforce_eager': True, 'enable_lora': True, 'enable_chunked_prefill': True, 'compilation_config': {'mode': None, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'splitting_ops': None, 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': None, 'compile_ranges_endpoints': None, 'inductor_compile_config': {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 20, 'group_fusion': True, 'disable_progress': False, 'verbose_progress': True, 'triton.multi_kernel': 0, 'triton.use_block_ptr': True, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True, 'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': None, 'static_all_moe_layers': []}, 'model': 'unsloth/Qwen2.5-7B-Instruct'}
WARNING 04-21 12:32:10 [envs.py:1744] Unknown vLLM environment variable detected: VLLM_USE_V1
WARNING 04-21 12:32:10 [envs.py:1744] Unknown vLLM environment variable detected: VLLM_USE_TRITON_FLASH_ATTN
WARNING 04-21 12:32:10 [arg_utils.py:1390] The global random seed is set to 0. Since VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may affect the random state of the Python process that launched vLLM.
INFO 04-21 12:32:10 [model.py:549] Resolved architecture: Qwen2ForCausalLM
Parse safetensors files: 100%|██████████| 4/4 [00:00<00:00, 15.83it/s]
INFO 04-21 12:32:11 [model.py:1678] Using max model len 3072
INFO 04-21 12:32:11 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=8192.
INFO 04-21 12:32:11 [vllm.py:790] Asynchronous scheduling is enabled.
WARNING 04-21 12:32:11 [vllm.py:848] Enforce eager set, disabling torch.compile and CUDAGraphs. This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
WARNING 04-21 12:32:11 [vllm.py:859] Inductor compilation was disabled by user settings, optimizations settings that are only active during inductor compilation will be ignored.
INFO 04-21 12:32:11 [vllm.py:1025] Cudagraph is disabled under eager mode
INFO 04-21 12:32:11 [compilation.py:292] Enabled custom fusions: norm_quant, act_quant
INFO 04-21 12:32:11 [core.py:105] Initializing a V1 LLM engine (v0.19.1) with config: model='unsloth/Qwen2.5-7B-Instruct', speculative_config=None, tokenizer='unsloth/Qwen2.5-7B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=3072, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=unsloth/Qwen2.5-7B-Instruct, enable_prefix_caching=True, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['+sparse_attn_indexer', 'all'], 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_images_per_batch': 0, 'compile_sizes': [], 'compile_ranges_endpoints': [8192], 'inductor_compile_config': {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 20, 'group_fusion': True, 'disable_progress': False, 'verbose_progress': True, 'triton.multi_kernel': 0, 'triton.use_block_ptr': True, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': False, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True, 'enable_auto_functionalized_v2': False, 'size_asserts': False, 'alignment_asserts': False, 'scalar_asserts': False}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 1, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': True, 'static_all_moe_layers': []}
INFO 04-21 12:32:11 [parallel_state.py:1400] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://134.199.202.72:44831 backend=nccl
[rank0]:[W421 12:32:11.410833891 ProcessGroupGloo.cpp:511] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
INFO 04-21 12:32:11 [parallel_state.py:1716] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, PCP rank 0, TP rank 0, EP rank N/A, EPLB rank N/A
INFO 04-21 12:32:12 [gpu_model_runner.py:4735] Starting to load model unsloth/Qwen2.5-7B-Instruct...
INFO 04-21 12:32:12 [rocm.py:496] Using ROCM_ATTN backend out of potential backends: ['ROCM_ATTN', 'TRITON_ATTN'].
WARNING 04-21 12:32:12 [compilation.py:1223] Op 'sparse_attn_indexer' not present in model, enabling with '+sparse_attn_indexer' has no effect
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:01<00:04,  1.54s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [00:03<00:03,  1.62s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [00:04<00:01,  1.55s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:05<00:00,  1.11s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [00:05<00:00,  1.28s/it]

INFO 04-21 12:32:18 [default_loader.py:384] Loading weights took 5.15 seconds
INFO 04-21 12:32:18 [punica_selector.py:20] Using PunicaWrapperGPU.
INFO 04-21 12:32:18 [gpu_model_runner.py:4820] Model loading took 14.43 GiB memory and 5.603437 seconds
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/scripts/run_grpo.py", line 465, in _safe_expand
[rank0]:     _orig_expand(*args, **kwargs)
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/_ops.py", line 1209, in __call__
[rank0]:     return self._op(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: NotImplementedError: Could not run 'vllm::lora_expand' with arguments from the 'CUDA' backend. This could be because the operator doesn't exist for this backend, or was omitted during the selective/custom build process (if using custom build). If you are a Facebook employee using PyTorch on mobile, please visit https://fburl.com/ptmfixes for possible resolutions. 'vllm::lora_expand' is only available for these backends: [CPU, Meta, BackendSelect, Python, FuncTorchDynamicLayerBackMode, Functionalize, Named, Conjugate, Negative, ZeroTensor, ADInplaceOrView, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradMPS, AutogradXPU, AutogradHPU, AutogradLazy, AutogradMTIA, AutogradMAIA, AutogradPrivateUse1, AutogradMeta, Tracer, AutocastCPU, AutocastMTIA, AutocastMAIA, AutocastXPU, AutocastMPS, AutocastCUDA, FuncTorchBatched, BatchedNestedTensor, FuncTorchVmapMode, Batched, VmapMode, FuncTorchGradWrapper, PythonTLSSnapshot, FuncTorchDynamicLayerFrontMode, PreDispatch, PythonDispatcher].

[rank0]: CPU: registered at /root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/utils/torch_utils.py:789 [kernel]
[rank0]: Meta: registered at /root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/utils/torch_utils.py:789 [kernel]
[rank0]: BackendSelect: fallthrough registered at /pytorch/aten/src/ATen/core/BackendSelectFallbackKernel.cpp:3 [backend fallback]
[rank0]: Python: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:198 [backend fallback]
[rank0]: FuncTorchDynamicLayerBackMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:477 [backend fallback]
[rank0]: Functionalize: registered at /pytorch/aten/src/ATen/FunctionalizeFallbackKernel.cpp:384 [backend fallback]
[rank0]: Named: registered at /pytorch/aten/src/ATen/core/NamedRegistrations.cpp:5 [backend fallback]
[rank0]: Conjugate: registered at /pytorch/aten/src/ATen/ConjugateFallback.cpp:17 [backend fallback]
[rank0]: Negative: registered at /pytorch/aten/src/ATen/native/NegateFallback.cpp:18 [backend fallback]
[rank0]: ZeroTensor: registered at /pytorch/aten/src/ATen/ZeroTensorFallback.cpp:115 [backend fallback]
[rank0]: ADInplaceOrView: fallthrough registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:103 [backend fallback]
[rank0]: AutogradOther: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:62 [backend fallback]
[rank0]: AutogradCPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:66 [backend fallback]
[rank0]: AutogradCUDA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:74 [backend fallback]
[rank0]: AutogradXLA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:86 [backend fallback]
[rank0]: AutogradMPS: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:94 [backend fallback]
[rank0]: AutogradXPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:70 [backend fallback]
[rank0]: AutogradHPU: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:107 [backend fallback]
[rank0]: AutogradLazy: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:90 [backend fallback]
[rank0]: AutogradMTIA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:78 [backend fallback]
[rank0]: AutogradMAIA: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:82 [backend fallback]
[rank0]: AutogradPrivateUse1: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:111 [backend fallback]
[rank0]: AutogradMeta: registered at /pytorch/aten/src/ATen/core/VariableFallbackKernel.cpp:98 [backend fallback]
[rank0]: Tracer: registered at /pytorch/torch/csrc/autograd/TraceTypeManual.cpp:296 [backend fallback]
[rank0]: AutocastCPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:324 [backend fallback]
[rank0]: AutocastMTIA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:468 [backend fallback]
[rank0]: AutocastMAIA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:506 [backend fallback]
[rank0]: AutocastXPU: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:544 [backend fallback]
[rank0]: AutocastMPS: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:209 [backend fallback]
[rank0]: AutocastCUDA: fallthrough registered at /pytorch/aten/src/ATen/autocast_mode.cpp:165 [backend fallback]
[rank0]: FuncTorchBatched: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:727 [backend fallback]
[rank0]: BatchedNestedTensor: registered at /pytorch/aten/src/ATen/functorch/LegacyBatchingRegistrations.cpp:754 [backend fallback]
[rank0]: FuncTorchVmapMode: fallthrough registered at /pytorch/aten/src/ATen/functorch/VmapModeRegistrations.cpp:22 [backend fallback]
[rank0]: Batched: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1072 [backend fallback]
[rank0]: VmapMode: fallthrough registered at /pytorch/aten/src/ATen/VmapModeRegistrations.cpp:32 [backend fallback]
[rank0]: FuncTorchGradWrapper: registered at /pytorch/aten/src/ATen/functorch/TensorWrapper.cpp:210 [backend fallback]
[rank0]: PythonTLSSnapshot: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:206 [backend fallback]
[rank0]: FuncTorchDynamicLayerFrontMode: registered at /pytorch/aten/src/ATen/functorch/DynamicLayer.cpp:473 [backend fallback]
[rank0]: PreDispatch: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:210 [backend fallback]
[rank0]: PythonDispatcher: registered at /pytorch/aten/src/ATen/core/PythonFallbackKernel.cpp:202 [backend fallback]


[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/unsloth_zoo/vllm_utils.py", line 2296, in load_vllm
[rank0]:     llm = LLM(**engine_args)
[rank0]:           ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 382, in __init__
[rank0]:     self.llm_engine = LLMEngine.from_engine_args(
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 177, in from_engine_args
[rank0]:     return cls(
[rank0]:            ^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 111, in __init__
[rank0]:     self.engine_core = EngineCoreClient.make_client(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 103, in make_client
[rank0]:     return InprocClient(vllm_config, executor_class, log_stats)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 285, in __init__
[rank0]:     self.engine_core = EngineCore(*args, **kwargs)
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 124, in __init__
[rank0]:     kv_cache_config = self._initialize_kv_caches(vllm_config)
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/engine/core.py", line 247, in _initialize_kv_caches
[rank0]:     available_gpu_memory = self.model_executor.determine_available_memory()
[rank0]:                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/executor/abstract.py", line 136, in determine_available_memory
[rank0]:     return self.collective_rpc("determine_available_memory")
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/executor/uniproc_executor.py", line 80, in collective_rpc
[rank0]:     result = run_method(self.driver_worker, method, args, kwargs)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/serial_utils.py", line 510, in run_method
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 370, in determine_available_memory
[rank0]:     self.model_runner.profile_run()
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 5782, in profile_run
[rank0]:     hidden_states, last_hidden_states = self._dummy_run(
[rank0]:                                         ^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
[rank0]:     return func(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 5474, in _dummy_run
[rank0]:     outputs = self.model(
[rank0]:               ^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 583, in forward
[rank0]:     hidden_states = self.model(
[rank0]:                     ^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/compilation/decorators.py", line 467, in __call__
[rank0]:     return self.forward(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 444, in forward
[rank0]:     hidden_states, residual = layer(positions, hidden_states, residual)
[rank0]:                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 304, in forward
[rank0]:     hidden_states = self.self_attn(
[rank0]:                     ^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen2.py", line 214, in forward
[rank0]:     qkv, _ = self.qkv_proj(hidden_states)
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/lora/layers/column_parallel_linear.py", line 137, in forward
[rank0]:     output_parallel = self.apply(input_, bias)
[rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/lora/layers/base_linear.py", line 134, in apply
[rank0]:     lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
[rank0]:                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/lora/punica_wrapper/punica_gpu.py", line 255, in add_lora_linear
[rank0]:     self.add_expand(
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/vllm/lora/punica_wrapper/punica_gpu.py", line 156, in add_expand
[rank0]:     lora_expand(
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/scripts/run_grpo.py", line 467, in _safe_expand
[rank0]:     _pt_lora_expand(*args, **kwargs)
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/scripts/run_grpo.py", line 442, in _pt_lora_expand
[rank0]:     _torch.bmm(_wb, _inp.unsqueeze(-1)).squeeze(-1).mul_(scale)
[rank0]:     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [8192, 16] but got: [8192, 4608].

[rank0]: During handling of the above exception, another exception occurred:

[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/scripts/run_grpo.py", line 777, in <module>
[rank0]:     main()
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/scripts/run_grpo.py", line 614, in main
[rank0]:     model, tokenizer = FastLanguageModel.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/unsloth/models/loader.py", line 716, in from_pretrained
[rank0]:     model, tokenizer = dispatch_model.from_pretrained(
[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/unsloth/models/qwen2.py", line 88, in from_pretrained
[rank0]:     return FastLlamaModel.from_pretrained(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/unsloth/models/llama.py", line 2493, in from_pretrained
[rank0]:     llm = load_vllm(**load_vllm_kwargs)
[rank0]:           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/dc_ops_training/dc-ops-amd/.venv/lib/python3.12/site-packages/unsloth_zoo/vllm_utils.py", line 2353, in load_vllm
[rank0]:     raise RuntimeError(error)
[rank0]: RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [8192, 16] but got: [8192, 4608].
[rank0]:[W421 12:32:20.223427577 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())