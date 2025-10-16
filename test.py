# test_vllm.py
from vllm import LLM
llm = LLM(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    trust_remote_code=True,
    gpu_memory_utilization=0.75,
)
print("✓ vLLM FP8 loaded successfully!")
print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
