import sys
from pathlib import Path

# 模拟打包后的环境
print(f"sys.frozen: {getattr(sys, 'frozen', False)}")
print(f"sys.executable: {sys.executable}")

app_dir = Path(sys.executable).parent
print(f"app_dir: {app_dir}")

# 检查各种路径
model_path_1 = app_dir / "model" / "Qwen3-30B-A3B-Q4_K_M.gguf"
model_path_2 = app_dir / "_internal" / "model" / "Qwen3-30B-A3B-Q4_K_M.gguf"
llama_path_1 = app_dir / "llama-cli.exe"
llama_path_2 = app_dir / "_internal" / "llama-cli.exe"

print(f"\nmodel_path_1: {model_path_1}")
print(f"model_path_1 exists: {model_path_1.exists()}")
print(f"model_path_2: {model_path_2}")
print(f"model_path_2 exists: {model_path_2.exists()}")

print(f"\nllama_path_1: {llama_path_1}")
print(f"llama_path_1 exists: {llama_path_1.exists()}")
print(f"llama_path_2: {llama_path_2}")
print(f"llama_path_2 exists: {llama_path_2.exists()}")

# 最终选择的路径
model_path = model_path_1 if model_path_1.exists() else model_path_2
llama_path = llama_path_1 if llama_path_1.exists() else llama_path_2

print(f"\n最终 model_path: {model_path}")
print(f"最终 llama_path: {llama_path}")
print(f"模型存在: {model_path.exists()}")
print(f"llama存在: {llama_path.exists()}")
