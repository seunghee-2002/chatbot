"""
llama.cpp 변환 스크립트 자동 다운로드 후 GGUF Q4_K_M 변환
사전 조건: pip install huggingface-hub
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download

MERGED_DIR  = "./qwen3-greeting-gguf"
BF16_GGUF   = "./qwen3-greeting-gguf/model-bf16.gguf"
OUTPUT_GGUF = "./qwen3-greeting-gguf/model-q4_k_m.gguf"
SCRIPTS_DIR = "./llama_scripts"

# ─────────────────────────────────────────────
# 1. 변환에 필요한 스크립트 다운로드
# ─────────────────────────────────────────────
print("[1/3] llama.cpp 변환 스크립트를 다운로드합니다...")
os.makedirs(SCRIPTS_DIR, exist_ok=True)

files_to_download = [
    "convert_hf_to_gguf.py",
    "gguf-py/gguf/__init__.py",
    "gguf-py/gguf/gguf_writer.py",
    "gguf-py/gguf/gguf_reader.py",
    "gguf-py/gguf/constants.py",
    "gguf-py/gguf/quants.py",
    "gguf-py/gguf/tensor_mapping.py",
    "gguf-py/gguf/vocab.py",
    "gguf-py/gguf/metadata.py",
    "gguf-py/gguf/lazy.py",
    "gguf-py/gguf/utility.py",
]

for file in files_to_download:
    try:
        hf_hub_download(
            repo_id="ggml-org/llama.cpp",
            filename=file,
            local_dir=SCRIPTS_DIR,
            repo_type="model"
        )
        print(f"  완료: {file}")
    except Exception as e:
        print(f"  건너뜀 ({file}): {e}")

# gguf 패키지 경로 추가
gguf_pkg = str(Path(SCRIPTS_DIR) / "gguf-py")
if gguf_pkg not in sys.path:
    sys.path.insert(0, gguf_pkg)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

convert_script = str(Path(SCRIPTS_DIR) / "convert_hf_to_gguf.py")

# ─────────────────────────────────────────────
# 2. safetensors → bf16 GGUF
# ─────────────────────────────────────────────
print("\n[2/3] safetensors → bf16 GGUF 변환 중...")
result = subprocess.run(
    [sys.executable, convert_script,
     MERGED_DIR,
     "--outfile", BF16_GGUF,
     "--outtype", "bf16"],
)

if result.returncode != 0:
    print("[오류] bf16 변환 실패")
    sys.exit(1)

print("bf16 변환 완료!")

# ─────────────────────────────────────────────
# 3. bf16 GGUF → Q4_K_M 양자화
# ─────────────────────────────────────────────
print("\n[3/3] bf16 → Q4_K_M 양자화 중...")

try:
    from llama_cpp import llama_model_quantize_params, llama_model_quantize
    import ctypes

    params = llama_model_quantize_params()
    params.nthread = 0
    params.ftype = 15  # Q4_K_M

    ret = llama_model_quantize(
        BF16_GGUF.encode(),
        OUTPUT_GGUF.encode(),
        ctypes.byref(params)
    )

    if ret != 0:
        raise RuntimeError(f"양자화 실패 코드: {ret}")

    os.remove(BF16_GGUF)
    print(f"\n{'='*50}")
    print("GGUF 변환 완료!")
    print(f"파일 위치: {OUTPUT_GGUF}")
    print(f"{'='*50}")
    print("\nOllama 등록 방법:")
    print("  1. Modelfile 파일 생성 후 아래 내용 작성:")
    print(f'       FROM {os.path.abspath(OUTPUT_GGUF)}')
    print('       PARAMETER stop "<|im_end|>"')
    print("  2. ollama create qwen3-greeting -f Modelfile")
    print("  3. ollama run qwen3-greeting")

except Exception as e:
    print(f"\n[안내] Q4 양자화 실패: {e}")
    print("bf16 GGUF가 생성되어 있어 Ollama에서 바로 사용 가능합니다.")
    print(f"  파일: {BF16_GGUF}  (약 15GB, Q4보다 크지만 동작은 동일)")
    print("\nOllama 등록 방법:")
    print("  1. Modelfile 파일 생성 후 아래 내용 작성:")
    print(f'       FROM {os.path.abspath(BF16_GGUF)}')
    print('       PARAMETER stop "<|im_end|>"')
    print("  2. ollama create qwen3-greeting -f Modelfile")
    print("  3. ollama run qwen3-greeting")