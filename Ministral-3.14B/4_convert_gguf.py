"""
4_convert_gguf.py
LoRA 어댑터 → 병합 모델 → GGUF 변환 → Ollama Modelfile 생성

사용법:
    python 4_convert_gguf.py
    python 4_convert_gguf.py --quant Q4_K_M --model_name npc-greeter

파이프라인:
    1. LoRA 어댑터를 베이스 모델에 병합
    2. 병합 모델을 HF safetensors로 저장
    3. llama.cpp convert_hf_to_gguf.py 로 GGUF 변환
    4. quantize (선택, Q4_K_M 기본)
    5. Ollama Modelfile 생성 + 등록 명령 출력

사전 요구사항:
    git clone https://github.com/ggerganov/llama.cpp
    pip install -r llama.cpp/requirements.txt
    (선택) llama.cpp/build/bin/llama-quantize 빌드
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import sys
import shutil
import argparse
import subprocess   
from pathlib import Path

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    "adapter_path":    "./lora_adapter",
    "merged_path":     "./merged_model",
    "gguf_dir":        "./gguf_output",
    "gguf_f16":        "mistral-npc-f16.gguf",
    "gguf_quant":      "mistral-npc-q4km.gguf",
    "quant_type":      "Q4_K_M",
    "model_name":      "npc-greeter",       # ollama model tag
    "llama_cpp_dir":   "./llama.cpp",       # llama.cpp 클론 경로
    "max_seq_length":  256,
    "load_in_4bit":    True,
}

SYSTEM_PROMPT = (
    "당신은 판타지 RPG 세계관의 모험가 NPC입니다. "
    "주어진 캐릭터 정보와 상황에 맞는 자연스러운 인사말을 한국어로 생성하세요. "
    "30자 이상 100자 이내로 작성하며, 영어 단어나 설정에 없는 내용은 포함하지 마세요."
)


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def run(cmd: list[str], cwd: str | None = None):
    """subprocess 실행 (실패 시 종료)"""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        print(f"[오류] 명령 실패 (returncode={result.returncode})")
        sys.exit(1)


def check_llama_cpp(llama_dir: str) -> Path:
    """llama.cpp 디렉터리와 변환 스크립트 확인"""
    p = Path(llama_dir)
    if not p.exists():
        print(f"[오류] llama.cpp 디렉터리가 없습니다: {p}")
        print("       git clone https://github.com/ggerganov/llama.cpp")
        sys.exit(1)

    convert_script = p / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"[오류] convert_hf_to_gguf.py 가 없습니다: {convert_script}")
        sys.exit(1)

    return p


def find_quantize_bin(llama_dir: Path) -> Path | None:
    """quantize 바이너리 탐색"""
    candidates = [
        llama_dir / "build" / "bin" / "llama-quantize",
        llama_dir / "build" / "bin" / "quantize",
        llama_dir / "llama-quantize",
        llama_dir / "quantize",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ──────────────────────────────────────────────
# Step 1: 어댑터 병합
# ──────────────────────────────────────────────
def merge_adapter(cfg: dict):
    print("\n[1/4] LoRA 어댑터 → 베이스 병합 중...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["adapter_path"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["load_in_4bit"],
    )

    merged_path = cfg["merged_path"]
    print(f"     병합 모델 저장: {merged_path}")
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print("     병합 완료\n")


# ──────────────────────────────────────────────
# Step 2: GGUF 변환 (F16)
# ──────────────────────────────────────────────
def convert_to_gguf(cfg: dict, llama_dir: Path):
    print("[2/4] HF → GGUF (F16) 변환 중...")
    gguf_dir = Path(cfg["gguf_dir"])
    gguf_dir.mkdir(parents=True, exist_ok=True)

    gguf_f16_path = gguf_dir / cfg["gguf_f16"]

    run([
        sys.executable,
        str(llama_dir / "convert_hf_to_gguf.py"),
        cfg["merged_path"],
        "--outtype", "f16",
        "--outfile", str(gguf_f16_path),
    ])
    print(f"     F16 GGUF 저장: {gguf_f16_path}\n")
    return gguf_f16_path


# ──────────────────────────────────────────────
# Step 3: Quantize
# ──────────────────────────────────────────────
def quantize_gguf(cfg: dict, gguf_f16_path: Path, llama_dir: Path) -> Path:
    quant_type  = cfg["quant_type"]
    gguf_dir    = Path(cfg["gguf_dir"])
    gguf_q_path = gguf_dir / cfg["gguf_quant"]

    quantize_bin = find_quantize_bin(llama_dir)
    if quantize_bin is None:
        print("[3/4] quantize 바이너리가 없음 → 양자화 스킵 (F16 그대로 사용)")
        print("      빌드 방법:")
        print("        cd llama.cpp && mkdir build && cd build")
        print("        cmake .. && cmake --build . --config Release")
        return gguf_f16_path

    print(f"[3/4] GGUF 양자화 ({quant_type}) 중...")
    run([
        str(quantize_bin),
        str(gguf_f16_path),
        str(gguf_q_path),
        quant_type,
    ])
    print(f"     양자화 GGUF 저장: {gguf_q_path}\n")
    return gguf_q_path


# ──────────────────────────────────────────────
# Step 4: Modelfile 생성
# ──────────────────────────────────────────────
def create_modelfile(cfg: dict, gguf_path: Path):
    print("[4/4] Ollama Modelfile 생성 중...")

    gguf_abs = gguf_path.resolve()

    modelfile_content = f"""# Ollama Modelfile
# 판타지 NPC 인사말 생성 모델 (Mistral-Small-3.1-14B LoRA)

FROM {gguf_abs}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"

# 추론 파라미터
PARAMETER temperature      0.75
PARAMETER top_p            0.9
PARAMETER repeat_penalty   1.1
PARAMETER num_predict      80
PARAMETER stop             \"<|im_end|>\"
PARAMETER stop             \"</s>\"
"""

    modelfile_path = Path(cfg["gguf_dir"]) / "Modelfile"
    modelfile_path.write_text(modelfile_content, encoding="utf-8")
    print(f"     Modelfile 저장: {modelfile_path}\n")

    print("=" * 60)
    print("  Ollama 등록 및 실행 명령")
    print("=" * 60)
    print(f"\n  # 모델 등록")
    print(f"  ollama create {cfg['model_name']} -f {modelfile_path}")
    print(f"\n  # 실행")
    print(f"  ollama run {cfg['model_name']}")
    print(f"\n  # API 테스트 (Unity WebRequest와 동일한 형식)")
    print(f"""  curl http://localhost:11434/api/generate -d '{{
    "model": "{cfg['model_name']}",
    "prompt": "성격: 야성형, 모험가타입: 전사, 나이: 30대, 성별: 남성, 모험가등급: A급, 방문횟수: 단골, 이전_아이템: 도끼, 재방문간격: 최근, 최근_의뢰: 대성공 일 때의 적절한 모험가 인사말을 생성해줘.",
    "stream": false
  }}'""")
    print("\n" + "=" * 60)

    return modelfile_path


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA → GGUF → Ollama 변환")
    parser.add_argument("--adapter_path",  default=DEFAULT_CONFIG["adapter_path"])
    parser.add_argument("--merged_path",   default=DEFAULT_CONFIG["merged_path"])
    parser.add_argument("--gguf_dir",      default=DEFAULT_CONFIG["gguf_dir"])
    parser.add_argument("--quant",         default=DEFAULT_CONFIG["quant_type"],
                        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                        help="양자화 타입 (기본: Q4_K_M)")
    parser.add_argument("--model_name",    default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--llama_cpp_dir", default=DEFAULT_CONFIG["llama_cpp_dir"])
    parser.add_argument("--skip_merge",    action="store_true",
                        help="이미 병합된 모델이 있으면 병합 스킵")
    parser.add_argument("--skip_quantize", action="store_true",
                        help="양자화 스킵 (F16 GGUF 그대로 사용)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {**DEFAULT_CONFIG}
    cfg.update({
        "adapter_path":  args.adapter_path,
        "merged_path":   args.merged_path,
        "gguf_dir":      args.gguf_dir,
        "quant_type":    args.quant,
        "model_name":    args.model_name,
        "llama_cpp_dir": args.llama_cpp_dir,
        "gguf_quant":    f"mistral-npc-{args.quant.lower().replace('_', '')}.gguf",
    })

    print(f"\n{'='*60}")
    print(f"  GGUF 변환 파이프라인")
    print(f"{'='*60}")
    print(f"  어댑터  : {cfg['adapter_path']}")
    print(f"  병합 저장: {cfg['merged_path']}")
    print(f"  GGUF 출력: {cfg['gguf_dir']}")
    print(f"  양자화   : {cfg['quant_type']}")
    print(f"  Ollama 태그: {cfg['model_name']}")
    print(f"{'='*60}")

    llama_dir = check_llama_cpp(cfg["llama_cpp_dir"])

    # Step 1: 병합
    if args.skip_merge and Path(cfg["merged_path"]).exists():
        print("\n[1/4] 병합 스킵 (기존 병합 모델 사용)")
    else:
        merge_adapter(cfg)

    # Step 2: F16 GGUF
    gguf_f16_path = convert_to_gguf(cfg, llama_dir)

    # Step 3: 양자화
    if args.quant == "F16" or args.skip_quantize:
        print("[3/4] 양자화 스킵 (F16 사용)\n")
        final_gguf = gguf_f16_path
    else:
        final_gguf = quantize_gguf(cfg, gguf_f16_path, llama_dir)

    # Step 4: Modelfile
    create_modelfile(cfg, final_gguf)

    print(f"\n변환 완료 → {final_gguf}")


if __name__ == "__main__":
    main()
