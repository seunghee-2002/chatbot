"""
3_convert_gguf.py — 머지 모델 → GGUF 변환 + Modelfile 생성

[사전 조건]
  - 1_train.py 실행 완료 (./lora_output/merged_model 존재)
  - git 설치: https://git-scm.com
  - cmake 설치: https://cmake.org/download (Visual Studio Build Tools 포함)

[실행]
  python 3_convert_gguf.py

[옵션]
  python 3_convert_gguf.py --merged_path ./lora_output/merged_model
  python 3_convert_gguf.py --quant Q5_K_M
  python 3_convert_gguf.py --skip_quant   # FP16 그대로 (빠른 테스트용)

[완료 후 Ollama 등록]
  ollama create npc-greeter -f ./gguf_output/Modelfile

양자화 타입별 특성 (RTX 4070 12GB 기준):
  Q4_K_M  : VRAM ~5GB, 속도 빠름, 품질 양호  <- 기본값
  Q5_K_M  : VRAM ~6GB, 품질 더 좋음
  Q8_0    : VRAM ~9GB, FP16에 가까운 품질
  Q2_K    : VRAM ~3GB, 품질 저하 (최소 사양용)
"""

import argparse
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────

MERGED_MODEL_PATH = "./lora_output/merged_model"
GGUF_QUANT        = "Q4_K_M"
OLLAMA_MODEL_NAME = "npc-greeter"
LLAMA_CPP_DIR     = "./llama.cpp"
GGUF_DIR          = "./gguf_output"

IS_WINDOWS = platform.system() == "Windows"

# ────────────────────────────────────────────────
# 헬퍼
# ────────────────────────────────────────────────

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def die(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)

def run(cmd: list, cwd: str = None):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

# ────────────────────────────────────────────────
# 1. 사전 조건 확인
# ────────────────────────────────────────────────

def check_prereqs(merged_path: str):
    log("사전 조건 확인 중...")

    if not Path(merged_path).is_dir():
        die(f"머지 모델 없음: {merged_path}\n-> 1_train.py 먼저 실행하세요.")

    if not shutil.which("git"):
        die("git 미설치.\n-> https://git-scm.com 에서 설치 후 재실행")

    if not shutil.which("cmake"):
        log("⚠  cmake 미발견 — 양자화 빌드 필요 시 설치 필요")
        log("   https://cmake.org/download (Visual Studio Build Tools 포함)")

    log("사전 조건 OK")

# ────────────────────────────────────────────────
# 2. llama.cpp 클론 / 업데이트
# ────────────────────────────────────────────────

def setup_llama_cpp():
    llama_dir = Path(LLAMA_CPP_DIR)

    if llama_dir.is_dir():
        log("기존 llama.cpp 업데이트 중...")
        run(["git", "-C", str(llama_dir), "pull", "--ff-only"])
    else:
        log("llama.cpp 클론 중...")
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp", str(llama_dir)])

    req_file = llama_dir / "requirements.txt"
    if req_file.exists():
        log("llama.cpp Python 의존성 설치 중...")
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)])

    log("llama.cpp 설정 완료")

# ────────────────────────────────────────────────
# 3. HuggingFace 머지 모델 → GGUF (FP16) 변환
# ────────────────────────────────────────────────

def convert_to_fp16(merged_path: str) -> Path:
    gguf_dir = Path(GGUF_DIR)
    gguf_dir.mkdir(parents=True, exist_ok=True)
    fp16_path = gguf_dir / "model_fp16.gguf"

    if fp16_path.exists():
        log(f"FP16 GGUF 이미 존재 → 스킵: {fp16_path}")
        return fp16_path

    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        die(f"변환 스크립트 없음: {convert_script}")

    log("HuggingFace 모델 → GGUF(FP16) 변환 중... (수 분 소요)")
    run([
        sys.executable, str(convert_script),
        merged_path,
        "--outfile", str(fp16_path),
        "--outtype", "f16",
    ])
    log(f"FP16 GGUF 저장 완료: {fp16_path}")
    return fp16_path

# ────────────────────────────────────────────────
# 4. 양자화
# ────────────────────────────────────────────────

def _build_llama_cpp():
    llama_dir = Path(LLAMA_CPP_DIR)
    build_dir = llama_dir / "build"

    if not shutil.which("cmake"):
        die("cmake 미설치.\n-> https://cmake.org/download 참고")

    log("llama.cpp 빌드 중... (첫 실행 시 수 분 소요)")
    run(["cmake", "-S", str(llama_dir), "-B", str(build_dir),
         "-DGGML_CUDA=OFF", "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(build_dir), "--config", "Release"])
    log("llama.cpp 빌드 완료")


def _find_quantize_bin() -> Path:
    bin_name = "llama-quantize.exe" if IS_WINDOWS else "llama-quantize"
    candidates = [
        Path(LLAMA_CPP_DIR) / "build" / "bin" / "Release" / bin_name,
        Path(LLAMA_CPP_DIR) / "build" / "bin" / bin_name,
        Path(LLAMA_CPP_DIR) / "build" / "Release" / bin_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def quantize(fp16_path: Path, quant_type: str) -> Path:
    quant_path = Path(GGUF_DIR) / f"model_{quant_type}.gguf"

    if quant_path.exists():
        log(f"양자화 GGUF 이미 존재 → 스킵: {quant_path}")
        return quant_path

    quantize_bin = _find_quantize_bin()
    if not quantize_bin:
        log("llama-quantize 바이너리 없음 → 빌드 시작")
        _build_llama_cpp()
        quantize_bin = _find_quantize_bin()

    if not quantize_bin:
        die("빌드 후에도 llama-quantize 바이너리를 찾을 수 없습니다.")

    log(f"양자화: FP16 → {quant_type} (수 분 소요)")
    run([str(quantize_bin), str(fp16_path), str(quant_path), quant_type])
    log(f"양자화 완료: {quant_path}")
    return quant_path

# ────────────────────────────────────────────────
# 5. Modelfile 생성
# ────────────────────────────────────────────────

def create_modelfile(gguf_path: Path) -> Path:
    modelfile_path = Path(GGUF_DIR) / "Modelfile"

    # Windows 역슬래시 → 슬래시 변환 (Ollama 호환)
    gguf_str = str(gguf_path.resolve()).replace("\\", "/")

    content = (
        f"FROM {gguf_str}\n"
        "\n"
        'SYSTEM """\n'
        "당신은 판타지 RPG 게임 속 모험가 NPC입니다.\n"
        "주어진 조건(성격, 직업, 등급, 방문 이력, 아이템, 모험 결과)에 맞춰 자연스러운 한국어 인사말을 한 문장으로 생성하세요.\n"
        "영어 단어, 몬스터명, 지역명은 사용하지 마세요.\n"
        '30자 이상 100자 이하로 작성하세요.\n'
        '"""\n'
        "\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER repeat_penalty 1.1\n"
        "PARAMETER num_predict 60\n"
        'PARAMETER stop "<|eot_id|>"\n'
        'PARAMETER stop "<|end_of_text|>"\n'
    )

    modelfile_path.write_text(content, encoding="utf-8")
    log(f"Modelfile 생성 완료: {modelfile_path}")
    return modelfile_path

# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GGUF 변환 및 Modelfile 생성")
    parser.add_argument("--merged_path", default=MERGED_MODEL_PATH, help="머지 모델 경로")
    parser.add_argument("--quant",       default=GGUF_QUANT,        help="양자화 타입 (기본: Q4_K_M)")
    parser.add_argument("--skip_quant",  action="store_true",        help="양자화 스킵, FP16 그대로 사용")
    args = parser.parse_args()

    log("=== GGUF 변환 시작 ===")
    log(f"  머지 모델 : {args.merged_path}")
    log(f"  양자화    : {'스킵 (FP16)' if args.skip_quant else args.quant}")

    check_prereqs(args.merged_path)
    setup_llama_cpp()

    fp16_path = convert_to_fp16(args.merged_path)

    if args.skip_quant:
        final_gguf = fp16_path
        log("양자화 스킵 → FP16 그대로 사용")
    else:
        final_gguf = quantize(fp16_path, args.quant)

    create_modelfile(final_gguf)

    log("")
    log("=== 완료 ===")
    log(f"  GGUF 파일 : {final_gguf}")
    log(f"  Modelfile : {Path(GGUF_DIR) / 'Modelfile'}")
    log("")
    log("Ollama 등록 방법 (Ollama 설치 후):")
    log(f"  ollama create {OLLAMA_MODEL_NAME} -f {Path(GGUF_DIR) / 'Modelfile'}")


if __name__ == "__main__":
    main()