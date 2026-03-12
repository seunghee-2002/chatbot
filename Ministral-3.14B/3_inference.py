"""
3_inference.py
파인튜닝된 LoRA 어댑터 추론 스크립트

사용법:
    # 대화형 모드
    python 3_inference.py

    # 단일 입력
    python 3_inference.py --instruction "성격: 야성형, 모험가타입: 전사, ..."

    # 배치 테스트 (내장 테스트케이스 실행)
    python 3_inference.py --batch_test
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import unsloth
import argparse
from unsloth import FastLanguageModel

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
ADAPTER_PATH    = "./lora_adapter"
MAX_SEQ_LENGTH  = 256
LOAD_IN_4BIT    = True

GEN_CONFIG = {
    "max_new_tokens":  80,
    "temperature":     0.75,
    "top_p":           0.9,
    "repetition_penalty": 1.1,
    "do_sample":       True,
}

SYSTEM_PROMPT = (
    "당신은 판타지 RPG 세계관의 모험가 NPC입니다. "
    "주어진 캐릭터 정보와 상황에 맞는 자연스러운 인사말을 한국어로 생성하세요. "
    "30자 이상 100자 이내로 작성하며, 영어 단어나 설정에 없는 내용은 포함하지 마세요."
)

# 내장 테스트케이스 (보고서 4차 학습 기준)
TEST_CASES = [
    {
        "label": "존칭형 도적 / 첫방문 / D급",
        "instruction": (
            "성격: 존칭형, 모험가타입: 도적, 나이: 20대, 성별: 여성, "
            "모험가등급: D급, 방문횟수: 첫방문, 이전_아이템: 없음, "
            "재방문간격: 없음, 최근_의뢰: 첫방문 일 때의 적절한 모험가 인사말을 생성해줘."
        ),
    },
    {
        "label": "야성형 전사 / 단골 / S급 / 도끼 / 대성공",
        "instruction": (
            "성격: 야성형, 모험가타입: 전사, 나이: 30대, 성별: 남성, "
            "모험가등급: S급, 방문횟수: 단골, 이전_아이템: 도끼, "
            "재방문간격: 최근, 최근_의뢰: 대성공 일 때의 적절한 모험가 인사말을 생성해줘."
        ),
    },
    {
        "label": "하대형 전사 / 오랜만 / 검 / 성공",
        "instruction": (
            "성격: 하대형, 모험가타입: 전사, 나이: 40대, 성별: 남성, "
            "모험가등급: A급, 방문횟수: 많음, 이전_아이템: 검, "
            "재방문간격: 오래됨, 최근_의뢰: 성공 일 때의 적절한 모험가 인사말을 생성해줘."
        ),
    },
    {
        "label": "단답형 도적 / 단검 / 대성공",
        "instruction": (
            "성격: 단답형, 모험가타입: 도적, 나이: 20대, 성별: 남성, "
            "모험가등급: A급, 방문횟수: 보통, 이전_아이템: 단검, "
            "재방문간격: 최근, 최근_의뢰: 대성공 일 때의 적절한 모험가 인사말을 생성해줘."
        ),
    },
    {
        "label": "★비매칭 너스레형 궁수 / 단검 / 성공",
        "instruction": (
            "성격: 너스레형, 모험가타입: 궁수, 나이: 30대, 성별: 여성, "
            "모험가등급: B급, 방문횟수: 보통, 이전_아이템: 단검, "
            "재방문간격: 보통, 최근_의뢰: 성공 일 때의 적절한 모험가 인사말을 생성해줘."
        ),
    },
    {
        "label": "★비매칭 존칭형 마법사 / 망치 / 실패",
        "instruction": (
            "성격: 존칭형, 모험가타입: 마법사, 나이: 20대, 성별: 여성, "
            "모험가등급: C급, 방문횟수: 적음, 이전_아이템: 망치, "
            "재방문간격: 보통, 최근_의뢰: 실패 일 때의 적절한 모험가 인사말을 생성해줘."
        ),
    },
]


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_model(adapter_path: str):
    print(f"[모델 로드] {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)

    # 비전 모델 tokenizer → 텍스트 전용 tokenizer로 교체
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print("  로드 완료\n")
    return model, tokenizer


# ──────────────────────────────────────────────
# 추론
# ──────────────────────────────────────────────
def generate(model, tokenizer, instruction: str) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=GEN_CONFIG["max_new_tokens"],
        max_length=None,          # ← 추가
        temperature=GEN_CONFIG["temperature"],
        top_p=GEN_CONFIG["top_p"],
        repetition_penalty=GEN_CONFIG["repetition_penalty"],
        do_sample=GEN_CONFIG["do_sample"],
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


# ──────────────────────────────────────────────
# 모드
# ──────────────────────────────────────────────
def run_batch_test(model, tokenizer):
    print("=" * 60)
    print("  배치 테스트 (내장 테스트케이스)")
    print("=" * 60)
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}] {tc['label']}")
        print(f"  입력: {tc['instruction'][:80]}...")
        response = generate(model, tokenizer, tc["instruction"])
        print(f"  출력: {response}")
        char_count = len(response)
        ok = "✓" if 30 <= char_count <= 100 else "✗"
        print(f"  길이: {char_count}자  {ok}")
    print("\n" + "=" * 60)


def run_single(model, tokenizer, instruction: str):
    print(f"\n입력: {instruction}")
    response = generate(model, tokenizer, instruction)
    print(f"출력: {response}")
    print(f"길이: {len(response)}자")


def run_interactive(model, tokenizer):
    print("=" * 60)
    print("  대화형 추론 모드  (종료: 'q' 또는 Ctrl+C)")
    print("=" * 60)
    print("파라미터 예시:")
    print("  성격: 야성형, 모험가타입: 전사, 나이: 30대, 성별: 남성,")
    print("  모험가등급: A급, 방문횟수: 단골, 이전_아이템: 도끼,")
    print("  재방문간격: 최근, 최근_의뢰: 대성공")
    print("  일 때의 적절한 모험가 인사말을 생성해줘.")
    print()

    while True:
        try:
            instruction = input("입력 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if instruction.lower() in ("q", "quit", "exit", "종료"):
            print("종료합니다.")
            break
        if not instruction:
            continue

        response = generate(model, tokenizer, instruction)
        print(f"출력 > {response}")
        print(f"(길이: {len(response)}자)\n")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA 어댑터 추론")
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--instruction",  default=None,
                        help="단일 instruction 문자열")
    parser.add_argument("--batch_test",   action="store_true",
                        help="내장 테스트케이스 배치 실행")
    parser.add_argument("--temperature",  type=float, default=GEN_CONFIG["temperature"])
    parser.add_argument("--max_new_tokens", type=int, default=GEN_CONFIG["max_new_tokens"])
    return parser.parse_args()


def main():
    args = parse_args()
    GEN_CONFIG["temperature"]    = args.temperature
    GEN_CONFIG["max_new_tokens"] = args.max_new_tokens

    model, tokenizer = load_model(args.adapter_path)

    if args.instruction:
        run_single(model, tokenizer, args.instruction)
    else:
        run_batch_test(model, tokenizer)


if __name__ == "__main__":
    main()
