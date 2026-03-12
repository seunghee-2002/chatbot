"""
2_infer.py — 파인튜닝 모델 추론 테스트

[사용법]
# LoRA 어댑터로 추론 (학습 직후 확인용)
python 2_infer.py --mode adapter --model_path ./lora_output/final_adapter --base_model meta-llama/Llama-3.1-8B-Instruct

# 머지 모델로 추론
python 2_infer.py --mode merged --model_path ./lora_output/merged_model

# Ollama로 추론 (3_convert_gguf.sh 실행 후)
python 2_infer.py --mode ollama --ollama_model npc-greeter
"""

import argparse
import json
import sys

# ────────────────────────────────────────────────
# 테스트케이스 (보고서 최종 추론 결과 기준)
# ────────────────────────────────────────────────

TEST_CASES = [
    {
        "label": "존칭형 도적 / 첫방문 / D급",
        "instruction": "성격: 존칭형, 모험가타입: 도적, 나이: 20대, 성별: 남성, 모험가등급: D급, 방문횟수: 첫방문, 이전_아이템: 없음, 재방문간격: 없음, 최근_의뢰: 첫방문 일 때의 적절한 모험가 인사말을 생성해줘.",
        "expected_keywords": ["처음"],
    },
    {
        "label": "야성형 전사 / 단골 / 도끼 / 대성공 / S급",
        "instruction": "성격: 야성형, 모험가타입: 전사, 나이: 30대, 성별: 남성, 모험가등급: S급, 방문횟수: 단골, 이전_아이템: 도끼, 재방문간격: 최근, 최근_의뢰: 대성공 일 때의 적절한 모험가 인사말을 생성해줘.",
        "expected_keywords": ["도끼"],
    },
    {
        "label": "하대형 전사 / 오랜만 / 검 / 성공",
        "instruction": "성격: 하대형, 모험가타입: 전사, 나이: 30대, 성별: 남성, 모험가등급: A급, 방문횟수: 보통, 이전_아이템: 검, 재방문간격: 오래됨, 최근_의뢰: 성공 일 때의 적절한 모험가 인사말을 생성해줘.",
        "expected_keywords": ["오랜만", "검"],
    },
    {
        "label": "단답형 도적 / 단검 / 대성공 ★비매칭 아님",
        "instruction": "성격: 단답형, 모험가타입: 도적, 나이: 20대, 성별: 여성, 모험가등급: A급, 방문횟수: 많음, 이전_아이템: 단검, 재방문간격: 보통, 최근_의뢰: 대성공 일 때의 적절한 모험가 인사말을 생성해줘.",
        "expected_keywords": ["단검"],
    },
    {
        "label": "너스레형 궁수 / 단검 / 성공 ★비매칭",
        "instruction": "성격: 너스레형, 모험가타입: 궁수, 나이: 30대, 성별: 여성, 모험가등급: A급, 방문횟수: 많음, 이전_아이템: 단검, 재방문간격: 오래됨, 최근_의뢰: 성공 일 때의 적절한 모험가 인사말을 생성해줘.",
        "expected_keywords": ["단검"],
    },
    {
        "label": "존칭형 마법사 / 망치 / 실패 ★비매칭",
        "instruction": "성격: 존칭형, 모험가타입: 마법사, 나이: 20대, 성별: 여성, 모험가등급: B급, 방문횟수: 보통, 이전_아이템: 망치, 재방문간격: 보통, 최근_의뢰: 실패 일 때의 적절한 모험가 인사말을 생성해줘.",
        "expected_keywords": ["망치"],
    },
]

SYSTEM_MSG = (
    "당신은 판타지 RPG 게임 속 모험가 NPC입니다. "
    "주어진 조건(성격, 직업, 등급, 방문 이력, 아이템, 모험 결과)에 맞춰 "
    "자연스러운 한국어 인사말을 한 문장으로 생성하세요. "
    "영어 단어, 몬스터명, 지역명은 사용하지 마세요. "
    "30자 이상 100자 이하로 작성하세요."
)


# ────────────────────────────────────────────────
# 추론 모드별 함수
# ────────────────────────────────────────────────

def infer_with_adapter(model_path: str, base_model: str, cases: list):
    """LoRA 어댑터 + 베이스 모델로 추론"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"베이스 모델 로드: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"LoRA 어댑터 로드: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return _run_hf_inference(model, tokenizer, cases)


def infer_with_merged(model_path: str, cases: list):
    """머지 모델로 추론"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"머지 모델 로드: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return _run_hf_inference(model, tokenizer, cases)


def _run_hf_inference(model, tokenizer, cases: list):
    """HuggingFace 모델 공통 추론 루프"""
    import torch

    results = []
    for case in cases:
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_MSG}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{case['instruction']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        results.append((case, generated))
    return results


def infer_with_ollama(ollama_model: str, cases: list):
    """Ollama API로 추론"""
    import urllib.request

    results = []
    for case in cases:
        payload = json.dumps({
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": case["instruction"]},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 60,
                "repeat_penalty": 1.1,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                generated = data["message"]["content"].strip()
        except Exception as e:
            generated = f"[오류] {e}"
        results.append((case, generated))
    return results


# ────────────────────────────────────────────────
# 결과 출력 및 검증
# ────────────────────────────────────────────────

def evaluate(results: list):
    print("\n" + "=" * 60)
    print("추론 결과")
    print("=" * 60)
    pass_count = 0
    for case, generated in results:
        length = len(generated)
        keyword_ok = all(kw in generated for kw in case.get("expected_keywords", []))
        length_ok = 30 <= length <= 100

        status = "✅" if (keyword_ok and length_ok) else "⚠️"
        if keyword_ok and length_ok:
            pass_count += 1

        print(f"\n[{status}] {case['label']}")
        print(f"  출력 : {generated}")
        print(f"  길이 : {length}자 {'(OK)' if length_ok else '(범위 초과)'}")
        if case.get("expected_keywords"):
            print(f"  키워드 체크: {case['expected_keywords']} → {'OK' if keyword_ok else 'FAIL'}")

    print(f"\n통과: {pass_count}/{len(results)}")
    print("=" * 60)


# ────────────────────────────────────────────────
# 단일 인터랙티브 추론
# ────────────────────────────────────────────────

def interactive_infer(infer_fn, **kwargs):
    """커스텀 파라미터 입력 추론"""
    print("\n[인터랙티브 추론] 조건을 입력하면 인사말을 생성합니다.")
    print("종료: Ctrl+C\n")
    params = [
        ("성격", ["평범형", "존칭형", "하대형", "단답형", "너스레형", "야성형"]),
        ("모험가타입", ["전사", "궁수", "마법사", "도적"]),
        ("나이", ["10대", "20대", "30대", "40대", "50대 이상"]),
        ("성별", ["남성", "여성"]),
        ("모험가등급", ["S급", "A급", "B급", "C급", "D급"]),
        ("방문횟수", ["첫방문", "적음", "보통", "많음", "단골"]),
        ("이전_아이템", ["없음", "검", "도끼", "창", "단검", "활", "석궁", "망치", "몽둥이", "지팡이", "마법서"]),
        ("재방문간격", ["없음", "최근", "보통", "오래됨"]),
        ("최근_의뢰", ["첫방문", "실패", "성공", "대성공"]),
    ]
    try:
        while True:
            values = {}
            for key, options in params:
                print(f"{key} 옵션: {options}")
                val = input(f"  {key}: ").strip()
                values[key] = val if val else options[0]

            instruction = (
                f"성격: {values['성격']}, 모험가타입: {values['모험가타입']}, "
                f"나이: {values['나이']}, 성별: {values['성별']}, "
                f"모험가등급: {values['모험가등급']}, 방문횟수: {values['방문횟수']}, "
                f"이전_아이템: {values['이전_아이템']}, 재방문간격: {values['재방문간격']}, "
                f"최근_의뢰: {values['최근_의뢰']} 일 때의 적절한 모험가 인사말을 생성해줘."
            )
            case = {"label": "커스텀", "instruction": instruction, "expected_keywords": []}
            results = infer_fn([case], **kwargs)
            print(f"\n생성된 인사말: {results[0][1]}\n")
    except KeyboardInterrupt:
        print("\n종료")


# ────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NPC 인사말 추론 테스트")
    parser.add_argument(
        "--mode",
        choices=["adapter", "merged", "ollama"],
        default="merged",
        help="추론 모드 (default: merged)",
    )
    parser.add_argument("--model_path", default="./lora_output/final_adapter")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ollama_model", default="npc-greeter")
    parser.add_argument("--interactive", action="store_true", help="인터랙티브 추론 모드")
    args = parser.parse_args()

    if args.mode == "adapter":
        results = infer_with_adapter(args.model_path, args.base_model, TEST_CASES)
        evaluate(results)
        if args.interactive:
            interactive_infer(
                lambda cases, **kw: infer_with_adapter(args.model_path, args.base_model, cases),
            )
    elif args.mode == "merged":
        results = infer_with_merged(args.model_path, TEST_CASES)
        evaluate(results)
        if args.interactive:
            interactive_infer(lambda cases, **kw: infer_with_merged(args.model_path, cases))
    elif args.mode == "ollama":
        results = infer_with_ollama(args.ollama_model, TEST_CASES)
        evaluate(results)
        if args.interactive:
            interactive_infer(
                lambda cases, **kw: infer_with_ollama(args.ollama_model, cases)
            )


if __name__ == "__main__":
    main()
