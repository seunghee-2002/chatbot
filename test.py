# -*- coding: utf-8 -*-
"""
학습된 Qwen3-8B LoRA 모델 추론 테스트
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

LORA_DIR = "./qwen3-greeting-lora/checkpoint-310"  # LoRA 모델이 저장된 디렉토리
MAX_NEW_TOKENS = 100

print("모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
model = AutoModelForCausalLM.from_pretrained(
    LORA_DIR,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()
print("로딩 완료!\n")

# 학습 데이터와 동일한 프롬프트 형식
def make_prompt(성격, 모험가타입, 나이, 성별, 모험가등급, 방문횟수, 이전_아이템, 재방문간격, 최근_의뢰):
    instruction = (
        f"성격: {성격}, 모험가타입: {모험가타입}, 나이: {나이}, 성별: {성별}, "
        f"모험가등급: {모험가등급}, 방문횟수: {방문횟수}, 이전_아이템: {이전_아이템}, "
        f"재방문간격: {재방문간격}, 최근_의뢰: {최근_의뢰} 일 때의 적절한 모험가 인사말을 생성해줘."
    )
    return instruction

test_cases = [
    {
        "설명": "야성형 전사, 단골, 대성공",
        "params": dict(성격="야성형", 모험가타입="전사", 나이="30대", 성별="남성",
                       모험가등급="A급", 방문횟수="단골", 이전_아이템="도끼",
                       재방문간격="최근", 최근_의뢰="대성공")
    },
    {
        "설명": "존칭형 마법사, 첫 방문, 성공",
        "params": dict(성격="존칭형", 모험가타입="마법사", 나이="20대", 성별="여성",
                       모험가등급="C급", 방문횟수="적음", 이전_아이템="지팡이",
                       재방문간격="보통", 최근_의뢰="성공")
    },
    {
        "설명": "하대형 도적, 보통 방문, 실패",
        "params": dict(성격="하대형", 모험가타입="도적", 나이="20대", 성별="남성",
                       모험가등급="B급", 방문횟수="보통", 이전_아이템="단검",
                       재방문간격="최근", 최근_의뢰="실패")
    },
    {
        "설명": "냉담형 궁수, 오랜만, 성공",
        "params": dict(성격="냉담형", 모험가타입="궁수", 나이="40대", 성별="여성",
                       모험가등급="S급", 방문횟수="단골", 이전_아이템="활",
                       재방문간격="오래됨", 최근_의뢰="성공")
    },
]

def generate(prompt):
    messages = [
        {
            "role": "system",
            "content": "당신은 판타지 세계관의 무기 대여점을 방문하는 모험가입니다. 상황에 맞는 인사말을 하세요."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return result


for i, case in enumerate(test_cases, 1):
    print(f"{'='*50}")
    print(f"[테스트 {i}] {case['설명']}")
    prompt = make_prompt(**case['params'])
    print(f"프롬프트: {prompt}")
    print(f"\n결과: ", end="", flush=True)
    result = generate(prompt)
    print(result)
    print()