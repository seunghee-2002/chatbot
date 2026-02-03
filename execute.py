import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 경로 및 모델 설정
base_model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
lora_model_path = "./exaone-greeting-lora/checkpoint-141"  # 학습 결과가 저장된 폴더

print("모델을 불러오는 중입니다... (VRAM 약 6-8GB 필요)")

# 2. 토크나이저 및 베이스 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 3. LoRA 어댑터 결합
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval() # 추론 모드로 설정

def generate_greeting(adventurer_info):
    """
    입력: "타입: 전사, 성격: 존칭형, 호감도: 매우높음, 방문: 단골, 이전결과: 대성공, 상성: 좋음"
    """
    prompt = f"{adventurer_info} 일 때의 적절한 모험가 인사말을 생성해줘."
    
    # EXAONE 공식 프롬프트 템플릿 적용
    messages = [
        {"role": "system", "content": "당신은 게임 속 NPC 대화를 생성하는 AI입니다."},
        {"role": "user", "content": prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    # 답변 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.3,
            top_p=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 생성된 텍스트만 추출 및 출력
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# --- 테스트 실행 ---
if __name__ == "__main__":
    test_cases = [
        "타입: 전사, 성격: 존칭형, 호감도: 매우높음, 방문: 단골, 이전결과: 대성공, 상성: 좋음",
        "타입: 도적, 성격: 하대형, 호감도: 매우낮음, 방문: 초면, 이전결과: 실패, 상성: 나쁨",
        "타입: 마법사, 성격: 단답형, 호감도: 낮음, 방문: 아는사이, 이전결과: 성공, 상성: 보통"
    ]

    print("\n" + "="*50)
    print("모험가 인사말 테스트 결과")
    print("="*50)
    
    for case in test_cases:
        result = generate_greeting(case)
        print(f"\n입력 조건: {case}")
        print(f"모험가 답변: {result}")