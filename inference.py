from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_model(base_model_id, lora_adapter_path):
    """베이스 모델 + LoRA 어댑터 로드"""
    print("모델을 로드합니다...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    
    return model, tokenizer

def generate_greeting(model, tokenizer, 
                     성격="평범형", 모험가타입="전사", 나이="20대", 
                     성별="남성", 모험가등급="C급", 방문횟수="첫방문",
                     이전_아이템="없음", 재방문간격="없음", 최근_의뢰="첫방문"):
    """주어진 속성으로 인사말 생성"""
    
    # 입력 프롬프트 구성
    instruction = (
        f"성격: {성격}, 모험가타입: {모험가타입}, 나이: {나이}, "
        f"성별: {성별}, 모험가등급: {모험가등급}, 방문횟수: {방문횟수}, "
        f"이전_아이템: {이전_아이템}, 재방문간격: {재방문간격}, "
        f"최근_의뢰: {최근_의뢰} "
        "일 때의 적절한 모험가 인사말을 생성해줘."
    )
    
    messages = [
        {"role": "system", "content": "당신은 판타지 세계관의 무기 대여점을 방문하는 모험가의 인사말을 생성하는 전문가입니다."},
        {"role": "user", "content": instruction}
    ]
    
    # 토크나이저로 변환
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 디코딩
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # assistant 응답만 추출
    if "[|assistant|]" in generated_text:
        greeting = generated_text.split("[|assistant|]")[-1].strip()
    else:
        greeting = generated_text.strip()
    
    return greeting

if __name__ == "__main__":
    # 모델 로드
    base_model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    lora_adapter_path = "./exaone-greeting-lora/checkpoint-510"  # 실제 체크포인트 경로로 수정
    
    model, tokenizer = load_model(base_model_id, lora_adapter_path)
    
    print("\n" + "="*60)
    print("모험가 인사말 생성 테스트")
    print("="*60 + "\n")
    
    # 테스트 케이스 1: 첫 방문 전사
    print("[테스트 1] 첫 방문 전사")
    greeting = generate_greeting(
        model, tokenizer,
        성격="존칭형", 모험가타입="전사", 나이="20대",
        성별="남성", 모험가등급="D급", 방문횟수="첫방문",
        이전_아이템="없음", 재방문간격="없음", 최근_의뢰="첫방문"
    )
    print(f"생성된 인사말: {greeting}\n")
    
    # 테스트 케이스 2: 단골 궁수 (대성공)
    print("[테스트 2] 단골 궁수 (활로 대성공)")
    greeting = generate_greeting(
        model, tokenizer,
        성격="너스레형", 모험가타입="궁수", 나이="30대",
        성별="여성", 모험가등급="A급", 방문횟수="단골",
        이전_아이템="활", 재방문간격="최근", 최근_의뢰="대성공"
    )
    print(f"생성된 인사말: {greeting}\n")
    
    # 테스트 케이스 3: 야성형 전사 (실패 후 재방문)
    print("[테스트 3] 야성형 전사 (도끼로 실패)")
    greeting = generate_greeting(
        model, tokenizer,
        성격="야성형", 모험가타입="전사", 나이="40대",
        성별="남성", 모험가등급="B급", 방문횟수="보통",
        이전_아이템="도끼", 재방문간격="보통", 최근_의뢰="실패"
    )
    print(f"생성된 인사말: {greeting}\n")
    
    print("="*60)
    print("테스트 완료")
    print("="*60)