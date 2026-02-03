import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 경로 설정
base_model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
lora_model_path = "./exaone-greeting-lora"  # train.py에서 설정한 output_dir
output_dir = "./exaone-greeting-final"

print("모델 병합을 시작합니다...")

# 2. 모델 및 토크나이저 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu",  # 병합은 메모리 확보를 위해 CPU에서 진행 권장
    trust_remote_code=True  # EXAONE 커스텀 코드 허용
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# 3. LoRA 어댑터 병합
model = PeftModel.from_pretrained(base_model, lora_model_path)
merged_model = model.merge_and_unload()

# 4. 최종 모델 저장
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"병합 완료! '{output_dir}' 폴더를 확인하세요.")