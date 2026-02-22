import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 경로 설정
base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
lora_model_path = "./qwen-greeting-lora/checkpoint-248" 
output_dir = "./qwen-greeting-final"

print("GPU를 사용하여 모델 병합을 시작합니다...")

# 2. 모델 및 토크나이저 로드 (GPU 사용 설정)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",  # GPU가 있으면 자동으로 할당
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# 3. LoRA 어댑터 병합
model = PeftModel.from_pretrained(base_model, lora_model_path)
merged_model = model.merge_and_unload()

# 4. 최종 모델 저장
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"병합 완료! '{output_dir}' 폴더를 확인하세요.")