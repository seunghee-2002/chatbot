# remerge.py — 어댑터 재머지 (1_train.py 재실행 없이)
import sys
sys.modules["torchvision"] = None

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL  = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./lora_output/final_adapter"
MERGED_PATH  = "./lora_output/merged_model"
HF_TOKEN     = "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"

print("베이스 모델 로드 중... (float16, CPU)")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",          # CPU로 로드 (양자화 없이)
    token=HF_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("LoRA 어댑터 로드 중...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("머지 중...")
model = model.merge_and_unload()

print("저장 중...")
model.save_pretrained(MERGED_PATH, safe_serialization=True)
tokenizer.save_pretrained(MERGED_PATH)
print(f"완료: {MERGED_PATH}")
print("이제 3_convert_gguf.py 실행하세요.")