from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# 토크나이저 로드 및 설정
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 로드
dataset = load_dataset('json', data_files='train_dataset.jsonl', split='train')

# EXAONE 공식 chat template 적용 함수
def formatting_prompts_func(example):
    """EXAONE 공식 apply_chat_template 방식 사용"""
    output_texts = []
    for i in range(len(example['instruction'])):
        messages = [
            {"role": "system", "content": "당신은 게임 속 NPC 대화를 생성하는 AI입니다."},
            {"role": "user", "content": example['instruction'][i]},
            {"role": "assistant", "content": example['output'][i]}
        ]
        # apply_chat_template 사용 (add_generation_prompt=False로 assistant 답변 포함)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

# LoRA 설정 (EXAONE 모델 구조 반영)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # EXAONE 구조
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Windows용: 8bit 양자화 없이 로드 (VRAM 12GB 이상 필요)
print("모델을 로드합니다... (VRAM 사용량 높음, 약 8-10GB)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # FP16으로 메모리 절약
    device_map="auto",
    trust_remote_code=True  # EXAONE 커스텀 코드 허용
)
model.enable_input_require_grads() # 학습을 위해 입력 임베딩 고정
model = get_peft_model(model, peft_config)

for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# 학습 설정
training_args = TrainingArguments(
    output_dir="./exaone-greeting-lora",
    per_device_train_batch_size=1,  # VRAM 부족 시 1로 감소
    gradient_accumulation_steps=8,  # batch_size=1일 때 보정
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,  # NVIDIA GPU 가속
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",  # wandb 비활성화
    gradient_checkpointing=True,  # 메모리 절약 (속도 약간 감소)
    optim="adamw_torch"  # Windows 호환 옵티마이저
)

# Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=False
)

print("학습을 시작합니다...")
print(f"총 데이터: {len(dataset)}개")
print(f"예상 학습 시간: RTX 3060 기준 약 40-60분")
trainer.train()
print("학습 완료!")