from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import torch

model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

# 토크나이저 로드 및 설정
print("토크나이저를 로드합니다...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 데이터셋 로드
print("데이터셋을 로드합니다...")
dataset = load_dataset('json', data_files='train_dataset.jsonl', split='train')
print(f"총 {len(dataset)}개의 학습 데이터가 로드되었습니다.")

# EXAONE 공식 chat template 적용 함수
def formatting_prompts_func(example):
    """EXAONE 공식 apply_chat_template 방식 사용"""
    output_texts = []
    for i in range(len(example['instruction'])):
        messages = [
            {"role": "system", "content": "당신은 판타지 세계관의 무기 대여점을 방문하는 모험가의 인사말을 생성하는 전문가입니다."},
            {"role": "user", "content": example['instruction'][i]},
            {"role": "assistant", "content": example['output'][i]}
        ]
        # apply_chat_template 사용
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        output_texts.append(text)
    return output_texts

# LoRA 설정
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", # 어텐션 계층 (말투/톤 학습)
        "gate_proj", "up_proj", "down_proj"      # MLP 계층 (지식/논리 연결 학습)
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 모델 로드 (FP16으로 메모리 절약)
print("모델을 로드합니다... (VRAM 사용량: 약 8-10GB)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 적용
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)

# LoRA 파라미터만 학습 가능하도록 설정
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

print("학습 가능한 파라미터 수:")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
print(f"  학습 가능: {trainable_params:,} / 전체: {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

# 학습 설정
training_args = TrainingArguments(
    output_dir="./exaone-greeting-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=10,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch"
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

print("\n" + "="*50)
print("학습을 시작합니다...")
print(f"총 데이터: {len(dataset)}개")
print(f"에포크: {training_args.num_train_epochs}")
print(f"배치 크기: {training_args.per_device_train_batch_size}")
print(f"그래디언트 누적: {training_args.gradient_accumulation_steps}")
print(f"예상 학습 시간: RTX 3060 기준 약 40-60분")
print("="*50 + "\n")

trainer.train()

print("\n" + "="*50)
print("학습 완료!")
print(f"모델 저장 위치: {training_args.output_dir}")
print("="*50)