import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ─────────────────────────────────────────────
# [설정]
# ─────────────────────────────────────────────
MODEL_ID     = "unsloth/Qwen3-8B"
DATASET_PATH = "./train_dataset.jsonl"
OUTPUT_DIR   = "./qwen3-greeting-lora"
GGUF_DIR     = "./qwen3-greeting-gguf"

# ─────────────────────────────────────────────
# 1. 모델 & 토크나이저 로드
# ─────────────────────────────────────────────
print("[1/5] 모델을 로드합니다...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=512,
    load_in_4bit=True,
    dtype=None,           # bfloat16 자동 선택
)

# ─────────────────────────────────────────────
# 2. LoRA 적용
# ─────────────────────────────────────────────
print("[2/5] LoRA를 적용합니다...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  학습 가능 파라미터: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

# ─────────────────────────────────────────────
# 3. 데이터셋 로드 및 포맷팅
# ─────────────────────────────────────────────
print("[3/5] 데이터셋을 로드합니다...")
raw_dataset   = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split["train"]
eval_dataset  = dataset_split["test"]
print(f"  학습: {len(train_dataset)}개 / 검증: {len(eval_dataset)}개")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        messages = [
            {
                "role": "system",
                "content": "당신은 판타지 세계관의 무기 대여점을 방문하는 모험가입니다. 상황에 맞는 인사말을 하세요."
            },
            {"role": "user",      "content": example["instruction"][i]},
            {"role": "assistant", "content": example["output"][i]}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False  # 인사말 태스크에 thinking 불필요
        )
        output_texts.append(text)
    return output_texts

# ─────────────────────────────────────────────
# 4. 학습
# ─────────────────────────────────────────────
print("[4/5] 학습을 시작합니다...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=5,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_eval_batch_size=1,
        save_total_limit=2,
        report_to="none",
        optim="adamw_8bit",
        max_seq_length=512,
        dataset_text_field=None,
        packing=False,
    ),
)

trainer.train()
print(f"  LoRA 어댑터 저장 위치: {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# 5. GGUF 변환 (Q4_K_M)
# ─────────────────────────────────────────────
print("[5/5] GGUF 변환을 시작합니다 (시간이 다소 소요됩니다)...")
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method="q4_k_m"
)

print(f"\n{'='*50}")
print("완료!")
print(f"GGUF 파일 위치: {GGUF_DIR}/")
print(f"{'='*50}")
print("\nOllama 등록 방법:")
print(f"  1. Modelfile 작성:")
print(f'       FROM ./{GGUF_DIR}/*.gguf')
print(f'       PARAMETER stop "<|im_end|>"')
print(f"  2. 등록: ollama create qwen3-greeting -f Modelfile")
print(f"  3. 실행: ollama run qwen3-greeting")