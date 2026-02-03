from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset('json', data_files='train_dataset.jsonl', split='train')

# LoRA 설정: 적은 메모리로 학습 가능하게 함
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["query_key_value", "dense"], # EXAONE 구조에 맞춤
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# 일반 PC(VRAM 8GB~12GB)를 위한 8bit 양자화 로드 권장
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    load_in_8bit=True, # 메모리 절약
    device_map="auto"
)

training_args = TrainingArguments(
    output_dir="./exaone-greeting-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,
    fp16=True, # NVIDIA GPU 가속
    logging_steps=10,
    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="instruction", # 위에서 만든 필드 사용
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()