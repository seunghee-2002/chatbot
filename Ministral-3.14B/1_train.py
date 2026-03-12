"""
1_train.py
Mistral-Small-3.1-14B LoRA 파인튜닝 스크립트
판타지 NPC 인사말 생성 도메인 특화 학습

사용법:
    python 1_train.py
    python 1_train.py --data_path custom_data.jsonl --epochs 5
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import unsloth
import json
import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DEFAULT_CONFIG = {
    # 모델
    "model_name": "unsloth/Ministral-3-8B-Instruct-2512-unsloth-bnb-4bit",  # HF Hub ID
    "max_seq_length": 256,
    "load_in_4bit": True,

    # LoRA
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",   # MLP 포함
    ],

    # 학습
    "epochs": 5,
    "batch_size": 4,
    "grad_accum": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "lr_scheduler": "cosine",
    "max_grad_norm": 1.0,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),

    # 경로
    "data_path": "train_dataset.jsonl",
    "output_dir": "./checkpoints",
    "log_path": "./training_log.jsonl",
    "adapter_save_path": "./lora_adapter",
}

SYSTEM_PROMPT = (
    "당신은 판타지 RPG 세계관의 모험가 NPC입니다. "
    "주어진 캐릭터 정보와 상황에 맞는 자연스러운 인사말을 한국어로 생성하세요. "
    "30자 이상 100자 이내로 작성하며, 영어 단어나 설정에 없는 내용은 포함하지 마세요."
)


# ──────────────────────────────────────────────
# 로그 콜백
# ──────────────────────────────────────────────
class JsonlLogCallback(TrainerCallback):
    """학습 지표를 JSONL로 기록 (시각화 스크립트에서 읽음)"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        # 새 학습 시작 시 초기화
        open(log_path, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "timestamp": time.time(),
        }
        if "loss" in logs:
            record["train_loss"] = logs["loss"]
        if "grad_norm" in logs:
            record["grad_norm"] = logs["grad_norm"]
        if "learning_rate" in logs:
            record["learning_rate"] = logs["learning_rate"]
        if "eval_loss" in logs:
            record["eval_loss"] = logs["eval_loss"]

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────
# 데이터 처리
# ──────────────────────────────────────────────
def load_dataset(data_path: str, tokenizer) -> Dataset:
    """JSONL → Alpaca 스타일 chat template 변환"""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)

    def format_sample(example):
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(records)
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)
    return dataset


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Mistral 14B LoRA 파인튜닝")
    parser.add_argument("--data_path",   default=DEFAULT_CONFIG["data_path"])
    parser.add_argument("--model_name",  default=DEFAULT_CONFIG["model_name"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--lora_r",      type=int,   default=DEFAULT_CONFIG["lora_r"])
    parser.add_argument("--output_dir",  default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--adapter_save_path", default=DEFAULT_CONFIG["adapter_save_path"])
    parser.add_argument("--log_path",    default=DEFAULT_CONFIG["log_path"])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {**DEFAULT_CONFIG}
    cfg.update({
        "data_path": args.data_path,
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lora_r": args.lora_r,
        "output_dir": args.output_dir,
        "adapter_save_path": args.adapter_save_path,
        "log_path": args.log_path,
    })

    print(f"\n{'='*55}")
    print(f"  Mistral-Small-3.1-14B LoRA 파인튜닝")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*55}")
    print(f"  모델  : {cfg['model_name']}")
    print(f"  데이터: {cfg['data_path']}")
    print(f"  LoRA r/alpha: {cfg['lora_r']}/{cfg['lora_r']*2}")
    print(f"  Epoch : {cfg['epochs']} | LR: {cfg['learning_rate']}")
    print(f"{'='*55}\n")

    # 1. 모델 로드
    print("[1/4] 모델 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=cfg["load_in_4bit"],
        attn_implementation="eager",
    )

    # 2. LoRA 설정
    print("[2/4] LoRA 어댑터 부착 중...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. 데이터셋
    print("[3/4] 데이터셋 로드 중...")
    dataset = load_dataset(cfg["data_path"], tokenizer)
    print(f"      총 {len(dataset)}개 샘플 로드 완료")

    # 4. 학습
    print("[4/4] 학습 시작...\n")
    steps_per_epoch = math.ceil(
        len(dataset) / (cfg["batch_size"] * cfg["grad_accum"])
    )
    total_steps = steps_per_epoch * cfg["epochs"]
    logging_steps = max(1, steps_per_epoch // 5)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=max(1, int(total_steps * cfg["warmup_ratio"])),
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler"],
        max_grad_norm=cfg["max_grad_norm"],
        fp16=cfg["fp16"],
        bf16=cfg["bf16"],
        logging_steps=logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        args=training_args,
        callbacks=[JsonlLogCallback(cfg["log_path"])],
    )

    trainer.train()

    # 어댑터 저장
    print(f"\n[완료] LoRA 어댑터 저장 중: {cfg['adapter_save_path']}")
    model.save_pretrained(cfg["adapter_save_path"])
    tokenizer.save_pretrained(cfg["adapter_save_path"])

    print(f"\n{'='*55}")
    print(f"  학습 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  어댑터 저장 경로: {cfg['adapter_save_path']}")
    print(f"  학습 로그:        {cfg['log_path']}")
    print(f"{'='*55}")
    print("\n다음 단계:")
    print("  시각화:    python 2_visualize.py")
    print("  추론:      python 3_inference.py")
    print("  GGUF 변환: python 4_convert_gguf.py")


if __name__ == "__main__":
    main()
