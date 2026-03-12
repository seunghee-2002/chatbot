"""
1_train.py — Llama-3.1/3.3-8B-Instruct LoRA 파인튜닝

[사전 설치]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft trl accelerate bitsandbytes

[실행]
python 1_train.py

학습 완료 후 → python 4_visualize.py 로 결과 시각화
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer

# ────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────

HF_TOKEN = "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"  # Hugging Face 토큰 (필요 시 입력)

@dataclass
class Config:
    # 모델: 둘 중 하나 선택
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

    dataset_path: str = "train_dataset.jsonl"
    output_dir: str = "./lora_output"

    # 시각화용 로그 저장 경로
    train_log_path: str = "./lora_output/train_log.json"

    # LoRA 하이퍼파라미터 (보고서 3차 학습 기준)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # 어텐션 + MLP 레이어 동시 학습 (보고서에서 효과 확인)
    target_modules: list = None

    # 학습 하이퍼파라미터
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 256

    # 4bit 양자화 (VRAM 절약)
    use_4bit: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",   # 어텐션
                "gate_proj", "up_proj", "down_proj",        # MLP
            ]


cfg = Config()

# ────────────────────────────────────────────────
# 데이터 로드 및 포맷
# ────────────────────────────────────────────────

def load_dataset(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    print(f"데이터 로드 완료: {len(records)}개")
    return Dataset.from_list(records)


def format_prompt(example: dict) -> dict:
    """
    Llama-3 Instruct 포맷으로 변환
    """
    system_msg = (
        "당신은 판타지 RPG 게임 속 모험가 NPC입니다. "
        "주어진 조건(성격, 직업, 등급, 방문 이력, 아이템, 모험 결과)에 맞춰 "
        "자연스러운 한국어 인사말을 한 문장으로 생성하세요. "
        "영어 단어, 몬스터명, 지역명은 사용하지 마세요. "
        "30자 이상 100자 이하로 작성하세요."
    )
    text = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>"
    )
    return {"text": text}


# ────────────────────────────────────────────────
# 모델 및 토크나이저 로드
# ────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: Config):
    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token = HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )
    model.config.use_cache = False

    return model, tokenizer


# ────────────────────────────────────────────────
# 학습 로그 콜백 (4_visualize.py 용)
# ────────────────────────────────────────────────

class TrainLogCallback(TrainerCallback):
    """
    step마다 loss / grad_norm / learning_rate 를 기록해
    train_log.json으로 저장한다.
    4_visualize.py가 이 파일을 읽어 시각화한다.
    """

    def __init__(self, log_path: str, cfg: Config):
        self.log_path = log_path
        self.cfg = cfg
        self.step_logs: list[dict] = []
        self.start_time = time.time()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        # train loss가 있는 step만 기록
        if "loss" not in logs:
            return
        entry = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0, 4),
            "loss": logs.get("loss"),
            "grad_norm": logs.get("grad_norm"),
            "learning_rate": logs.get("learning_rate"),
            "elapsed_sec": round(time.time() - self.start_time, 1),
        }
        self.step_logs.append(entry)
        self._flush()

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # 학습 설정 메타 정보 함께 저장
        meta = {
            "model_id": self.cfg.model_id,
            "dataset_path": self.cfg.dataset_path,
            "num_train_epochs": self.cfg.num_train_epochs,
            "lora_r": self.cfg.lora_r,
            "lora_alpha": self.cfg.lora_alpha,
            "lora_dropout": self.cfg.lora_dropout,
            "learning_rate": self.cfg.learning_rate,
            "batch_size": self.cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "target_modules": self.cfg.target_modules,
            "total_steps": state.global_step,
            "total_elapsed_sec": round(time.time() - self.start_time, 1),
        }
        payload = {"meta": meta, "steps": self.step_logs}
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"학습 로그 저장 완료: {self.log_path}")

    def _flush(self):
        """학습 중 중단돼도 로그 유실 방지"""
        payload = {"meta": {}, "steps": self.step_logs}
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


# ────────────────────────────────────────────────
# LoRA 적용
# ────────────────────────────────────────────────
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

def apply_lora(model, cfg: Config):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ────────────────────────────────────────────────
# 학습
# ────────────────────────────────────────────────

def train(cfg: Config):
    # 데이터
    raw_dataset = load_dataset(cfg.dataset_path)
    dataset = raw_dataset.map(format_prompt, remove_columns=raw_dataset.column_names)

    # 모델
    model, tokenizer = load_model_and_tokenizer(cfg)
    model = apply_lora(model, cfg)

    # ── 추가: 텍스트 → 토큰 변환 ──────────────────────
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    dataset = dataset.map(tokenize, remove_columns=["text"])
    # ───────────────────────────────────────────────────

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    log_callback = TrainLogCallback(log_path=cfg.train_log_path, cfg=cfg)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        # dataset_text_field 제거
        # max_seq_length 제거
        args=training_args,
        callbacks=[log_callback],
    )

    print("학습 시작...")
    trainer.train()

    # LoRA 어댑터 저장
    adapter_path = os.path.join(cfg.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"LoRA 어댑터 저장 완료: {adapter_path}")

    # LoRA 머지 후 풀 모델 저장 (GGUF 변환 전 필요)
    merged_path = os.path.join(cfg.output_dir, "merged_model")
    print("LoRA 머지 중...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print(f"머지 모델 저장 완료: {merged_path}")
    print(f"학습 로그: {cfg.train_log_path}")
    print("다음 단계 1: python 4_visualize.py  (학습 결과 시각화)")
    print("다음 단계 2: ./3_convert_gguf.sh    (Ollama 배포)")


if __name__ == "__main__":
    train(cfg)
