"""
2_visualize.py
학습 지표 시각화 스크립트
training_log.jsonl을 읽어 loss / grad_norm / lr 그래프를 저장

사용법:
    python 2_visualize.py
    python 2_visualize.py --log_path custom_log.jsonl --out_dir ./plots
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 한글 폰트 자동 설정
_KR_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleGothic.ttf",           # macOS
    "C:/Windows/Fonts/malgun.ttf",                     # Windows
]
for _fp in _KR_FONT_CANDIDATES:
    import os as _os
    if _os.path.exists(_fp):
        _fm.fontManager.addfont(_fp)
        _prop = _fm.FontProperties(fname=_fp)
        matplotlib.rc("font", family=_prop.get_name())
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DEFAULT_LOG_PATH = "./training_log.jsonl"
DEFAULT_OUT_DIR  = "./training_plots"

STYLE = {
    "train_loss":    {"color": "#2563EB", "label": "Train Loss",    "lw": 2.0},
    "eval_loss":     {"color": "#DC2626", "label": "Eval Loss",     "lw": 1.8, "ls": "--"},
    "grad_norm":     {"color": "#7C3AED", "label": "Grad Norm",     "lw": 1.8},
    "learning_rate": {"color": "#059669", "label": "Learning Rate", "lw": 1.8},
}


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
def load_log(log_path: str) -> list[dict]:
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"로그 파일이 비어 있습니다: {log_path}")
    return records


def extract_series(records: list[dict], key: str):
    """key가 존재하는 레코드에서 (epoch, value) 시리즈 추출"""
    xs, ys = [], []
    for r in records:
        if key in r and r[key] is not None:
            xs.append(r.get("epoch", r.get("step", 0)))
            ys.append(r[key])
    return xs, ys


# ──────────────────────────────────────────────
# 플롯 생성
# ──────────────────────────────────────────────
def plot_loss(records: list[dict], out_dir: Path):
    """Train Loss (+ Eval Loss if available) 그래프"""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    tx, ty = extract_series(records, "train_loss")
    ex, ey = extract_series(records, "eval_loss")

    if not tx:
        print("  [경고] train_loss 데이터가 없습니다. 스킵.")
        plt.close(fig)
        return

    s = STYLE["train_loss"]
    ax.plot(tx, ty, color=s["color"], lw=s["lw"], label=s["label"])

    if ex:
        s = STYLE["eval_loss"]
        ax.plot(ex, ey, color=s["color"], lw=s["lw"],
                ls=s.get("ls", "-"), label=s["label"])

    # 최솟값 마커
    min_idx = ty.index(min(ty))
    ax.scatter(tx[min_idx], ty[min_idx], color=STYLE["train_loss"]["color"],
               zorder=5, s=60, label=f"최솟값 {ty[min_idx]:.4f}")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("학습 손실 (Loss)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    fig.tight_layout()

    path = out_dir / "loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_grad_norm(records: list[dict], out_dir: Path):
    """Gradient Norm 그래프"""
    xs, ys = extract_series(records, "grad_norm")
    if not xs:
        print("  [경고] grad_norm 데이터가 없습니다. 스킵.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    s = STYLE["grad_norm"]
    ax.plot(xs, ys, color=s["color"], lw=s["lw"], label=s["label"])
    ax.fill_between(xs, ys, alpha=0.10, color=s["color"])

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Grad Norm", fontsize=11)
    ax.set_title("Gradient Norm", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "grad_norm.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_lr(records: list[dict], out_dir: Path):
    """Learning Rate 스케줄 그래프"""
    xs, ys = extract_series(records, "learning_rate")
    if not xs:
        print("  [경고] learning_rate 데이터가 없습니다. 스킵.")
        return

    fig, ax = plt.subplots(figsize=(8, 3.5))
    s = STYLE["learning_rate"]
    ax.plot(xs, ys, color=s["color"], lw=s["lw"], label=s["label"])

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("Learning Rate 스케줄", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    fig.tight_layout()

    path = out_dir / "learning_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path}")


def plot_overview(records: list[dict], out_dir: Path):
    """3개 지표를 한 화면에 (overview)"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("학습 과정 요약", fontsize=14, fontweight="bold", y=1.01)

    panels = [
        ("train_loss",    axes[0], "Loss",          "%.4f"),
        ("grad_norm",     axes[1], "Grad Norm",      "%.2f"),
        ("learning_rate", axes[2], "Learning Rate",  None),
    ]

    for key, ax, ylabel, fmt in panels:
        xs, ys = extract_series(records, key)
        if not xs:
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(ylabel)
            continue

        s = STYLE[key]
        ax.plot(xs, ys, color=s["color"], lw=s["lw"], label=s["label"])
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(s["label"], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if fmt:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
        else:
            ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # eval_loss 오버레이
        if key == "train_loss":
            ex, ey = extract_series(records, "eval_loss")
            if ex:
                se = STYLE["eval_loss"]
                ax.plot(ex, ey, color=se["color"], lw=se["lw"],
                        ls=se.get("ls", "-"), label=se["label"])
            ax.legend(fontsize=9)

    fig.tight_layout()
    path = out_dir / "overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


# ──────────────────────────────────────────────
# 요약 출력
# ──────────────────────────────────────────────
def print_summary(records: list[dict]):
    _, losses = extract_series(records, "train_loss")
    _, gnorms = extract_series(records, "grad_norm")

    print("\n  ── 학습 요약 ─────────────────────────────")
    print(f"  로그 레코드 수 : {len(records)}")
    if losses:
        print(f"  초기 Train Loss: {losses[0]:.4f}")
        print(f"  최종 Train Loss: {losses[-1]:.4f}")
        print(f"  최솟 Train Loss: {min(losses):.4f}")
    if gnorms:
        print(f"  평균 Grad Norm : {sum(gnorms)/len(gnorms):.2f}")
    print("  ──────────────────────────────────────────\n")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="학습 지표 시각화")
    parser.add_argument("--log_path", default=DEFAULT_LOG_PATH)
    parser.add_argument("--out_dir",  default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = args.log_path
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[시각화] 로그 파일: {log_path}")
    records = load_log(log_path)
    print(f"         {len(records)}개 레코드 로드 완료\n")

    print("[그래프 생성]")
    plot_loss(records, out_dir)
    plot_grad_norm(records, out_dir)
    plot_lr(records, out_dir)
    plot_overview(records, out_dir)

    print_summary(records)
    print(f"모든 그래프 저장 완료 → {out_dir}/")


if __name__ == "__main__":
    main()
