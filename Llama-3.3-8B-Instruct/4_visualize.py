"""
4_visualize.py — LoRA 파인튜닝 학습 결과 시각화

[사전 설치]
pip install matplotlib numpy

[실행]
python 4_visualize.py                                      # 기본 경로
python 4_visualize.py --log ./lora_output/train_log.json  # 경로 지정
python 4_visualize.py --demo                               # 실제 로그 없이 샘플 데이터로 미리보기

[출력]
./lora_output/train_result.png  — 4-패널 종합 차트 (PNG)
"""

import argparse
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 한글 폰트 자동 감지 및 등록
def _setup_korean_font():
    import glob
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    candidates += glob.glob("/usr/share/fonts/**/*anum*.ttf", recursive=True)
    candidates += glob.glob("/usr/share/fonts/**/*oto*CJK*.ttc", recursive=True)
    for path in candidates:
        if os.path.exists(path):
            matplotlib.font_manager.fontManager.addfont(path)
            prop = matplotlib.font_manager.FontProperties(fname=path)
            name = prop.get_name()
            matplotlib.rcParams["font.family"] = name
            return name
    matplotlib.rcParams["font.family"] = "sans-serif"
    return "sans-serif"

_FONT_NAME = _setup_korean_font()
matplotlib.rcParams["axes.unicode_minus"] = False

# ────────────────────────────────────────────────────────────────────────────
# 팔레트 / 스타일
# ────────────────────────────────────────────────────────────────────────────

BG        = "#0f1117"
PANEL_BG  = "#181c27"
GRID_C    = "#252a38"
ACCENT1   = "#7EB8F7"   # loss 라인
ACCENT2   = "#F7A85E"   # grad_norm 라인
ACCENT3   = "#7EF7C0"   # lr 라인
EPOCH_C   = "#ffffff"
TEXT_C    = "#c8d0e0"
MUTED_C   = "#5a6070"
TITLE_C   = "#e8edf5"

# ────────────────────────────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────────────────────────────

def load_log(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_demo_data() -> dict:
    """실제 로그 없이 미리보기용 샘플 데이터 생성 (보고서 수치 기반)"""
    total_steps = 170           # 544샘플 / (batch4 * accum4) * 5epoch ≈ 170
    steps_per_epoch = total_steps // 5

    steps, losses, grads, lrs, elapsed = [], [], [], [], []

    # Cosine LR schedule + 노이즈 섞인 loss 곡선
    lr_max = 2e-4
    warmup = int(total_steps * 0.05)
    init_loss = 4.2

    for i in range(1, total_steps + 1):
        # LR schedule
        if i <= warmup:
            lr = lr_max * i / warmup
        else:
            progress = (i - warmup) / (total_steps - warmup)
            lr = lr_max * 0.5 * (1 + math.cos(math.pi * progress))

        # loss 곡선: 지수 감소 + 노이즈
        decay = math.exp(-4.5 * i / total_steps)
        noise = np.random.normal(0, 0.015 * (1 + decay))
        loss = max(0.3, init_loss * decay + 0.38 + noise)

        # grad_norm
        gn = 22 * math.exp(-2.5 * i / total_steps) + np.random.normal(0, 0.4)
        gn = max(4.0, gn)

        steps.append(i)
        losses.append(round(loss, 4))
        grads.append(round(gn, 3))
        lrs.append(round(lr, 8))
        elapsed.append(round(i * 12.5, 1))   # step당 약 12.5초 가정

    meta = {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "num_train_epochs": 5,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "total_steps": total_steps,
        "total_elapsed_sec": elapsed[-1],
    }
    step_logs = [
        {"step": s, "epoch": round((s / steps_per_epoch), 4),
         "loss": l, "grad_norm": g, "learning_rate": lr, "elapsed_sec": e}
        for s, l, g, lr, e in zip(steps, losses, grads, lrs, elapsed)
    ]
    return {"meta": meta, "steps": step_logs}


def extract_series(steps: list[dict]) -> tuple:
    s_arr  = np.array([d["step"]          for d in steps])
    ep_arr = np.array([d["epoch"]         for d in steps])
    l_arr  = np.array([d["loss"]          for d in steps], dtype=float)
    g_arr  = np.array([d.get("grad_norm") or np.nan for d in steps], dtype=float)
    lr_arr = np.array([d.get("learning_rate") or np.nan for d in steps], dtype=float)
    el_arr = np.array([d.get("elapsed_sec") or 0 for d in steps], dtype=float)
    return s_arr, ep_arr, l_arr, g_arr, lr_arr, el_arr


# ────────────────────────────────────────────────────────────────────────────
# 헬퍼: epoch 경계 수직선
# ────────────────────────────────────────────────────────────────────────────

def draw_epoch_lines(ax, steps, epochs, total_epochs, show_label=True):
    total_steps = steps[-1]
    steps_per_epoch = total_steps / total_epochs
    for ep in range(1, total_epochs):
        x = ep * steps_per_epoch
        ax.axvline(x, color=EPOCH_C, lw=0.6, alpha=0.25, ls="--")
        if show_label:
            ax.text(x + total_steps * 0.003, ax.get_ylim()[1] * 0.97,
                    f"ep{ep}", fontsize=6.5, color=MUTED_C, va="top")


def smooth(arr, w=7):
    """단순 이동평균 스무딩"""
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


# ────────────────────────────────────────────────────────────────────────────
# 패널 그리기
# ────────────────────────────────────────────────────────────────────────────

def panel_loss(ax, steps, epochs, losses, meta):
    """패널 1: Train Loss 곡선"""
    ax.set_facecolor(PANEL_BG)
    s_smooth = smooth(losses, w=9)
    ax.plot(steps, losses, color=ACCENT1, lw=0.7, alpha=0.35)
    ax.plot(steps, s_smooth, color=ACCENT1, lw=1.8, label="Train Loss (smoothed)")

    # 최솟값 강조
    min_idx = np.argmin(losses)
    ax.scatter(steps[min_idx], losses[min_idx], color="#ffffff", s=40, zorder=5)
    ax.annotate(
        f"min {losses[min_idx]:.4f}",
        xy=(steps[min_idx], losses[min_idx]),
        xytext=(steps[min_idx] + len(steps) * 0.04, losses[min_idx] + 0.1),
        fontsize=7.5, color=TEXT_C,
        arrowprops=dict(arrowstyle="->", color=MUTED_C, lw=0.8),
    )

    draw_epoch_lines(ax, steps, epochs, meta["num_train_epochs"])
    ax.set_xlabel("Step", fontsize=9, color=TEXT_C, labelpad=4)
    ax.set_ylabel("Loss", fontsize=9, color=TEXT_C, labelpad=4)
    ax.set_title("Train Loss", fontsize=11, color=TITLE_C, pad=8, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.2, labelcolor=TEXT_C)


def panel_grad(ax, steps, epochs, grads, meta):
    """패널 2: Gradient Norm"""
    ax.set_facecolor(PANEL_BG)
    valid = ~np.isnan(grads)
    if valid.any():
        g_smooth = smooth(grads[valid], w=9)
        ax.plot(steps[valid], grads[valid], color=ACCENT2, lw=0.7, alpha=0.35)
        ax.plot(steps[valid], g_smooth, color=ACCENT2, lw=1.8, label="Grad Norm (smoothed)")

    draw_epoch_lines(ax, steps, epochs, meta["num_train_epochs"], show_label=False)
    ax.set_xlabel("Step", fontsize=9, color=TEXT_C, labelpad=4)
    ax.set_ylabel("Gradient Norm", fontsize=9, color=TEXT_C, labelpad=4)
    ax.set_title("Gradient Norm", fontsize=11, color=TITLE_C, pad=8, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.2, labelcolor=TEXT_C)


def panel_lr(ax, steps, epochs, lrs, meta):
    """패널 3: Learning Rate Schedule"""
    ax.set_facecolor(PANEL_BG)
    valid = ~np.isnan(lrs)
    if valid.any():
        ax.plot(steps[valid], lrs[valid], color=ACCENT3, lw=1.8, label="Learning Rate")
        ax.fill_between(steps[valid], lrs[valid], alpha=0.12, color=ACCENT3)

    # Warmup 영역 표시
    warmup_end = int(steps[-1] * meta.get("warmup_ratio", 0.05))
    ax.axvspan(0, warmup_end, color=ACCENT3, alpha=0.05)
    ax.text(warmup_end * 0.5, lrs[valid].max() * 0.9 if valid.any() else 1e-4,
            "warmup", fontsize=6.5, color=MUTED_C, ha="center")

    draw_epoch_lines(ax, steps, epochs, meta["num_train_epochs"], show_label=False)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    ax.set_xlabel("Step", fontsize=9, color=TEXT_C, labelpad=4)
    ax.set_ylabel("Learning Rate", fontsize=9, color=TEXT_C, labelpad=4)
    ax.set_title("LR Schedule (Cosine)", fontsize=11, color=TITLE_C, pad=8, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.2, labelcolor=TEXT_C)


def panel_epoch_summary(ax, steps, epochs, losses, elapsed, meta):
    """패널 4: Epoch별 평균 Loss 바 차트 + 소요 시간"""
    ax.set_facecolor(PANEL_BG)
    total_epochs = meta["num_train_epochs"]

    ep_mean_loss = []
    ep_labels = []
    for ep in range(1, total_epochs + 1):
        mask = (epochs >= ep - 1) & (epochs < ep)
        if mask.any():
            ep_mean_loss.append(float(np.mean(losses[mask])))
            ep_labels.append(f"Epoch {ep}")

    x = np.arange(len(ep_labels))
    bars = ax.bar(x, ep_mean_loss, color=ACCENT1, alpha=0.75, width=0.55,
                  edgecolor=ACCENT1, linewidth=0.8)

    # 바 위에 수치 표시
    for bar, val in zip(bars, ep_mean_loss):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(ep_mean_loss) * 0.015,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8, color=TEXT_C)

    # 색상 그라디언트: 손실이 낮을수록 밝게
    if ep_mean_loss:
        min_l, max_l = min(ep_mean_loss), max(ep_mean_loss)
        for bar, val in zip(bars, ep_mean_loss):
            ratio = 1 - (val - min_l) / max(max_l - min_l, 1e-9)
            bar.set_alpha(0.45 + 0.5 * ratio)

    ax.set_xticks(x)
    ax.set_xticklabels(ep_labels, fontsize=8.5, color=TEXT_C)
    ax.set_ylabel("Avg Train Loss", fontsize=9, color=TEXT_C, labelpad=4)
    ax.set_title("Epoch별 평균 Loss", fontsize=11, color=TITLE_C, pad=8, fontweight="bold")

    # 총 소요 시간 표기
    total_sec = elapsed[-1] if len(elapsed) > 0 else 0
    h, m, s = int(total_sec // 3600), int((total_sec % 3600) // 60), int(total_sec % 60)
    time_str = f"총 학습 시간: {h}h {m}m {s}s" if h else f"총 학습 시간: {m}m {s}s"
    ax.text(0.98, 0.97, time_str, transform=ax.transAxes,
            fontsize=8, color=MUTED_C, ha="right", va="top")


# ────────────────────────────────────────────────────────────────────────────
# 공통 Axes 스타일
# ────────────────────────────────────────────────────────────────────────────

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_C)
    ax.grid(True, color=GRID_C, linewidth=0.6, alpha=0.8)
    pass  # xlim은 데이터 범위에 맞게 자동 설정


# ────────────────────────────────────────────────────────────────────────────
# 메인 렌더러
# ────────────────────────────────────────────────────────────────────────────

def render(data: dict, out_path: str):
    meta  = data.get("meta", {})
    steps_data = data.get("steps", [])

    if not steps_data:
        print("로그 데이터가 비어 있습니다.")
        return

    s_arr, ep_arr, l_arr, g_arr, lr_arr, el_arr = extract_series(steps_data)

    # ── 레이아웃
    fig = plt.figure(figsize=(14, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(
        3, 2,
        hspace=0.48, wspace=0.32,
        left=0.07, right=0.97,
        top=0.88, bottom=0.07,
    )
    ax_loss  = fig.add_subplot(gs[0, :])   # 상단 풀 너비
    ax_grad  = fig.add_subplot(gs[1, 0])
    ax_lr    = fig.add_subplot(gs[1, 1])
    ax_epoch = fig.add_subplot(gs[2, :])   # 하단 풀 너비

    for ax in [ax_loss, ax_grad, ax_lr, ax_epoch]:
        style_ax(ax)

    panel_loss(ax_loss, s_arr, ep_arr, l_arr, meta)
    panel_grad(ax_grad, s_arr, ep_arr, g_arr, meta)
    panel_lr(ax_lr, s_arr, ep_arr, lr_arr, meta)
    panel_epoch_summary(ax_epoch, s_arr, ep_arr, l_arr, el_arr, meta)

    # ── 헤더
    model_short = meta.get("model_id", "").split("/")[-1]
    header_left = (
        f"Model: {model_short}  |  "
        f"LoRA r={meta.get('lora_r','?')} α={meta.get('lora_alpha','?')} "
        f"dropout={meta.get('lora_dropout','?')}"
    )
    header_right = (
        f"Epochs: {meta.get('num_train_epochs','?')}  |  "
        f"LR: {meta.get('learning_rate','?')}  |  "
        f"Batch: {meta.get('batch_size','?')}×{meta.get('gradient_accumulation_steps','?')} (eff. {(meta.get('batch_size',1)*meta.get('gradient_accumulation_steps',1))})"
    )
    fig.text(0.07, 0.935, header_left,  fontsize=9,  color=MUTED_C, va="bottom")
    fig.text(0.97, 0.935, header_right, fontsize=9,  color=MUTED_C, va="bottom", ha="right")
    fig.text(0.5, 0.965, "LoRA Fine-tuning — Training Result",
             fontsize=15, color=TITLE_C, ha="center", va="bottom", fontweight="bold")

    # ── 저장
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"시각화 저장 완료: {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LoRA 학습 결과 시각화")
    parser.add_argument("--log",  default="./lora_output/train_log.json", help="train_log.json 경로")
    parser.add_argument("--out",  default="./lora_output/train_result.png", help="출력 PNG 경로")
    parser.add_argument("--demo", action="store_true", help="샘플 데이터로 미리보기")
    args = parser.parse_args()

    if args.demo:
        print("데모 모드: 샘플 데이터로 시각화합니다.")
        data = make_demo_data()
        base, ext = os.path.splitext(args.out)
        out  = base + "_demo" + ext if not base.endswith("_demo") else args.out
    else:
        if not os.path.exists(args.log):
            print(f"로그 파일 없음: {args.log}")
            print("  → 1_train.py 실행 후 다시 시도하거나, --demo 플래그로 미리보기")
            return
        data = load_log(args.log)
        out  = args.out

    render(data, out)


if __name__ == "__main__":
    main()
