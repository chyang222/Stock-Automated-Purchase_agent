"""PatchTST 기반 주가 상승/하락 분류 모델

입력 : 5분봉 과거 256봉 × 7채널
       [ret, log_vol, rsi, ma5_ratio, ma20_ratio, vol20, ret_60]
출력 : 60분 후(12봉) 수익률 > 수수료(0.21%) 여부 (이진 분류)

학습 데이터 : data/minute5/ 아래 *_5m.csv 파일들
저장 경로   : data/artifacts/patchtst_stock.pt

pip install torch transformers
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import MINUTE5_DIR, ARTIFACT_DIR

# ------------------------------------------------------------------ 하이퍼파라미터

CONTEXT_LEN   = 128    # 입력 길이 (128 × 5min ≈ 1.7 거래일) — CPU 속도 절충
PATCH_LEN     = 16     # 패치 크기
STRIDE        = 8      # 패치 간격  → 패치 수 = (128-16)/8+1 = 15개
N_CHANNELS    = 7      # 피처 수
HORIZON       = 12     # 예측 대상: 12봉 후 = 60분
FEE_THRESHOLD = 0.0021 # 왕복 수수료 0.21% — 이 이상 올라야 UP
SAMPLE_STRIDE = 12     # 슬라이딩 윈도우 보폭 (60분 간격) — 샘플 수 절반으로

PATCHTST_PATH   = ARTIFACT_DIR / "patchtst_stock.pt"
PATCHTST_METRICS = ARTIFACT_DIR / "patchtst_metrics.json"


# ------------------------------------------------------------------ 피처 계산

def _make_features(df: pd.DataFrame) -> np.ndarray:
    """5분봉 DataFrame → (T, 7) float32 배열 (모든 피처 스케일 불변)"""
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    ret = close.pct_change().fillna(0)

    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rsi   = (100 - 100 / (1 + gain / (loss + 1e-8))) / 100  # 0~1 정규화

    ma5  = close.rolling(5,  min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma5_ratio  = (close / (ma5  + 1e-8) - 1).clip(-0.2, 0.2)
    ma20_ratio = (close / (ma20 + 1e-8) - 1).clip(-0.2, 0.2)

    vol20  = ret.rolling(20, min_periods=1).std().fillna(0)
    ret_60 = close.pct_change(60).fillna(0).clip(-0.3, 0.3)  # 60봉(5시간) 누적 수익률

    feat = pd.DataFrame({
        "ret":        ret.clip(-0.1, 0.1),
        "log_vol":    np.log1p(volume),
        "rsi":        rsi,
        "ma5_ratio":  ma5_ratio,
        "ma20_ratio": ma20_ratio,
        "vol20":      vol20,
        "ret_60":     ret_60,
    })
    return feat.fillna(0).values.astype(np.float32)


# ------------------------------------------------------------------ Dataset

class StockDataset:
    def __init__(
        self,
        data_dir: Path = MINUTE5_DIR,
        context_len: int = CONTEXT_LEN,
        horizon: int = HORIZON,
        fee_threshold: float = FEE_THRESHOLD,
        max_files: int | None = None,
    ):
        import torch
        from torch.utils.data import TensorDataset

        self.context_len = context_len
        xs, ys = [], []

        files = sorted(data_dir.glob("*_5m.csv"))
        if max_files:
            files = files[:max_files]

        min_rows = context_len + horizon + 20
        print(f"데이터 로딩: {len(files)}개 파일")

        for path in files:
            try:
                df = pd.read_csv(path)
                if "close" not in df.columns or len(df) < min_rows:
                    continue

                feat  = _make_features(df)
                close = df["close"].astype(float).values
                T     = len(feat)

                for i in range(context_len, T - horizon, SAMPLE_STRIDE):
                    x = feat[i - context_len : i].copy()  # (context_len, 7)

                    # 채널 0(log_vol) 시퀀스 내 z-score 정규화 (거래량 수준 차이 제거)
                    mu, sigma = x[:, 1].mean(), x[:, 1].std() + 1e-8
                    x[:, 1] = (x[:, 1] - mu) / sigma

                    future_ret = (close[i + horizon - 1] - close[i - 1]) / (close[i - 1] + 1e-8)
                    label = int(future_ret > fee_threshold)

                    xs.append(x)
                    ys.append(label)
            except Exception:
                continue

        if not xs:
            raise RuntimeError("학습 데이터 없음 — python app.py collect-all 먼저 실행하세요")

        X = torch.tensor(np.stack(xs), dtype=torch.float32)   # (N, context_len, 7)
        Y = torch.tensor(ys, dtype=torch.long)                  # (N,)

        up_ratio = Y.float().mean().item()
        print(f"총 샘플: {len(Y):,}개 | UP 비율: {up_ratio:.1%}")

        split = int(len(Y) * 0.8)
        self.train = torch.utils.data.TensorDataset(X[:split], Y[:split])
        self.val   = torch.utils.data.TensorDataset(X[split:], Y[split:])


# ------------------------------------------------------------------ 모델 빌더

def build_model(context_len: int = CONTEXT_LEN) -> "PatchTSTForClassification":
    try:
        from transformers import PatchTSTConfig, PatchTSTForClassification
    except ImportError:
        raise ImportError("pip install transformers torch 를 먼저 실행하세요")

    config = PatchTSTConfig(
        num_input_channels=N_CHANNELS,
        context_length=context_len,
        patch_length=PATCH_LEN,
        stride=STRIDE,
        d_model=64,             # 128→64: attention 4배 경량화
        num_attention_heads=4,
        num_hidden_layers=2,    # 3→2: 레이어 감소
        ffn_dim=128,            # 256→128
        dropout=0.1,
        head_dropout=0.1,
        pooling_type="mean",
        channel_attention=False,
        scaling=False,
        num_targets=2,
    )
    return PatchTSTForClassification(config)


# ------------------------------------------------------------------ 학습

def train_patchtst(
    data_dir: Path = MINUTE5_DIR,
    artifact_dir: Path = ARTIFACT_DIR,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-4,
    max_files: int | None = None,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nPatchTST 학습 시작 | device={device}\n")

    ds  = StockDataset(data_dir=data_dir, max_files=max_files)
    import torch
    n_workers = 0 if device.type == "cpu" else 2
    pin_mem   = device.type != "cpu"
    train_loader = DataLoader(ds.train, batch_size=batch_size, shuffle=True,  num_workers=n_workers, pin_memory=pin_mem)
    val_loader   = DataLoader(ds.val,   batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=pin_mem)

    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"파라미터: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    import torch.nn.functional as F

    best_val_acc, best_state = 0.0, None

    n_batches = len(train_loader)
    for epoch in range(1, epochs + 1):
        # ── 학습
        model.train()
        t_loss = t_correct = t_total = 0
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(past_values=x).prediction_logits  # (B, 2)
            loss   = F.cross_entropy(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item()
            t_correct += (logits.argmax(-1) == y).sum().item()
            t_total   += len(y)
            if batch_idx % 50 == 0 or batch_idx == n_batches:
                print(f"  epoch {epoch} [{batch_idx}/{n_batches}] loss={t_loss/batch_idx:.4f}", flush=True)

        # ── 검증
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(past_values=x).prediction_logits
                v_correct += (logits.argmax(-1) == y).sum().item()
                v_total   += len(y)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        print(f"Epoch {epoch:2d}/{epochs} | loss={t_loss/len(train_loader):.4f} | train={t_acc:.3f} | val={v_acc:.3f}")

        scheduler.step()

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, PATCHTST_PATH)
            print(f"  ★ best 모델 저장 (val_acc={best_val_acc:.4f})", flush=True)

    metrics = {"best_val_acc": round(best_val_acc, 4), "epochs": epochs, "n_params": n_params}
    PATCHTST_METRICS.write_text(json.dumps(metrics, indent=2))

    print(f"\n모델 저장: {PATCHTST_PATH}")
    print(f"최고 val_acc: {best_val_acc:.3f}")
    return metrics


# ------------------------------------------------------------------ 추론

def predict(stock_code: str, data_dir: Path = MINUTE5_DIR) -> dict:
    """PatchTST로 단일 종목 상승 확률 예측"""
    import torch

    if not PATCHTST_PATH.exists():
        return {
            "stock_code": stock_code,
            "up_probability": None,
            "note": "PatchTST 모델 없음 — python app.py train-patchtst 실행 필요",
        }

    matches = list(data_dir.glob(f"{stock_code.zfill(6)}_*.csv"))
    if not matches:
        return {"stock_code": stock_code, "up_probability": None, "note": "5분봉 데이터 없음"}

    df = pd.read_csv(matches[0])
    if len(df) < CONTEXT_LEN + 20:
        return {"stock_code": stock_code, "up_probability": None, "note": f"데이터 부족 ({len(df)}봉 < {CONTEXT_LEN})"}

    feat = _make_features(df)
    x    = feat[-CONTEXT_LEN:].copy()

    # 거래량 채널 정규화 (학습 시와 동일 처리)
    mu, sigma = x[:, 1].mean(), x[:, 1].std() + 1e-8
    x[:, 1]   = (x[:, 1] - mu) / sigma

    x_t = torch.tensor(x[np.newaxis], dtype=torch.float32)  # (1, 256, 7)

    model = build_model()
    model.load_state_dict(torch.load(PATCHTST_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        logits = model(past_values=x_t).prediction_logits[0]
        up_prob = float(torch.softmax(logits, dim=-1)[1])

    return {
        "stock_code": stock_code,
        "up_probability": round(up_prob, 4),
        "signal": "UP" if up_prob > 0.55 else "DOWN" if up_prob < 0.45 else "NEUTRAL",
        "model": "PatchTST",
    }
