"""Transformer for next-day return prediction, evaluated on a held-out test period against simple baselines."""

import argparse
import json
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = [
    "ret_1", "ret_5", "ret_10", "log_volume_z",
    "close_sma_ratio", "close_ema_ratio", "rsi", "macd_norm", "macd_signal_norm",
    "bb_position", "volatility_10", "obv_change",
]


@dataclass
class Config:
    ticker: str = "GOOG"
    period: str = "10y"
    seq_length: int = 60
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-4
    batch_size: int = 64
    train_frac: float = 0.70
    val_frac: float = 0.15
    cost_bps: float = 5.0  # one-way transaction cost in basis points
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------- FEATURES ----------------
def load_prices(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"]
    log_close = np.log(close)

    out["ret_1"] = log_close.diff()
    out["ret_5"] = log_close.diff(5)
    out["ret_10"] = log_close.diff(10)

    log_vol = np.log(df["Volume"].replace(0, np.nan))
    out["log_volume_z"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()

    out["close_sma_ratio"] = close / close.rolling(14).mean() - 1
    out["close_ema_ratio"] = close / close.ewm(span=14, adjust=False).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi"] = (100 - 100 / (1 + gain / (loss + 1e-9))) / 100

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    out["macd_norm"] = macd / close
    out["macd_signal_norm"] = macd.ewm(span=9, adjust=False).mean() / close

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_position"] = (close - sma20) / (2 * std20 + 1e-9)

    out["volatility_10"] = out["ret_1"].rolling(10).std()

    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * df["Volume"]).cumsum()
    out["obv_change"] = obv.diff(5) / df["Volume"].rolling(20).mean()

    # target = next day's return
    out["target"] = out["ret_1"].shift(-1)
    out["close"] = close
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def make_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int):
    """Window ending at row i (inclusive) predicts targets[i] = return i -> i+1."""
    X, y = [], []
    for i in range(seq_length - 1, len(features)):
        X.append(features[i - seq_length + 1 : i + 1])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------- MODEL ----------------
class StockTransformer(nn.Module):
    def __init__(self, n_features: int, seq_length: int, d_model: int = 32,
                 num_layers: int = 2, num_heads: int = 4, ff_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.encoder(self.input_proj(x) + self.pos)
        return self.head(h[:, -1, :]).squeeze(-1)


# ---------------- TRAINING ----------------
def train(model, train_ds, val_ds, cfg: Config, device):
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_X, val_y = (t.to(device) for t in val_ds.tensors)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X), val_y).item()
        if val_loss < best_val - 1e-6:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  val_mse(scaled)={val_loss:.4f}  best={best_val:.4f}")
        if bad_epochs >= cfg.patience:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


# ---------------- EVALUATION ----------------
def regression_metrics(pred: np.ndarray, actual: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))
    nonzero = actual != 0
    dir_acc = float(np.mean(np.sign(pred[nonzero]) == np.sign(actual[nonzero])))
    return {"rmse": rmse, "mae": mae, "directional_accuracy": dir_acc}


def backtest(pred: np.ndarray, actual: np.ndarray, cost_bps: float) -> dict:
    """Long when the predicted return is positive, flat otherwise. Costs charged on each position change."""
    position = (pred > 0).astype(float)
    trades = np.abs(np.diff(np.concatenate([[0.0], position])))
    simple_ret = np.expm1(actual)
    strat = position * simple_ret - trades * cost_bps / 1e4
    equity = np.cumprod(1 + strat)
    bh_equity = np.cumprod(1 + simple_ret)

    def sharpe(r):
        return float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(252))

    def max_drawdown(eq):
        peak = np.maximum.accumulate(eq)
        return float(np.min(eq / peak - 1))

    return {
        "strategy_total_return": float(equity[-1] - 1),
        "buy_hold_total_return": float(bh_equity[-1] - 1),
        "strategy_sharpe": sharpe(strat),
        "buy_hold_sharpe": sharpe(simple_ret),
        "strategy_max_drawdown": max_drawdown(equity),
        "buy_hold_max_drawdown": max_drawdown(bh_equity),
        "time_in_market": float(position.mean()),
        "num_trades": int(trades.sum()),
        "equity": equity,
        "bh_equity": bh_equity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=Config.ticker)
    parser.add_argument("--period", default=Config.period)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--plot", action="store_true", help="save results_<ticker>.png")
    args = parser.parse_args()
    cfg = Config(ticker=args.ticker, period=args.period, epochs=args.epochs)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = add_features(load_prices(cfg.ticker, cfg.period))
    n = len(data)
    train_end = int(n * cfg.train_frac)
    val_end = int(n * (cfg.train_frac + cfg.val_frac))
    print(f"{cfg.ticker}: {n} rows | train {data.index[0].date()}..{data.index[train_end - 1].date()}"
          f" | val ..{data.index[val_end - 1].date()} | test ..{data.index[-1].date()}")

    # fit scalers on train only
    x_scaler = StandardScaler().fit(data[FEATURES].iloc[:train_end])
    y_scaler = StandardScaler().fit(data[["target"]].iloc[:train_end])
    feats = x_scaler.transform(data[FEATURES]).astype(np.float32)
    targs = y_scaler.transform(data[["target"]]).ravel().astype(np.float32)

    X, y = make_sequences(feats, targs, cfg.seq_length)
    # assign each window to the split of its last row; drop the boundary rows whose
    # target falls in the next split
    end_rows = np.arange(cfg.seq_length - 1, n)
    tr = end_rows < train_end - 1
    va = (end_rows >= train_end) & (end_rows < val_end - 1)
    te = end_rows >= val_end

    def ds(mask):
        return TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(y[mask]))

    model = StockTransformer(len(FEATURES), cfg.seq_length).to(device)
    model = train(model, ds(tr), ds(va), cfg, device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X[te]).to(device)).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual = data["target"].to_numpy()[end_rows[te]]
    train_mean = float(data["target"].iloc[:train_end].mean())

    model_m = regression_metrics(pred, actual)
    zero_m = regression_metrics(np.zeros_like(actual), actual)
    mean_m = regression_metrics(np.full_like(actual, train_mean), actual)
    up_rate = float(np.mean(actual[actual != 0] > 0))
    bt = backtest(pred, actual, cfg.cost_bps)

    report = {
        "ticker": cfg.ticker,
        "test_period": [str(data.index[end_rows[te][0]].date()), str(data.index[end_rows[te][-1]].date())],
        "test_days": int(te.sum()),
        "model": model_m,
        "baseline_zero_return": {k: zero_m[k] for k in ("rmse", "mae")},
        "baseline_train_mean": mean_m,
        "majority_class_rate_up": up_rate,
        "backtest": {k: v for k, v in bt.items() if not isinstance(v, np.ndarray)},
    }
    print(json.dumps(report, indent=2))
    with open(f"results_{cfg.ticker}.json", "w") as f:
        json.dump(report, f, indent=2)

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dates = data.index[end_rows[te]]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(dates, bt["bh_equity"], label="Buy & hold")
        ax.plot(dates, bt["equity"], label=f"Transformer long/flat ({cfg.cost_bps:.0f} bps costs)")
        ax.set_title(f"{cfg.ticker}: out-of-sample test period")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"results_{cfg.ticker}.png", dpi=120)


if __name__ == "__main__":
    main()
