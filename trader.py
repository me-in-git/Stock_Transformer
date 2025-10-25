import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
TICKER = "GOOG"
HIST_PERIOD = "5y"
SEQ_LENGTH = 60
DELAY_DAYS = 2
EPOCHS = 100
LR = 3e-4
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = ['Open','High','Low','Close','Volume',
            'SMA','EMA','RSI','MACD','MACD_SIGNAL',
            'BB_UPPER','BB_LOWER','OBV']
FEATURE_SIZE = len(FEATURES)

def SMA(df, period=14, column='Close'):
    return df[column].rolling(window=period).mean()

def EMA(df, period=14, column='Close'):
    return df[column].ewm(span=period, adjust=False).mean()

def RSI(df, period=14, column='Close'):
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def MACD(df, fast=12, slow=26, signal=9, column='Close'):
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def Bollinger_Bands(df, period=20, column='Close'):
    sma = df[column].rolling(period).mean()
    std = df[column].rolling(period).std()
    df['BB_UPPER'] = sma + 2 * std
    df['BB_LOWER'] = sma - 2 * std
    return df

def OBV(df):
    obv = [0]
    close = df['Close'].values  
    volume = df['Volume'].values
    for i in range(1, len(df)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    return df


def add_indicators(df):
    df['SMA'] = SMA(df)
    df['EMA'] = EMA(df)
    df['RSI'] = RSI(df)
    df['MACD'], df['MACD_SIGNAL'] = MACD(df)
    df = Bollinger_Bands(df)
    df = OBV(df)
    return df.dropna()

def create_features_array(df):
    return df[FEATURES].values

def create_sequences(data_array, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(seq_length, len(data_array)):
        X.append(data_array[i-seq_length:i])
        y.append(data_array[i, FEATURES.index('Close')])
    return np.array(X), np.array(y)

# MODEL
class StockTransformer(nn.Module):
    def __init__(self, feature_size=FEATURE_SIZE, seq_length=SEQ_LENGTH,
                 embed_dim=64, num_layers=4, num_heads=8,
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.positional_encoding
        x = self.transformer(x)
        return self.fc(x[:, -1, :])

# TRAINING
def train_model(X, y):
    # Scale features
    feature_scaler = MinMaxScaler()
    n_samples, seq_len, n_feats = X.shape
    X_2d = X.reshape(-1, n_feats)
    X_scaled_2d = feature_scaler.fit_transform(X_2d)
    X_scaled = X_scaled_2d.reshape(n_samples, seq_len, n_feats)

    # Scale target
    y_scaled = MinMaxScaler().fit_transform(y.reshape(-1,1))
    target_scaler = MinMaxScaler()
    target_scaler.fit(y.reshape(-1,1))
    y_scaled = target_scaler.transform(y.reshape(-1,1))

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = StockTransformer().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Training on {DEVICE}...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        rmse = np.sqrt(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss (MSE scaled)={avg_loss:.6f}, RMSE (scaled)={rmse:.6f}")

    return model, feature_scaler, target_scaler


def walk_forward_eval(model, feature_scaler, target_scaler, df, seq_length=SEQ_LENGTH, delay_days=DELAY_DAYS):
    features = create_features_array(df)
    close_idx = FEATURES.index('Close')

    preds, actuals, dates = [], [], []

    for i in range(seq_length, len(features) - delay_days):
        past_window = features[i-seq_length:i]
        scaled = feature_scaler.transform(past_window)
        X_input = torch.tensor(scaled.reshape(1, seq_length, FEATURE_SIZE), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            pred_scaled = model(X_input).cpu().numpy().reshape(-1)[0]

        pred_price = target_scaler.inverse_transform([[pred_scaled]])[0,0]
        actual_price = features[i+delay_days, close_idx]

        preds.append(pred_price)
        actuals.append(actual_price)
        dates.append(df.index[i+delay_days])

    preds = np.array(preds)
    actuals = np.array(actuals)
    mse = np.mean((preds - actuals)**2)
    rmse = np.sqrt(mse)
    print(f"\nWalk-forward evaluation with {delay_days}-day lag:")
    print(f"MSE: {mse:.4f} USD^2, RMSE: {rmse:.2f} USD, Mean % Error: {np.mean(np.abs(preds-actuals)/actuals)*100:.2f}%")
    return dates, preds, actuals

def simulate_trading(dates, preds, actuals, initial_cash=10000):
    cash = initial_cash
    shares = 0
    portfolio_values = []

    for i in range(len(preds)-1):
        if preds[i+1] > actuals[i]:
            if cash > 0:
                shares += cash / actuals[i]
                cash = 0
        elif preds[i+1] < actuals[i]:
            cash += shares * actuals[i]
            shares = 0
        portfolio_values.append(cash + shares * actuals[i])

    # final day
    cash += shares * actuals[-1]
    portfolio_values.append(cash)
    profit = cash - initial_cash
    print(f"Total Profit: ${profit:.2f} on initial ${initial_cash}")
    return portfolio_values

if __name__ == "__main__":
    df = yf.download(TICKER, period=HIST_PERIOD)
    df = add_indicators(df)
    cutoff = df.index[-DELAY_DAYS]
    train_df = df[:cutoff]
    test_df = df

    X_train, y_train = create_sequences(create_features_array(train_df), SEQ_LENGTH)
    model, feature_scaler, target_scaler = train_model(X_train, y_train)

    dates, preds, actuals = walk_forward_eval(model, feature_scaler, target_scaler, test_df)
    portfolio_values = simulate_trading(dates, preds, actuals)

    # Plot
    plt.figure(figsize=(14,6))
    plt.plot(dates, actuals, label="Actual Close")
    plt.plot(dates, preds, label="Predicted Close")
    plt.plot(dates, portfolio_values, label="Portfolio Value")
    plt.legend()
    plt.title(f"{TICKER} Predictions & Simulated Portfolio")
    plt.show()
