import numpy as np
from utils.data import load_stock
from utils.indicators import add_indicators
from model.features import scale_features, create_supervised
from model.lstm import build_lstm

def train_lstm(ticker):
    df = load_stock(ticker)
    df = add_indicators(df)

    feature_cols = ["Close", "rsi", "macd", "volatility", "return"]
    data = df[feature_cols].values

    scaled, scaler = scale_features(data)
    X, y = create_supervised(scaled, lookback=60, horizon=7)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm((X.shape[1], X.shape[2]), horizon=7)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    preds = model.predict(X_test)

    # Inverse transform only the Close column
    close_scaler = scaler

    def inverse(pred):
        dummy = np.zeros((7, data.shape[1]))
        dummy[:, 0] = pred
        inv = close_scaler.inverse_transform(dummy)
        return inv[:, 0]

    preds_inv = [inverse(p) for p in preds]
    y_inv = [inverse(t) for t in y_test]

    return df, preds_inv, y_inv
