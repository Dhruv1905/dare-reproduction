"""
Feedforward NN digital twin: (v_T, T_UP) -> T_FCL.
Paper doesn't specify exact architecture; this is a reasonable default.
"""

import numpy as np
import torch
import torch.nn as nn


class DigitalTwin(nn.Module):
    def __init__(self, n_inputs=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(X_train, y_train, epochs=300, lr=1e-3, batch_size=512, seed=0, verbose=True):
    """Train the digital twin. Returns the fitted model + input/output scalers."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    # Simple standardization for stable training. Save stats for inference.
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = y.mean(), y.std() + 1e-8
    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    Xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(yn)

    model = DigitalTwin(n_inputs=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = X.shape[0]
    for ep in range(epochs):
        idx = np.random.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            pred = model(Xt[b])
            loss = loss_fn(pred, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)
        if verbose and (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1:3d}/{epochs}  loss={total/n:.5f}")

    scalers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, scalers


def predict(model, X, scalers):
    """Predict in original units."""
    X = np.asarray(X, dtype=np.float32)
    Xn = (X - scalers["x_mean"]) / scalers["x_std"]
    model.eval()
    with torch.no_grad():
        yn = model(torch.from_numpy(Xn)).numpy()
    return yn * scalers["y_std"] + scalers["y_mean"]