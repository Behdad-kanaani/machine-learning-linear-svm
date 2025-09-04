# ==============================================================
#  Linear SVM Implementation with Platt Scaling (Probability)
#  Author: Behdad Kanaani
#  GitHub: https://github.com/Behdad-kanaani
# ==============================================================

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------------------- Utility Functions ----------------------

def addintercp(X):
    """
    Add intercept term (bias) to feature matrix.
    """
    return np.hstack([X, np.ones((X.shape[0], 1))])

def shuf_togeth(a, b, rng):
    """
    Shuffle X and y together with same permutation.
    """
    idx = rng.permutation(a.shape[0])
    return a[idx], b[idx]

# ---------------------- Linear SVM Class ----------------------

@dataclass
class LinSvm:
    C: float = 1.0        # Regularization strength
    lr: float = 0.1       # Learning rate
    epochs: int = 50      # Number of epochs
    bsize: int = 64       # Mini-batch size
    fit_icpt: bool = True # Whether to add intercept
    tol: float = 1e-6     # Convergence tolerance
    rnd: int | None = None  # Random seed

    def __post_init__(self):
        self.w = None
        self.A = None
        self.B = None
        self.rng = np.random.default_rng(self.rnd)

    def fit(self, X, y):
        """
        Train linear SVM with hinge loss using SGD.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        if self.fit_icpt:
            Xx = addintercp(X)
        else:
            Xx = X

        n, m = Xx.shape
        self.w = np.zeros(m)
        last = 1e18

        for ep in range(self.epochs):
            # Shuffle data each epoch
            Xx, y = shuf_togeth(Xx, y, self.rng)

            # Mini-batch SGD
            for st in range(0, n, self.bsize):
                en = min(st + self.bsize, n)
                Xb = Xx[st:en]
                yb = y[st:en]

                marg = yb * (Xb @ self.w)
                ms = marg < 1

                # Gradient of hinge loss
                if np.any(ms):
                    grad_h = -self.C * (yb[ms][:, None] * Xb[ms]).sum(0)
                else:
                    grad_h = np.zeros_like(self.w)

                # Weight update
                self.w -= self.lr * (self.w + grad_h)

            # Compute objective for convergence check
            marg2 = y * (Xx @ self.w)
            loss = np.maximum(0, 1 - marg2)
            obj = 0.5 * np.dot(self.w, self.w) + self.C * loss.sum()

            if abs(last - obj) < self.tol:
                break
            last = obj

        return self

    def decfun(self, X):
        """
        Decision function (raw scores).
        """
        X = np.asarray(X, float)
        if self.fit_icpt:
            X = addintercp(X)
        return X @ self.w

    def predict(self, X):
        """
        Predict class labels {-1, +1}.
        """
        f = self.decfun(X)
        return np.where(f >= 0, 1.0, -1.0)

    def fitplatt(self, X, y, max_it=100, tol=1e-6):
        """
        Fit Platt scaling to convert SVM scores into probabilities.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        y01 = (y == 1).astype(float)
        f = self.decfun(X)

        # Initialize A and B
        A, B = 0.0, np.log((y01.mean() + 1e-6) / (1 - y01.mean() + 1e-6))

        for _ in range(max_it):
            z = A * f + B
            p = 1 / (1 + np.exp(-z))

            # Gradients
            gA = np.sum((p - y01) * f)
            gB = np.sum(p - y01)

            # Hessian
            W = p * (1 - p)
            HAA = np.sum(W * f * f) + 1e-9
            HAB = np.sum(W * f)
            HBB = np.sum(W) + 1e-9

            det = HAA * HBB - HAB * HAB
            if det <= 0:
                break

            # Newton step
            dA = -(HBB * gA - HAB * gB) / det
            dB = -(-HAB * gA + HAA * gB) / det

            A2, B2 = A + dA, B + dB

            # Convergence check
            if max(abs(dA), abs(dB)) < tol:
                A, B = A2, B2
                break
            A, B = A2, B2

        self.A, self.B = A, B
        return self

    def proba(self, X):
        """
        Predict probability estimates using Platt scaling.
        """
        f = self.decfun(X)
        z = self.A * f + self.B
        p = 1 / (1 + np.exp(-z))
        return np.vstack([1 - p, p]).T

# ---------------------- Demo Section ----------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400

    # Generate two Gaussian clusters
    X1 = rng.normal([2, 2], 0.7, (n // 2, 2))
    X2 = rng.normal([-2, -2], 0.7, (n // 2, 2))

    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(n // 2), -np.ones(n // 2)])

    # Shuffle data
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    # Train classifier
    clf = LinSvm(C=1.0, lr=0.05, epochs=100, bsize=32, rnd=42)
    clf.fit(X, y)
    clf.fitplatt(X, y)

    # Print accuracy
    print("acc:", (clf.predict(X) == y).mean())

    # Plot decision boundaries
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.proba(grid)[:, 1].reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, levels=20, cmap="RdBu", alpha=0.6)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="blue")
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c="red")

    # Decision boundary
    dec = clf.decfun(grid).reshape(xx.shape)
    plt.contour(xx, yy, dec, levels=[0], colors="k")

    plt.show()
