import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix


class EASE:
    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, df):
        if self.reg <= 0:
            raise ValueError("reg должен быть больше нуля")

        if df.duplicated(["user_id", "business_id"]).any():
            raise ValueError("Есть повторные оценки пользователя и ресторана")

        users = sorted(df["user_id"].unique())
        items = sorted(df["business_id"].unique())

        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}

        self.items = np.array(items)

        # Используем все оценки от 1 до 5
        rows = df["user_id"].map(self.user_to_idx).to_numpy()
        cols = df["business_id"].map(self.item_to_idx).to_numpy()
        ratings = df["stars"].to_numpy(dtype=np.float64)

        if not np.isfinite(ratings).all():
            raise ValueError("В оценках есть NaN или inf")

        X = csr_matrix(
            (ratings, (rows, cols)), shape=(len(users), len(items)), dtype=np.float64
        )

        self.X = X

        # Матрица связей между ресторанами
        G = (X.T @ X).toarray()

        # L2-регуляризация
        G.flat[:: len(items) + 1] += self.reg

        # Вычисляем обратную матрицу
        factor = cho_factor(G, lower=True, check_finite=False)

        P = cho_solve(factor, np.eye(len(items), dtype=np.float64), check_finite=False)

        # Матрица весов EASE
        diagonal = np.diag(P).copy()

        self.B = -P / diagonal[np.newaxis, :]
        np.fill_diagonal(self.B, 0.0)

        return self

    def recommend(self, user_id, candidates, k=10):
        if user_id not in self.user_to_idx:
            return []

        valid = [item for item in candidates if item in self.item_to_idx]

        if not valid:
            return []

        indices = np.array([self.item_to_idx[item] for item in valid], dtype=int)

        user_idx = self.user_to_idx[user_id]

        history = self.X.getrow(user_idx)

        scores = np.asarray(history @ self.B[:, indices]).ravel()

        order = np.lexsort((indices, -scores))[:k]

        return [valid[j] for j in order]
