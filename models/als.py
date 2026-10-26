import numpy as np
from scipy.sparse import csr_matrix


class ExplicitALS:
    def __init__(self, factors=32, reg=10.0, epochs=8, random_state=42):
        self.factors = factors
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, df):
        self.users = np.array(sorted(df["user_id"].unique()))
        self.items = np.array(sorted(df["business_id"].unique()))

        self.user_to_idx = {
            user: i for i, user in enumerate(self.users)
        }
        self.item_to_idx = {
            item: i for i, item in enumerate(self.items)
        }

        rows = df["user_id"].map(self.user_to_idx).to_numpy()
        cols = df["business_id"].map(self.item_to_idx).to_numpy()
        ratings = df["stars"].to_numpy(dtype=np.float64)

        assert np.isfinite(ratings).all()
        assert np.all((ratings >= 1) & (ratings <= 5))

        self.mean = float(ratings.mean())

        R = csr_matrix(
            (ratings - self.mean, (rows, cols)),
            shape=(len(self.users), len(self.items)),
            dtype=np.float64
        )

        R.sort_indices()

        self.R = R
        RT = R.T.tocsr()

        rng = np.random.default_rng(self.random_state)

        self.U = rng.normal(
            0, 0.1, size=(len(self.users), self.factors)
        ).astype(np.float64)

        self.V = rng.normal(
            0, 0.1, size=(len(self.items), self.factors)
        ).astype(np.float64)

        eye = np.eye(self.factors, dtype=np.float64)

        for epoch in range(self.epochs):

            for u in range(len(self.users)):
                start, end = R.indptr[u:u + 2]
                item_ids = R.indices[start:end]

                if len(item_ids) == 0:
                    continue

                X = self.V[item_ids]
                y = R.data[start:end]

                A = X.T @ X + self.reg * eye
                b = X.T @ y

                self.U[u] = np.linalg.solve(A, b)

            for i in range(len(self.items)):
                start, end = RT.indptr[i:i + 2]
                user_ids = RT.indices[start:end]

                if len(user_ids) == 0:
                    continue

                X = self.U[user_ids]
                y = RT.data[start:end]

                A = X.T @ X + self.reg * eye
                b = X.T @ y

                self.V[i] = np.linalg.solve(A, b)

            if not np.isfinite(self.U).all():
                raise ValueError(
                    f"NaN или inf в U на эпохе {epoch + 1}"
                )

            if not np.isfinite(self.V).all():
                raise ValueError(
                    f"NaN или inf в V на эпохе {epoch + 1}"
                )

        return self

    def recommend(self, user_id, candidates, k=10):
        if user_id not in self.user_to_idx:
            return []

        valid = [
            item for item in candidates
            if item in self.item_to_idx
        ]

        if not valid:
            return []

        indices = np.array([
            self.item_to_idx[item] for item in valid
        ])

        user_vector = self.U[self.user_to_idx[user_id]]

        scores = self.V[indices] @ user_vector + self.mean

        order = np.lexsort((indices, -scores))[:k]

        return [valid[j] for j in order]