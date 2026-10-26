
import numpy as np
import pandas as pd
import torch
from torch import nn


class _TimeNet(nn.Module):
    def __init__(self, n_users, n_items, factors):
        super().__init__()

        # Скрытые факторы пользователей и ресторанов
        self.P = nn.Embedding(n_users, factors)
        self.Q = nn.Embedding(n_items, factors)

        # Факторы истории взаимодействий (SVD++)
        self.Y = nn.Embedding(
            n_items + 1,
            factors,
            padding_idx=n_items
        )

        # Смещения оценок
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)

        # Временные изменения
        self.alpha = nn.Embedding(n_users, 1)
        self.time_item = nn.Embedding(n_items, 1)

        # Общая средняя оценка
        self.mu = nn.Parameter(torch.zeros(()))

        for emb in (self.P, self.Q):
            nn.init.normal_(emb.weight, std=0.03)

        nn.init.normal_(self.Y.weight, std=0.03)

        with torch.no_grad():
            self.Y.weight[-1].zero_()

        for emb in (
            self.bu,
            self.bi,
            self.alpha,
            self.time_item
        ):
            nn.init.zeros_(emb.weight)

    def forward(self, u, i, dev, day, implicit):
        user_vec = self.P(u) + implicit

        prediction = (
            self.mu
            + self.bu(u).squeeze(-1)
            + self.bi(i).squeeze(-1)
            + self.alpha(u).squeeze(-1) * dev
            + self.time_item(i).squeeze(-1) * day
            + (user_vec * self.Q(i)).sum(dim=-1)
        )

        return prediction


class TimeSVDpp:
    def __init__(
        self,
        factors=16,
        epochs=10,
        lr=0.003,
        reg=1e-4,
        batch_size=1024,
        seed=42,
        verbose=False
    ):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

    def fit(self, df):
        torch.manual_seed(self.seed)

        if df.duplicated(["user_id", "business_id"]).any():
            raise ValueError(
                "В данных есть повторные пары user_id и business_id"
            )

        if not np.isfinite(df["stars"].to_numpy()).all():
            raise ValueError("В оценках есть NaN или inf")

        # Индексы пользователей и ресторанов
        self.users = np.array(sorted(df["user_id"].unique()))
        self.items = np.array(sorted(df["business_id"].unique()))

        self.user_to_idx = {
            user: idx for idx, user in enumerate(self.users)
        }

        self.item_to_idx = {
            item: idx for idx, item in enumerate(self.items)
        }

        users = df["user_id"].map(
            self.user_to_idx
        ).to_numpy(dtype=np.int64)

        items = df["business_id"].map(
            self.item_to_idx
        ).to_numpy(dtype=np.int64)

        # Подготовка временных признаков
        dates = pd.to_datetime(df["date"])

        self.origin = dates.min()
        self.cutoff = dates.max()

        days = (
            (dates - self.origin).dt.total_seconds()
            .to_numpy() / 86400.0
        )

        self.global_day_mean = float(days.mean())
        self.day_std = max(float(np.std(days)), 1.0)

        counts = np.bincount(
            users,
            minlength=len(self.users)
        )

        self.day_mean = (
            np.bincount(
                users,
                weights=days,
                minlength=len(self.users)
            )
            / np.maximum(counts, 1)
        )

        # Центрированное время
        normalized_days = (
            days - self.global_day_mean
        ) / self.day_std

        # Отклонение даты от среднего времени пользователя
        delta = days - self.day_mean[users]

        deviations = (
            np.sign(delta)
            * (np.abs(delta) / self.day_std) ** 0.4
        )

        # Обучающие данные
        u = torch.tensor(users, dtype=torch.long)
        i = torch.tensor(items, dtype=torch.long)

        d = torch.tensor(
            normalized_days,
            dtype=torch.float32
        )

        dev = torch.tensor(
            deviations,
            dtype=torch.float32
        )

        y = torch.tensor(
            df["stars"].to_numpy(dtype=np.float32)
        )

        # История взаимодействий каждого пользователя
        self.history = [
            np.unique(items[users == j])
            for j in range(len(self.users))
        ]

        max_len = max(len(h) for h in self.history)

        padding_idx = len(self.items)

        h = np.full(
            (len(self.users), max_len),
            padding_idx,
            dtype=np.int64
        )

        mask = np.zeros(
            (len(self.users), max_len),
            dtype=np.float32
        )

        for j, hist in enumerate(self.history):
            n = len(hist)

            if n > 0:
                h[j, :n] = hist
                mask[j, :n] = 1.0 / np.sqrt(n)

        self.hidx = torch.from_numpy(h)
        self.hmask = torch.from_numpy(mask)

        # Создание модели
        self.net = _TimeNet(
            len(self.users),
            len(self.items),
            self.factors
        )

        with torch.no_grad():
            self.net.mu.fill_(float(y.mean()))

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.reg
        )

        # Обучение
        for epoch in range(self.epochs):
            self.net.train()

            total_loss = 0.0
            total_samples = 0

            for ids in torch.randperm(len(y)).split(
                self.batch_size
            ):
                uu = u[ids]

                implicit = (
                    self.net.Y(self.hidx[uu])
                    * self.hmask[uu].unsqueeze(-1)
                ).sum(dim=1)

                predictions = self.net(
                    uu,
                    i[ids],
                    dev[ids],
                    d[ids],
                    implicit
                )

                loss = (
                    (predictions - y[ids]) ** 2
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(ids)
                total_samples += len(ids)

            if self.verbose:
                mse = total_loss / total_samples

                print(
                    f"Epoch {epoch + 1}/{self.epochs} "
                    f"MSE = {mse:.4f}"
                )

        self.net.eval()
        return self

    @torch.no_grad()
    def recommend(self, user_id, candidates, k=10):
        if user_id not in self.user_to_idx:
            return []

        valid = sorted(
            item for item in candidates
            if item in self.item_to_idx
        )

        if not valid:
            return []

        uid = self.user_to_idx[user_id]

        # Для рекомендаций используем конец train
        day = float(
            (self.cutoff - self.origin).total_seconds()
            / 86400.0
        )

        normalized_day = (
            day - self.global_day_mean
        ) / self.day_std

        delta = day - self.day_mean[uid]

        deviation = (
            np.sign(delta)
            * (abs(delta) / self.day_std) ** 0.4
        )

        u = torch.tensor([uid], dtype=torch.long)

        implicit = (
            self.net.Y(self.hidx[u])
            * self.hmask[u].unsqueeze(-1)
        ).sum(dim=1)

        scores = []

        for start in range(0, len(valid), 2048):
            batch = valid[start:start + 2048]

            item_indices = torch.tensor(
                [self.item_to_idx[item] for item in batch],
                dtype=torch.long
            )

            n = len(item_indices)

            predictions = self.net(
                u.expand(n),
                item_indices,
                torch.full(
                    (n,),
                    float(deviation),
                    dtype=torch.float32
                ),
                torch.full(
                    (n,),
                    float(normalized_day),
                    dtype=torch.float32
                ),
                implicit.expand(n, -1)
            )

            scores.extend(predictions.numpy().tolist())

        order = np.argsort(
            -np.asarray(scores),
            kind="stable"
        )[:k]

        return [valid[j] for j in order]
