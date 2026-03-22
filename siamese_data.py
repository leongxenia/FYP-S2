import numpy as np
import torch

from torch.utils.data import Dataset

from siamese_utils import to_dense_float32


def prepare_encoded_data(preprocess, X_train, X_val, X_test, T_train, T_val, T_test):
    X_train_enc = preprocess.fit_transform(X_train)
    X_val_enc = preprocess.transform(X_val)
    X_test_enc = preprocess.transform(X_test)

    Xtr = to_dense_float32(X_train_enc)
    Xva = to_dense_float32(X_val_enc)
    Xte = to_dense_float32(X_test_enc)

    Ttr = np.asarray(T_train, dtype=np.int64)
    Tva = np.asarray(T_val, dtype=np.int64)
    Tte = np.asarray(T_test, dtype=np.int64)

    return Xtr, Xva, Xte, Ttr, Tva, Tte


class PairDataset(Dataset):
    """
    Pair labels:
      same = 0 : (A,A) or (B,B)
      diff = 1 : (A,B) or (B,A)
    """

    def __init__(self, X, T, n_pairs=200000, pos_fraction=0.5, seed=42, include_ba=True):
        rng = np.random.default_rng(seed)
        self.X = X
        self.T = T

        idx_A = np.where(T == 0)[0]
        idx_B = np.where(T == 1)[0]

        if len(idx_A) == 0 or len(idx_B) == 0:
            raise ValueError("Need both group A and group B samples to form pairs.")

        n_diff = int(n_pairs * pos_fraction)
        n_same = n_pairs - n_diff

        if include_ba:
            n_ab = n_diff // 2
            n_ba = n_diff - n_ab

            a1 = rng.choice(idx_A, size=n_ab, replace=True)
            b1 = rng.choice(idx_B, size=n_ab, replace=True)
            pairs_ab = np.stack([a1, b1], axis=1)

            b2 = rng.choice(idx_B, size=n_ba, replace=True)
            a2 = rng.choice(idx_A, size=n_ba, replace=True)
            pairs_ba = np.stack([b2, a2], axis=1)

            diff_pairs = np.concatenate([pairs_ab, pairs_ba], axis=0)
        else:
            a1 = rng.choice(idx_A, size=n_diff, replace=True)
            b1 = rng.choice(idx_B, size=n_diff, replace=True)
            diff_pairs = np.stack([a1, b1], axis=1)

        diff_labels = np.ones(len(diff_pairs), dtype=np.int64)

        n_same_A = n_same // 2
        n_same_B = n_same - n_same_A

        a3 = rng.choice(idx_A, size=n_same_A, replace=True)
        a4 = rng.choice(idx_A, size=n_same_A, replace=True)
        sameA = np.stack([a3, a4], axis=1)

        b3 = rng.choice(idx_B, size=n_same_B, replace=True)
        b4 = rng.choice(idx_B, size=n_same_B, replace=True)
        sameB = np.stack([b3, b4], axis=1)

        same_pairs = np.concatenate([sameA, sameB], axis=0)
        same_labels = np.zeros(len(same_pairs), dtype=np.int64)

        self.pairs = np.concatenate([diff_pairs, same_pairs], axis=0)
        self.labels = np.concatenate([diff_labels, same_labels], axis=0)

        perm = rng.permutation(len(self.labels))
        self.pairs = self.pairs[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        idx1, idx2 = self.pairs[i]
        x1 = torch.from_numpy(self.X[idx1])
        x2 = torch.from_numpy(self.X[idx2])
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x1, x2, y


class BagPairDataset(Dataset):
    """
    Bag pair labels:
      same = 0 : (A,A) or (B,B)
      diff = 1 : (A,B) or (B,A)
    """

    def __init__(
        self,
        X,
        T,
        bag_size=10,
        n_pairs=50000,
        pos_fraction=0.5,
        include_ba=True,
        seed=42,
        within_bag_replace=False,
        max_row_reuse=None,
    ):
        rng = np.random.default_rng(seed)

        self.X = X
        self.T = T
        self.bag_size = int(bag_size)

        idx_A = np.where(T == 0)[0]
        idx_B = np.where(T == 1)[0]

        if len(idx_A) == 0 or len(idx_B) == 0:
            raise ValueError("Need both A and B samples.")

        n_diff = int(n_pairs * pos_fraction)
        n_same = n_pairs - n_diff

        reuse = np.zeros(len(T), dtype=np.int32) if max_row_reuse is not None else None

        def sample_bag(idx_pool):
            k = self.bag_size

            if (not within_bag_replace) and (len(idx_pool) >= k):
                replace_within = False
            else:
                replace_within = True

            if reuse is None:
                return rng.choice(idx_pool, size=k, replace=replace_within)

            candidates = idx_pool[reuse[idx_pool] < max_row_reuse]
            if len(candidates) < k:
                candidates = idx_pool

            bag = rng.choice(candidates, size=k, replace=replace_within)
            for j in bag:
                reuse[j] += 1
            return bag

        pairs = []
        labels = []

        if include_ba:
            n_ab = n_diff // 2
            n_ba = n_diff - n_ab

            for _ in range(n_ab):
                pairs.append((sample_bag(idx_A), sample_bag(idx_B)))
                labels.append(1)

            for _ in range(n_ba):
                pairs.append((sample_bag(idx_B), sample_bag(idx_A)))
                labels.append(1)
        else:
            for _ in range(n_diff):
                pairs.append((sample_bag(idx_A), sample_bag(idx_B)))
                labels.append(1)

        n_same_A = n_same // 2
        n_same_B = n_same - n_same_A

        for _ in range(n_same_A):
            pairs.append((sample_bag(idx_A), sample_bag(idx_A)))
            labels.append(0)

        for _ in range(n_same_B):
            pairs.append((sample_bag(idx_B), sample_bag(idx_B)))
            labels.append(0)

        perm = rng.permutation(len(labels))
        self.pairs = [pairs[i] for i in perm]
        self.labels = np.asarray([labels[i] for i in perm], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        bag1_idx, bag2_idx = self.pairs[i]
        bag1 = torch.from_numpy(self.X[bag1_idx])
        bag2 = torch.from_numpy(self.X[bag2_idx])
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return bag1, bag2, y