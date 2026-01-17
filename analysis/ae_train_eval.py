#!/usr/bin/env python3
"""
Train and evaluate a benign-only autoencoder anomaly detector.

This script does NOT modify the original supervised pipeline. It:
  - Loads outputs/hybrid_features.npz
  - Builds the same byte + structural feature vector
  - Trains on benign samples only
  - Uses a benign-only validation percentile for thresholding
  - Evaluates on the full val/test splits (benign + malicious)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


def compute_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def deduplicate_by_hash(
    paths: np.ndarray,
    X_bytes: np.ndarray,
    X_struct: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    priority = {"landfall": 2, "general_mal": 1, "benign": 0}
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(paths):
        h = compute_hash(str(p))
        groups.setdefault(h, []).append(i)

    kept: List[Tuple[int, str]] = []
    for _, idxs in groups.items():
        group_labels = [labels[i] for i in idxs]
        label = max(group_labels, key=lambda x: priority.get(x, -1))
        i0 = idxs[0]
        kept.append((i0, label))

    kept.sort(key=lambda x: x[0])
    indices = [i for i, _ in kept]
    labels_new = np.array([label for _, label in kept])
    return X_bytes[indices], X_struct[indices], labels_new, paths[indices]


def magika_bytes_to_features(X_bytes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return X_bytes.astype(np.float32) / 256.0
    if mode != "hist":
        raise ValueError(f"Unknown bytes mode: {mode}")

    beg = X_bytes[:, :1024]
    end = X_bytes[:, 1024:]
    bins = 257
    feats = []
    for block in (beg, end):
        h = np.zeros((block.shape[0], bins), dtype=np.float32)
        for i in range(block.shape[0]):
            counts = np.bincount(block[i].astype(np.int64), minlength=bins)
            h[i] = counts / float(block.shape[1])
        feats.append(h)
    return np.concatenate(feats, axis=1)


def build_struct_features(X_struct: np.ndarray, names: List[str]) -> np.ndarray:
    log_fields = {
        "min_width",
        "min_height",
        "ifd_entry_max",
        "subifd_count_sum",
        "new_subfile_types_unique",
        "total_opcodes",
        "unknown_opcodes",
        "max_opcode_id",
        "opcode_list1_bytes",
        "opcode_list2_bytes",
        "opcode_list3_bytes",
    }
    X = X_struct.astype(np.float32).copy()
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in log_fields:
        idx = name_to_idx.get(name)
        if idx is not None:
            X[:, idx] = np.log1p(X[:, idx])
    return X


def split_indices(n: int, seed: int, splits: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.0

    order = np.argsort(y_score)
    y_score_sorted = y_score[order]
    y_true_sorted = y_true[order]

    ranks = np.zeros_like(y_score_sorted, dtype=np.float64)
    i = 0
    while i < len(y_score_sorted):
        j = i
        while j + 1 < len(y_score_sorted) and y_score_sorted[j + 1] == y_score_sorted[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[i : j + 1] = avg_rank
        i = j + 1

    sum_ranks_pos = np.sum(ranks[y_true_sorted == 1])
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_score >= thr).astype(np.int64)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def per_class_stats(labels: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for cls in ("landfall", "general_mal", "benign"):
        mask = labels == cls
        if not np.any(mask):
            continue
        cls_scores = scores[mask]
        flagged = int(np.sum(cls_scores >= thr))
        total = int(mask.sum())
        if cls == "benign":
            rate = flagged / max(1, total)
            out[cls] = {
                "total": total,
                "flagged": flagged,
                "fpr": float(rate),
                "score_min": float(cls_scores.min()),
                "score_mean": float(cls_scores.mean()),
                "score_max": float(cls_scores.max()),
            }
        else:
            rate = flagged / max(1, total)
            out[cls] = {
                "total": total,
                "flagged": flagged,
                "recall": float(rate),
                "score_min": float(cls_scores.min()),
                "score_mean": float(cls_scores.mean()),
                "score_max": float(cls_scores.max()),
            }
    return out


def build_autoencoder(input_dim: int, hidden: List[int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for h in hidden:
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    for h in reversed(hidden[:-1]):
        x = tf.keras.layers.Dense(h, activation="relu")(x)
    outputs = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", default="outputs/hybrid_features.npz")
    parser.add_argument("--output-json", default="outputs/ae_metrics.json")
    parser.add_argument("--output-model", default="outputs/ae_model.keras")
    parser.add_argument("--bytes-mode", choices=["raw", "hist"], default="hist")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="0.7,0.15,0.15")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-percentile", type=float, default=99.0)
    parser.add_argument("--hidden", default="256,64")
    args = parser.parse_args()

    splits = tuple(float(x) for x in args.split.split(","))
    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]

    data = np.load(args.input_npz, allow_pickle=True)
    X_bytes = data["X_bytes"]
    X_struct = data["X_struct"]
    labels = data["labels"]
    paths = data["paths"]
    struct_names = [str(x) for x in data["struct_feature_names"]]

    X_bytes, X_struct, labels, paths = deduplicate_by_hash(paths, X_bytes, X_struct, labels)

    X_bytes_feat = magika_bytes_to_features(X_bytes, args.bytes_mode)
    X_struct_feat = build_struct_features(X_struct, struct_names)
    X = np.concatenate([X_bytes_feat, X_struct_feat], axis=1)

    n = X.shape[0]
    train_idx, val_idx, test_idx = split_indices(n, args.seed, splits)

    benign_mask = labels == "benign"
    train_benign_idx = train_idx[benign_mask[train_idx]]
    val_benign_idx = val_idx[benign_mask[val_idx]]

    if train_benign_idx.size == 0:
        raise SystemExit("No benign samples in training split.")

    # Standardize using benign train only
    mean = X[train_benign_idx].mean(axis=0)
    std = X[train_benign_idx].std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    Xn = (X - mean) / std_safe

    model = build_autoencoder(X.shape[1], hidden)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), loss="mse")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    model.fit(
        Xn[train_benign_idx],
        Xn[train_benign_idx],
        validation_data=(Xn[val_benign_idx], Xn[val_benign_idx]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    recon = model.predict(Xn, batch_size=args.batch_size, verbose=0)
    errors = np.mean((Xn - recon) ** 2, axis=1)

    thr = float(np.percentile(errors[val_benign_idx], args.threshold_percentile))

    y_true = np.where(labels == "benign", 0, 1).astype(np.int64)
    val_metrics = metrics_at_threshold(y_true[val_idx], errors[val_idx], thr)
    test_metrics = metrics_at_threshold(y_true[test_idx], errors[test_idx], thr)
    test_metrics["roc_auc"] = roc_auc(y_true[test_idx], errors[test_idx])

    results = {
        "model": "autoencoder",
        "bytes_mode": args.bytes_mode,
        "hidden": hidden,
        "threshold_percentile": args.threshold_percentile,
        "threshold": thr,
        "counts": {
            "total": int(n),
            "train": int(train_idx.size),
            "val": int(val_idx.size),
            "test": int(test_idx.size),
            "train_benign": int(train_benign_idx.size),
            "val_benign": int(val_benign_idx.size),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_per_class": per_class_stats(labels[val_idx], errors[val_idx], thr),
        "test_per_class": per_class_stats(labels[test_idx], errors[test_idx], thr),
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
