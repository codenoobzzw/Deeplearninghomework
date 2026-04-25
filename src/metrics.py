from __future__ import annotations

import numpy as np



def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())



def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix



def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm)
    denom = cm.sum(axis=1)
    result = np.zeros(cm.shape[0], dtype=np.float32)
    non_zero = denom > 0
    result[non_zero] = cm.diagonal()[non_zero] / denom[non_zero]
    return result



def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)



def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    probs = softmax(logits)
    eps = 1e-12
    loss = -np.log(probs[np.arange(targets.shape[0]), targets] + eps).mean()
    return float(loss)
