"""Dependency-free classifier over stylometric features."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from ray_unsloth_apps.scribe.profile import FEATURE_NAMES, feature_vector


@dataclass(slots=True)
class StyleClassifier:
    weights: list[float]
    bias: float
    feature_means: list[float]
    feature_stds: list[float]
    feature_names: list[str]

    def predict_proba(self, text: str) -> float:
        vector = _standardize(feature_vector(text), self.feature_means, self.feature_stds)
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, vector, strict=False))
        return _sigmoid(score)


def train_classifier(
    pos_texts: Iterable[str],
    neg_texts: Iterable[str],
    *,
    epochs: int = 200,
    lr: float = 0.1,
    l2: float = 1e-3,
    seed: int = 0,
) -> StyleClassifier:
    texts = [(1.0, str(text)) for text in pos_texts] + [(0.0, str(text)) for text in neg_texts]
    if not texts:
        return StyleClassifier(
            weights=[0.0] * len(FEATURE_NAMES),
            bias=0.0,
            feature_means=[0.0] * len(FEATURE_NAMES),
            feature_stds=[1.0] * len(FEATURE_NAMES),
            feature_names=list(FEATURE_NAMES),
        )

    vectors = [feature_vector(text) for _label, text in texts]
    feature_means = [_mean(column) for column in zip(*vectors, strict=False)]
    feature_stds = [max(_std(column), 1e-3) for column in zip(*vectors, strict=False)]
    standardized = [_standardize(vector, feature_means, feature_stds) for vector in vectors]

    rng = random.Random(seed)
    weights = [rng.uniform(-0.01, 0.01) for _ in FEATURE_NAMES]
    bias = 0.0
    indexed = list(range(len(texts)))

    for _ in range(max(1, epochs)):
        rng.shuffle(indexed)
        grad_w = [0.0] * len(weights)
        grad_b = 0.0
        for index in indexed:
            label, _text = texts[index]
            vector = standardized[index]
            score = bias + sum(weight * value for weight, value in zip(weights, vector, strict=False))
            prob = _sigmoid(score)
            error = prob - label
            for position, value in enumerate(vector):
                grad_w[position] += error * value
            grad_b += error
        scale = 1.0 / len(texts)
        for position in range(len(weights)):
            grad_w[position] = grad_w[position] * scale + l2 * weights[position]
            weights[position] -= lr * grad_w[position]
        bias -= lr * grad_b * scale

    return StyleClassifier(
        weights=weights,
        bias=bias,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_names=list(FEATURE_NAMES),
    )


def auc(pos_texts: Iterable[str], neg_texts: Iterable[str]) -> float:
    pos = [str(text) for text in pos_texts]
    neg = [str(text) for text in neg_texts]
    if not pos or not neg:
        return 0.5
    classifier = train_classifier(pos, neg, epochs=80, lr=0.05, l2=1e-3, seed=0)
    pos_scores = [classifier.predict_proba(text) for text in pos]
    neg_scores = [classifier.predict_proba(text) for text in neg]
    return _rank_auc([(score, 1) for score in pos_scores] + [(score, 0) for score in neg_scores])


def _standardize(vector: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(value - mean_value) / max(std, 1e-3) for value, mean_value, std in zip(vector, means, stds, strict=False)]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _std(values: Iterable[float]) -> float:
    items = list(values)
    avg = sum(items) / len(items)
    return math.sqrt(sum((value - avg) ** 2 for value in items) / len(items))


def _rank_auc(scored: list[tuple[float, int]]) -> float:
    scored.sort(key=lambda item: item[0])
    rank = 1
    total_rank = 0.0
    index = 0
    while index < len(scored):
        end = index + 1
        while end < len(scored) and scored[end][0] == scored[index][0]:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        for position in range(index, end):
            if scored[position][1] == 1:
                total_rank += average_rank
        rank += end - index
        index = end
    n_pos = sum(label for _score, label in scored)
    n_neg = len(scored) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    u = total_rank - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)
