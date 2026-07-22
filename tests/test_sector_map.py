"""Unit tests for marketlab.sector_map (the parts that don't need the network)."""

import numpy as np

from marketlab.sector_map import pca_2d, sector_similarity_test


def test_pca_2d_shapes_and_variance_order():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 16)) * np.linspace(5, 0.1, 16)  # decaying scales
    proj, explained = pca_2d(x)
    assert proj.shape == (200, 2)
    assert explained.shape == (2,)
    assert explained[0] >= explained[1] > 0
    assert explained.sum() <= 1.0


def test_sector_similarity_detects_real_clusters():
    rng = np.random.default_rng(1)
    centers = rng.normal(0, 1, (4, 16))
    labels = np.repeat(np.arange(4), 30).astype(str)
    emb = centers[np.repeat(np.arange(4), 30)] + rng.normal(0, 0.3, (120, 16))
    diff, p = sector_similarity_test(emb, labels, n_perm=200, seed=2)
    assert diff > 0.1
    assert p < 0.01


def test_sector_similarity_null_on_random_labels():
    rng = np.random.default_rng(3)
    emb = rng.normal(0, 1, (120, 16))
    labels = rng.choice(list("ABCD"), 120)
    _, p = sector_similarity_test(emb, labels, n_perm=200, seed=4)
    assert p > 0.05
