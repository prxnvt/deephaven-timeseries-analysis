"""Unit tests for marketlab.discriminator."""

import numpy as np
import pytest

from marketlab.discriminator import auc_score, train_discriminator


def test_auc_hand_example():
    # pos all above neg -> AUC 1. pos=[1,3] vs neg=[2,4]: of the four ordered
    # pairs only (3,2) is a win -> AUC 1/4.
    assert auc_score(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc_score(np.array([1.0, 3.0]), np.array([2.0, 4.0])) == pytest.approx(0.25)


def test_separable_classes_get_high_accuracy():
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 0.01, (800, 60))
    synth = rng.normal(0.0, 0.03, (800, 60))  # obviously hotter vol
    result = train_discriminator(real, synth, epochs=15, seed=1)
    assert result["accuracy"] > 0.9
    assert result["auc"] > 0.95


def test_identical_distributions_near_chance():
    rng = np.random.default_rng(2)
    real = rng.normal(0.0, 0.02, (800, 60))
    synth = rng.normal(0.0, 0.02, (800, 60))
    result = train_discriminator(real, synth, epochs=15, seed=3)
    assert 0.4 < result["accuracy"] < 0.6
    assert 0.4 < result["auc"] < 0.6
