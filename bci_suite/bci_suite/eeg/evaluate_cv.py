"""
Repeated, participant-level cross-validation for the EEG model — replaces
trusting a single train/val/test split's accuracy as THE number.

Direct motivation: three runs against the identical participant split
(differing only in random seed / not seeded at all) produced test
accuracies of 44.4% (pre-leakage-fix), 48.6%, 45.3%, and 30.4% — a 30-49%
range purely from training stochasticity on a 5-participant held-out set.
Reporting any single one of these as "the" accuracy is misleading. This
script reports a mean +/- std across multiple participant-disjoint folds
instead, which is what should actually be cited going forward.

Flagged by external audit (EXTERNAL_REVIEW.md F-007/F-020), and confirmed
necessary by direct observation during remediation (see docs/decisions.md).

Usage:
    python -m bci_suite.eeg.evaluate_cv --n-folds 5 --seeds-per-fold 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from bci_suite.eeg.eegnet import EEGNet
from bci_suite.eeg.loaders import WINDOW_SAMPLES, EEG_CHANNELS, load_dreamer_windows
from shared.valence_arousal import IDX_TO_QUADRANT

RESULTS_PATH = Path("models/eeg/cv_results.json")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _train_one_run(X_train, y_train, X_test, y_test, seed: int) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(4), y=y_train
    )
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=BATCH_SIZE,
    )

    model = EEGNet(n_classes=4, n_channels=len(EEG_CHANNELS), n_timepoints=WINDOW_SAMPLES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    model.train()
    for epoch in range(EPOCHS):
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += xb.size(0)
    return correct / total


def main(n_folds: int, seeds_per_fold: int):
    print("Loading DREAMER windows...")
    X, y, groups = load_dreamer_windows()

    gkf = GroupKFold(n_splits=n_folds)
    all_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_participants = set(groups[train_idx])
        test_participants = set(groups[test_idx])
        assert not (train_participants & test_participants), "Leakage detected — aborting."

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        fold_accs = []
        for seed in range(seeds_per_fold):
            acc = _train_one_run(X_train, y_train, X_test, y_test, seed=seed)
            fold_accs.append(acc)
            print(f"Fold {fold_idx + 1}/{n_folds}, seed {seed}: "
                  f"test_participants={sorted(test_participants)}, acc={acc:.4f}")

        all_results.append({
            "fold": fold_idx,
            "test_participants": sorted(int(p) for p in test_participants),
            "seed_accuracies": fold_accs,
            "fold_mean": float(np.mean(fold_accs)),
            "fold_std": float(np.std(fold_accs)),
        })

    all_accs = [acc for fold in all_results for acc in fold["seed_accuracies"]]
    fold_means = [fold["fold_mean"] for fold in all_results]

    # IMPORTANT: seeds within a fold share the identical held-out participants
    # — they are repeated measurements of one group's difficulty, not
    # independent draws of "a new random set of people." Treating all
    # n_folds*seeds_per_fold runs as independent samples overstates
    # confidence. The statistically correct unit of analysis is the FOLD
    # (n_folds independent samples), using the within-fold seed average to
    # reduce noise in each fold's estimate, not to inflate degrees of
    # freedom. Both are reported; the fold-level one is what should actually
    # be cited/trusted.
    from scipy import stats as scipy_stats
    fold_mean_of_means = float(np.mean(fold_means))
    fold_std = float(np.std(fold_means, ddof=1)) if len(fold_means) > 1 else float("nan")
    t_stat, p_value = scipy_stats.ttest_1samp(fold_means, 0.25) if len(fold_means) > 1 else (float("nan"), float("nan"))

    summary = {
        "n_folds": n_folds,
        "seeds_per_fold": seeds_per_fold,
        "naive_pooled_mean": float(np.mean(all_accs)),
        "naive_pooled_std": float(np.std(all_accs)),
        "naive_pooled_note": (
            "NOT the statistically correct estimate — treats correlated "
            "within-fold seed reruns as independent samples, overstating "
            "confidence. See fold_level_* fields instead."
        ),
        "fold_level_mean": fold_mean_of_means,
        "fold_level_std": fold_std,
        "fold_level_n": n_folds,
        "ttest_vs_chance_t": float(t_stat) if not np.isnan(t_stat) else None,
        "ttest_vs_chance_p": float(p_value) if not np.isnan(p_value) else None,
        "ttest_note": (
            f"One-sample t-test of the {n_folds} independent fold-means "
            f"against chance (0.25), df={n_folds - 1}. p < 0.05 would "
            "indicate a statistically distinguishable-from-chance result — "
            "check this value before citing any accuracy number as 'working'."
        ),
        "overall_min": float(np.min(all_accs)),
        "overall_max": float(np.max(all_accs)),
        "per_fold": all_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Cross-validated result ({n_folds} folds x {seeds_per_fold} seeds = "
          f"{n_folds * seeds_per_fold} runs) ===")
    print(f"Naive pooled (INCORRECT, do not cite): "
          f"{summary['naive_pooled_mean']*100:.1f}% ± {summary['naive_pooled_std']*100:.1f}%")
    print(f"Fold-level (CORRECT unit of analysis, n={n_folds}): "
          f"{fold_mean_of_means*100:.1f}% ± {fold_std*100:.1f}%")
    if p_value is not None and not np.isnan(p_value):
        significant = "YES" if p_value < 0.05 else "NO"
        print(f"t-test vs 25% chance: t={t_stat:.2f}, p={p_value:.3f} "
              f"— statistically distinguishable from chance? {significant}")
    print(f"Range across all runs: [{summary['overall_min']*100:.1f}%, {summary['overall_max']*100:.1f}%]")
    print(f"Saved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5,
                         help="Number of participant-disjoint folds (GroupKFold)")
    parser.add_argument("--seeds-per-fold", type=int, default=3,
                         help="Number of random-seed reruns per fold, to separate "
                              "fold-to-fold variance from seed-to-seed variance")
    args = parser.parse_args()
    main(args.n_folds, args.seeds_per_fold)
