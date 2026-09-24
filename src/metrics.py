"""
metrics.py — Métriques d'évaluation du modèle PD et calcul de la perte attendue.
Projet : Credit Risk Scoring.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, recall_score, precision_score, confusion_matrix


def compute_auc(y_true, y_pred_proba) -> float:
    """AUC-ROC du modèle."""
    return roc_auc_score(y_true, y_pred_proba)


def compute_roc_curve(y_true, y_pred_proba):
    """Retourne (fpr, tpr, thresholds) pour tracer la courbe ROC."""
    return roc_curve(y_true, y_pred_proba)


def compute_brier_score(y_true, y_pred_proba) -> float:
    """Brier score : qualité de calibration des probabilités prédites."""
    return brier_score_loss(y_true, y_pred_proba)


def compute_recall(y_true, y_pred_proba, threshold: float = 0.5) -> float:
    """
    Rappel (recall / sensibilité) : proportion de vrais défauts correctement
    détectés par le modèle, à un seuil de décision donné.
    Rappel = VP / (VP + FN).
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return recall_score(y_true, y_pred)


def compute_precision(y_true, y_pred_proba, threshold: float = 0.5) -> float:
    """
    Précision : proportion de prêts classés comme défaut qui le sont
    réellement, à un seuil de décision donné.
    Précision = VP / (VP + FP).
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    return precision_score(y_true, y_pred, zero_division=0)


def confusion_matrix_at_threshold(y_true, y_pred_proba, threshold: float = 0.5) -> np.ndarray:
    """Matrice de confusion à un seuil de décision donné."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def optimal_threshold_by_cost(y_true, y_pred_proba, cost_fn: float, cost_fp: float):
    """
    Détermine le seuil qui minimise le coût métier total.
    cost_fn : coût d'un faux négatif (défaut non détecté)
    cost_fp : coût d'un faux positif (bon client refusé)
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        costs.append(fn * cost_fn + fp * cost_fp)
    best_idx = int(np.argmin(costs))
    return thresholds[best_idx], costs[best_idx]


def expected_loss(pd_pred, lgd_pred, ead) -> pd.Series:
    """Perte attendue par prêt : EL = PD x LGD x EAD."""
    return pd_pred * lgd_pred * ead


def expected_loss_by_segment(df: pd.DataFrame, segment_col: str, el_col: str = "expected_loss"):
    """Agrège l'EL par segment (ex. grade, tranche de montant)."""
    return df.groupby(segment_col)[el_col].agg(["sum", "mean", "count"])