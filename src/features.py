"""
features.py — Feature engineering pour le modèle PD.
Projet : Credit Risk Scoring.
"""

import pandas as pd
import numpy as np

# Ordre fixe des grades Lending Club (A = meilleur risque, G = plus risqué).
# Figé volontairement plutôt que déduit des données reçues, pour garantir un
# encodage identique quel que soit le sous-ensemble passé à encode_grade
# (train, test, ou un sous-échantillon comme les prêts en défaut).
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}


def add_credit_age(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule l'ancienneté du dossier de crédit en années à partir de earliest_cr_line."""
    df = df.copy()
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
    df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    df["credit_age_years"] = (
        (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
    )
    return df


def add_loan_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """Ratio montant du prêt / revenu annuel."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    return df


def encode_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Encode 'grade' (A-G) en ordinal fixe, car il représente déjà une
    hiérarchie de risque. L'ordre est figé (GRADE_ORDER), pas déduit des
    données reçues, pour garantir un encodage identique quel que soit le
    sous-ensemble passé en argument."""
    df = df.copy()
    df["grade_ordinal"] = df["grade"].map(GRADE_ORDER)
    return df


def add_term_months(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait la durée du prêt en mois depuis la colonne term (ex. '36 months' -> 36)."""
    df = df.copy()
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)
    return df


def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encoding pour les variables catégorielles nominales."""
    return pd.get_dummies(df, columns=columns, drop_first=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applique l'ensemble du pipeline de feature engineering."""
    df = add_credit_age(df)
    df = add_loan_to_income(df)
    df = encode_grade(df)
    df = add_term_months(df)
    df = one_hot_encode(df, ["purpose", "home_ownership", "verification_status"])
    return df