"""
data_prep.py — Fonctions de chargement et nettoyage des données Lending Club.
Projet : Credit Risk Scoring.
"""

import pandas as pd

# Colonnes à exclure de la modélisation PD : connues seulement après l'octroi
# (data leakage). Voir le guide du projet pour la justification détaillée.
LEAKAGE_COLUMNS = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "out_prncp", "out_prncp_inv", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_fico_range_high", "last_fico_range_low",
]

# Colonnes retenues, connues au moment de la décision d'octroi
FEATURE_COLUMNS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "dti", "purpose", "fico_range_low",
    "fico_range_high", "open_acc", "total_acc", "revol_bal", "revol_util",
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "earliest_cr_line",
    "issue_d", "loan_status",
]

# Statuts de prêt considérés comme "soldés" (issue connue)
CLOSED_STATUSES = ["Fully Paid", "Charged Off", "Default"]
DEFAULT_STATUSES = ["Charged Off", "Default"]

# Colonnes utiles à charger dès le départ (features + colonnes nécessaires pour LGD/EAD)
COLUMNS_TO_LOAD = FEATURE_COLUMNS + [
    "recoveries", "collection_recovery_fee", "total_pymnt", "total_rec_prncp",
]


def load_and_filter(path: str, year_min: int = 2007, year_max: int = 2017,
                     chunksize: int = 200_000) -> pd.DataFrame:
    """
    Charge le CSV Lending Club par blocs et filtre chaque bloc immédiatement
    (prêts soldés + fenêtre d'années) pour éviter de saturer la mémoire.
    """
    dtype_map = {
        "term": "category", "grade": "category", "sub_grade": "category",
        "emp_length": "category", "home_ownership": "category",
        "verification_status": "category", "purpose": "category",
        "loan_status": "category",
    }

    chunks_filtered = []
    reader = pd.read_csv(
        path,
        usecols=COLUMNS_TO_LOAD,
        dtype=dtype_map,
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["loan_status"].isin(CLOSED_STATUSES)]
        chunk["issue_d"] = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        chunk = chunk[
            (chunk["issue_d"].dt.year >= year_min) & (chunk["issue_d"].dt.year <= year_max)
        ]
        chunks_filtered.append(chunk)
        print(f"Bloc {i+1} traité — lignes conservées cumulées : "
              f"{sum(len(c) for c in chunks_filtered):,}".replace(",", " "))

    return pd.concat(chunks_filtered, ignore_index=True)


def filter_closed_loans(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les prêts avec une issue connue (exclut 'Current' etc.)."""
    return df[df["loan_status"].isin(CLOSED_STATUSES)].copy()


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne cible binaire 'default'."""
    df = df.copy()
    df["default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les colonnes autorisées pour la modélisation PD (anti-leakage)."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns] + ["default"]
    return df[cols].copy()