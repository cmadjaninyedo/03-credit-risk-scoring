"""
app.py — Dashboard Streamlit de démonstration
Projet : Credit Risk Scoring (Lending Club)

Lancement : streamlit run app/app.py (depuis la racine du projet)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Racine du projet (le dossier parent de ce fichier)
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

st.set_page_config(
    page_title="Credit Risk Scoring — Lending Club",
    page_icon="💳",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Chargement des artefacts (mis en cache pour ne pas recharger à chaque clic)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    reports = ROOT / "reports"
    artifacts = {
        "pd_model": joblib.load(reports / "model_pd_xgb.pkl"),
        "pd_columns": joblib.load(reports / "feature_columns_pd.pkl"),
        "pd_medians": joblib.load(reports / "feature_medians_pd.pkl"),
        "lgd_model": joblib.load(reports / "model_lgd.pkl"),
        "lgd_columns": joblib.load(reports / "feature_columns_lgd.pkl"),
        "lgd_medians": joblib.load(reports / "feature_medians_lgd.pkl"),
    }
    return artifacts


@st.cache_data
def load_portfolio_data():
    data = ROOT / "data" / "processed"
    reports = ROOT / "reports"
    portfolio = pd.read_parquet(data / "test_final_with_stress.parquet")
    summary = pd.read_csv(reports / "summary_final_metrics.csv")
    walk_forward = None
    wf_path = reports / "walk_forward_results.csv"
    if wf_path.exists():
        walk_forward = pd.read_csv(wf_path)
    return portfolio, summary, walk_forward


def remaining_balance(installment, annual_rate_pct, term_months, months_elapsed):
    """Solde restant dû après `months_elapsed` mensualités (amortissement standard)."""
    r = annual_rate_pct / 100 / 12
    n = term_months
    k = np.clip(months_elapsed, 0, n)
    safe_r = r if r != 0 else 1e-6
    balance = installment * (1 - (1 + safe_r) ** (-(n - k))) / safe_r
    return max(balance, 0)


def build_feature_vector(inputs: dict, columns: list, medians: pd.Series) -> pd.DataFrame:
    """
    Construit un vecteur de features aligné sur les colonnes d'entraînement.
    Part de la médiane du train pour toute colonne non fournie (baseline neutre),
    puis surcharge avec les valeurs saisies dans l'app, y compris les colonnes
    one-hot correspondant aux choix catégoriels.
    """
    row = pd.Series(0.0, index=columns)
    # Valeurs numériques de référence (médiane du train)
    for col in columns:
        if col in medians.index:
            row[col] = medians[col]

    for key, value in inputs.items():
        if key in row.index:
            row[key] = value

    return pd.DataFrame([row])[columns]


def set_dummy(row_dict: dict, columns: list, prefix: str, value: str):
    """Active la colonne one-hot correspondant à `value` pour un préfixe donné
    (ex. prefix='grade', value='C' -> active 'grade_C' si elle existe)."""
    target_col = f"{prefix}_{value}"
    for col in columns:
        if col.startswith(prefix + "_"):
            row_dict[col] = 1.0 if col == target_col else 0.0


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("💳 Credit Risk Scoring — Lending Club")
st.caption(
    "Portfolio Analytics Engineer / Data Analyst — Crespino Marius ADJANINYEDO · "
    "Pipeline PD → LGD → EAD → Perte Attendue, avec stress test."
)

try:
    artifacts = load_artifacts()
    portfolio, summary, walk_forward = load_portfolio_data()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    st.error(
        f"Fichier manquant : {e}. Vérifie que tous les notebooks ont bien été "
        "exécutés jusqu'au bout et que les artefacts sont sauvegardés dans reports/ "
        "et data/processed/ (voir les cellules de sauvegarde ajoutées aux notebooks 02 et 03)."
    )

tab1, tab2, tab3 = st.tabs(["📊 Vue d'ensemble du portefeuille", "🧮 Simulateur de prêt", "🔧 Détails techniques"])

# ---------------------------------------------------------------------------
# Onglet 1 — Vue d'ensemble du portefeuille
# ---------------------------------------------------------------------------
with tab1:
    if artifacts_ok:
        st.subheader("Perte attendue sur le portefeuille test (2017)")

        el_base = portfolio["expected_loss"].sum()
        el_stress = portfolio["expected_loss_stress"].sum()
        total_portfolio = portfolio["funded_amnt"].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portefeuille total", f"{total_portfolio:,.0f} $".replace(",", " "))
        col2.metric("Perte attendue (base)", f"{el_base:,.0f} $".replace(",", " "),
                    f"{el_base / total_portfolio:.1%} du portefeuille")
        col3.metric("Perte attendue (stress)", f"{el_stress:,.0f} $".replace(",", " "),
                    f"+{(el_stress - el_base) / el_base:.1%}")
        col4.metric("Provision supplémentaire", f"{el_stress - el_base:,.0f} $".replace(",", " "),
                    f"+{(el_stress - el_base) / total_portfolio:.2%} du portefeuille")

        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Perte attendue par grade — base vs stress**")
            sensitivity = portfolio.groupby("grade", observed=True).agg(
                el_base=("expected_loss", "sum"),
                el_stress=("expected_loss_stress", "sum"),
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            sensitivity.plot(kind="bar", ax=ax)
            ax.set_ylabel("Perte attendue ($)")
            ax.set_xlabel("Grade")
            plt.xticks(rotation=0)
            st.pyplot(fig)

        with col_right:
            st.markdown("**EL / montant prêté vs taux d'intérêt facturé, par grade**")
            reco = portfolio.groupby("grade", observed=True).agg(
                el_ratio_moyen=("expected_loss", lambda x: (x / portfolio.loc[x.index, "loan_amnt"]).mean()),
                taux_interet_moyen=("int_rate", "mean"),
            )
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(reco.index, reco["el_ratio_moyen"] * 100, marker="o", label="EL / montant prêté (%)")
            ax2.plot(reco.index, reco["taux_interet_moyen"], marker="o", label="Taux d'intérêt moyen (%)")
            ax2.set_xlabel("Grade")
            ax2.set_ylabel("%")
            ax2.legend()
            st.pyplot(fig2)

        st.divider()
        st.markdown("**Métriques de synthèse du projet**")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Charge d'abord tous les artefacts requis (voir message d'erreur ci-dessus).")

# ---------------------------------------------------------------------------
# Onglet 2 — Simulateur de prêt
# ---------------------------------------------------------------------------
with tab2:
    if artifacts_ok:
        st.subheader("Simuler le risque d'un nouveau prêt")
        st.caption(
            "Renseigne les caractéristiques d'un emprunteur pour estimer sa "
            "probabilité de défaut (PD), la perte en cas de défaut (LGD), "
            "l'exposition (EAD) et la perte attendue (EL)."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amnt = st.number_input("Montant du prêt ($)", 1000, 40000, 15000, step=500)
            term = st.selectbox("Durée", ["36 months", "60 months"])
            grade = st.selectbox("Grade Lending Club", ["A", "B", "C", "D", "E", "F", "G"], index=2)
            int_rate = st.slider("Taux d'intérêt (%)", 5.0, 31.0, 14.0, step=0.1)
        with c2:
            annual_inc = st.number_input("Revenu annuel déclaré ($)", 10000, 300000, 60000, step=1000)
            dti = st.slider("DTI — ratio dette/revenu (%)", 0.0, 40.0, 18.0, step=0.5)
            emp_length = st.selectbox(
                "Ancienneté d'emploi",
                ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                 "6 years", "7 years", "8 years", "9 years", "10+ years", "Unknown"],
                index=10,
            )
            home_ownership = st.selectbox("Statut logement", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        with c3:
            fico = st.slider("Score FICO (bas de fourchette)", 640, 845, 700, step=5)
            revol_util = st.slider("Taux d'utilisation du crédit renouvelable (%)", 0.0, 150.0, 40.0, step=1.0)
            inq_last_6mths = st.number_input("Demandes de crédit (6 derniers mois)", 0, 10, 0)
            purpose = st.selectbox(
                "Objet du prêt",
                ["debt_consolidation", "credit_card", "home_improvement", "small_business",
                 "major_purchase", "medical", "other"],
            )

        if st.button("Calculer le risque", type="primary"):
            term_months = int(term.split()[0])

            # --- Vecteur PD ---
            pd_inputs = {
                "loan_amnt": loan_amnt,
                "funded_amnt": loan_amnt,
                "int_rate": int_rate,
                "installment": loan_amnt * (int_rate / 100 / 12) / (1 - (1 + int_rate / 100 / 12) ** -term_months),
                "annual_inc": annual_inc,
                "dti": dti,
                "fico_range_low": fico,
                "fico_range_high": fico + 4,
                "revol_util": revol_util,
                "inq_last_6mths": inq_last_6mths,
                "loan_to_income": loan_amnt / annual_inc,
                "grade_ordinal": ["A", "B", "C", "D", "E", "F", "G"].index(grade),
            }
            set_dummy(pd_inputs, artifacts["pd_columns"], "grade", grade)
            set_dummy(pd_inputs, artifacts["pd_columns"], "term", term)
            set_dummy(pd_inputs, artifacts["pd_columns"], "home_ownership", home_ownership)
            set_dummy(pd_inputs, artifacts["pd_columns"], "purpose", purpose)
            set_dummy(pd_inputs, artifacts["pd_columns"], "emp_length", emp_length)

            X_pd = build_feature_vector(pd_inputs, artifacts["pd_columns"], artifacts["pd_medians"])
            pd_pred = artifacts["pd_model"].predict_proba(X_pd)[:, 1][0]

            # --- Vecteur LGD ---
            lgd_inputs = {
                "loan_amnt": loan_amnt,
                "int_rate": int_rate,
                "term_months": term_months,
                "dti": dti,
                "annual_inc": annual_inc,
                "months_on_book_default": artifacts["lgd_medians"].get("months_on_book_default", 10),
            }
            set_dummy(lgd_inputs, artifacts["lgd_columns"], "grade", grade)
            set_dummy(lgd_inputs, artifacts["lgd_columns"], "home_ownership", home_ownership)
            set_dummy(lgd_inputs, artifacts["lgd_columns"], "purpose", purpose)

            X_lgd = build_feature_vector(lgd_inputs, artifacts["lgd_columns"], artifacts["lgd_medians"])
            if "const" in artifacts["lgd_columns"]:
                X_lgd["const"] = 1.0
            lgd_pred = float(artifacts["lgd_model"].predict(X_lgd).iloc[0])

            # --- EAD (amortissement au délai moyen du grade) ---
            avg_months = artifacts["lgd_medians"].get("months_on_book_default", 10)
            installment_val = pd_inputs["installment"]
            ead = remaining_balance(installment_val, int_rate, term_months, avg_months)

            expected_loss = pd_pred * lgd_pred * ead

            st.divider()
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Probabilité de défaut (PD)", f"{pd_pred:.1%}")
            r2.metric("Perte en cas de défaut (LGD)", f"{lgd_pred:.1%}")
            r3.metric("Exposition estimée (EAD)", f"{ead:,.0f} $".replace(",", " "))
            r4.metric("Perte attendue (EL)", f"{expected_loss:,.0f} $".replace(",", " "))

            el_ratio = expected_loss / loan_amnt
            st.progress(min(el_ratio, 1.0), text=f"EL / montant prêté : {el_ratio:.1%}")

            if el_ratio > (int_rate / 100):
                st.warning(
                    "⚠️ La perte attendue rapportée au montant dépasse le taux d'intérêt "
                    "facturé — ce profil serait structurellement risqué à ce niveau de "
                    "tarification (comparaison simplifiée, voir limites du README)."
                )
            else:
                st.success(
                    "✅ La perte attendue rapportée au montant reste inférieure au taux "
                    "d'intérêt facturé sur ce profil."
                )
    else:
        st.info("Charge d'abord tous les artefacts requis (voir message d'erreur ci-dessus).")

# ---------------------------------------------------------------------------
# Onglet 3 — Détails techniques
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Comparaison des modèles PD")
    st.markdown(
        """
        | Modèle | AUC-ROC | Average Precision | Brier score |
        |---|---|---|---|
        | Régression logistique | 0,7042 | 0,3959 | 0,1633 |
        | Random Forest | 0,7003 | 0,3961 | 0,2190 |
        | **XGBoost (retenu)** | **0,7121** | **0,4118** | **0,1607** |
        """
    )

    if artifacts_ok and walk_forward is not None:
        st.subheader("Stabilité temporelle du modèle (walk-forward validation)")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(walk_forward["test_annee"], walk_forward["auc"], marker="o")
        ax3.set_xlabel("Année de test")
        ax3.set_ylabel("AUC-ROC")
        ax3.set_ylim(0.6, 0.8)
        st.pyplot(fig3)

    st.subheader("Figures sauvegardées du pipeline")
    figures_dir = ROOT / "reports" / "figures"
    if figures_dir.exists():
        fig_files = sorted(figures_dir.glob("*.png"))
        if fig_files:
            selected = st.selectbox("Choisir une figure", [f.name for f in fig_files])
            st.image(str(figures_dir / selected), use_container_width=True)
        else:
            st.caption("Aucune figure trouvée dans reports/figures/.")

    st.divider()
    st.caption(
        "⚠️ Cette application est une démonstration pédagogique. Le stress test, "
        "l'EAD par amortissement et le modèle LGD comportent des simplifications "
        "explicitement documentées dans le README du projet."
    )
