# Projet : Credit Risk Scoring (Lending Club)

Portfolio Analytics Engineer / Data Analyst — Crespino Marius ADJANINYEDO

## 1. Contexte et objectif

Ce projet construit un pipeline complet de scoring de risque de crédit à l'échelle
bancaire : probabilité de défaut (PD), perte en cas de défaut (LGD), exposition
au moment du défaut (EAD), et perte attendue (Expected Loss, EL) — complété par
un stress test simplifié et des recommandations d'octroi par segment de risque.

**Changement de dataset par rapport au cahier des charges initial.** Le German
Credit Data (UCI, 1 000 lignes) proposé à l'origine a été écarté au profit du
dataset Lending Club (prêts accordés, 2007-2018) pour deux raisons :
- **Volume** : German Credit Data ne permet pas de démontrer un pipeline à
  l'échelle d'un cas d'usage bancaire réel.
- **Richesse des variables financières** : contrairement à German Credit Data,
  Lending Club fournit les montants réellement remboursés, les recouvrements et
  les frais de recouvrement — ce qui permet de calculer une **LGD et une EAD
  empiriques réelles**, plutôt que de les poser comme hypothèses forfaitaires
  (l'approche que German Credit Data aurait imposée).

## 2. Données

- **Source** : Kaggle, "All Lending Club loan data" (prêts acceptés, 2007-2018)
- **Périmètre retenu** : prêts émis entre 2007 et 2017, avec une issue connue
  (Fully Paid, Charged Off, Default — les prêts encore en cours, "Current",
  sont exclus). 2018 a été volontairement écarté pour limiter l'effet de
  censure à droite sur les cohortes trop récentes (voir section 5).
- **Chargement** : le fichier brut pèse ~1,6 Go ; il est lu et filtré par blocs
  de 200 000 lignes pour rester dans les limites mémoire d'un poste de travail
  standard, puis mis en cache au format Parquet.
- **Volumétrie finale** : **1 289 032 prêts**, taux de défaut global **20,15%**
- **Cible** : `default = 1` si le prêt est Charged Off ou Default, `0` si Fully Paid
- **Split train/test** : chronologique (train ≤ 2016, 1 119 711 prêts, défaut
  19,70% ; test 2017, 169 321 prêts, défaut 23,13%) — un split aléatoire
  aurait été moins réaliste pour un usage en production, où un modèle est
  toujours évalué sur des données futures inconnues à l'entraînement.

## 3. Méthodologie

### 3.1 Anti-data-leakage
Les colonnes connues seulement après l'octroi (`total_pymnt`, `recoveries`,
`out_prncp`, dates et montants de paiement, statuts de renégociation) sont
explicitement exclues du jeu de features PD. Une seconde fuite de données plus
subtile a été identifiée et corrigée en cours de projet : un proxy du délai
avant défaut basé sur `total_pymnt / installment` contaminait indirectement le
modèle LGD, puisque `total_pymnt` entre aussi dans le calcul de la cible LGD.
Il a été remplacé par un rechargement ciblé de la vraie variable `last_pymnt_d`.

### 3.2 Modélisation PD
Trois modèles comparés sur trois métriques complémentaires (AUC-ROC, Average
Precision, Brier score) :

| Modèle | AUC-ROC | Average Precision | Brier score |
|---|---|---|---|
| Régression logistique | 0,7042 | 0,3959 | 0,1633 |
| Random Forest | 0,7003 | 0,3961 | 0,2190 |
| **XGBoost (retenu)** | **0,7121** | **0,4118** | **0,1607** |

XGBoost l'emporte sur les trois métriques simultanément et est retenu comme
modèle final. Random Forest, malgré un AUC comparable à la régression
logistique, a un Brier score nettement dégradé — effet du paramètre
`class_weight="balanced"`, qui améliore la détection de la classe minoritaire
au prix de la calibration des probabilités.

**Seuil de décision** : optimisé sur un coût métier (coût d'un défaut non
détecté 5 fois supérieur au coût d'un bon client refusé) plutôt que sur le
seuil par défaut de 0,5, donnant un seuil optimal de **0,140**.

**Validation temporelle (walk-forward)** : le modèle a été ré-entraîné et
testé sur 4 fenêtres glissantes successives (2014 à 2017). L'AUC reste stable
entre 0,709 et 0,733, sans dérive de performance dans le temps.

**Analyse complémentaire — apport de la politique d'octroi Lending Club** :
en retirant `grade`, `sub_grade`, `int_rate` et `installment` (variables liées
à la décision d'octroi propriétaire de Lending Club), l'AUC ne baisse que de
0,7121 à 0,6998 (écart de 0,0124). Les caractéristiques propres de
l'emprunteur (ancienneté, revenu, historique de crédit) portent donc déjà une
grande partie du signal prédictif, indépendamment de toute information du
prêteur.

### 3.3 Modélisation LGD
Régression quasi-binomiale à lien logit (adaptée à une proportion bornée entre
0 et 1), calculée uniquement sur les prêts en défaut du jeu de test.

- **Version par grade seul** : pseudo R² de 0,0094 — proche d'une simple
  moyenne par grade (gain de seulement 0,23% par rapport à cette moyenne
  naïve)
- **Version enrichie** (ajout du délai avant défaut, de l'ancienneté du
  crédit, du statut de propriété, de l'objectif du prêt, du DTI et du
  revenu) : pseudo R² de 0,0327, gain de **21,4%** par rapport à la moyenne
  par grade — la variable la plus significative étant le délai avant défaut
  (`months_on_book_default`, p < 0,001)

### 3.4 EAD
Deux approches comparées :
- **V1 (retenue comme référence prudente)** : EAD = montant financé initial
  (`funded_amnt`) — hypothèse la plus conservatrice pour un prêt amortissable
  classique, sans risque de tirage supplémentaire contrairement à une ligne
  de crédit renouvelable
- **V2 (approche enrichie, retenue pour les résultats finaux)** : EAD estimée
  par la formule d'amortissement standard, appliquée au délai moyen avant
  défaut de chaque grade — validée par comparaison à l'EAD empirique observée
  sur les prêts en défaut (écart de -0,7% à -1,9% selon le grade)

### 3.5 Perte attendue et stress test
```
EL = PD × LGD × EAD
```
calculée prêt par prêt sur le portefeuille test, puis agrégée par segment.

**Stress test** : choc simplifié et différencié sur la PD (non calibré sur une
récession historique réelle, assumé comme limite) — +3 points pour les grades
A à C, +7 points pour D-E, +12 points pour F-G, reflétant une sensibilité
accrue des profils les plus fragiles à une dégradation macroéconomique.

## 4. Résultats

### Perte attendue (EL), version finale (EAD par amortissement)
- **EL de base** : 363 312 095 $ — soit **15,00%** du portefeuille test
  (2 421 502 464 $)
- **EL sous stress** : 431 421 960 $ (+18,7%)
- **Provision supplémentaire nécessaire sous stress** : +2,81% du portefeuille

Un ratio EL/portefeuille de 15% peut sembler élevé comparé à une banque
généraliste (typiquement 1-3%) — c'est cohérent avec le profil du dataset :
crédit à la consommation **non garanti** (sans collatéral), avec un taux de
défaut de base élevé (~20%) et une LGD élevée (~78,6% en moyenne), profil de
risque nettement supérieur à un portefeuille bancaire diversifié incluant du
crédit immobilier garanti.

### Backtesting — l'étape de validation la plus importante
| Version EAD | Écart EL prédite vs perte observée |
|---|---|
| V1 (funded_amnt) | +6,6% (surestimation) |
| V2 (amortissement) | -10,9% (sous-estimation) |

Les deux versions encadrent la perte réelle sans se tromper de plus de 11% —
un résultat solide pour un projet de portfolio. Le passage d'une surestimation
à une sous-estimation illustre un compromis méthodologique instructif : l'EAD
par amortissement améliore la précision individuelle de l'exposition, mais en
appliquant un délai moyen de défaut par grade à l'ensemble du portefeuille
(y compris aux prêts qui ne feront jamais défaut), elle sous-pondère
systématiquement l'exposition des défauts précoces — qui conservent, au
moment de l'incident, un solde restant dû plus élevé que la moyenne du
segment.

### Sensibilité au stress test, par grade
| Grade | EL de base | EL sous stress | Hausse relative |
|---|---|---|---|
| A | 12 552 381 $ | 19 331 088 $ | +54,0% |
| B | 55 265 837 $ | 67 055 383 $ | +21,3% |
| C | 126 496 080 $ | 142 007 878 $ | +12,3% |
| D | 81 278 658 $ | 98 550 276 $ | +21,2% |
| E | 48 387 456 $ | 56 228 365 $ | +16,2% |
| F | 23 618 896 $ | 29 002 715 $ | +22,8% |
| G | 15 712 788 $ | 19 246 255 $ | +22,5% |

### Recommandations d'octroi — EL/montant prêté vs taux d'intérêt facturé
| Grade | EL/montant | Taux d'intérêt moyen | Marge apparente |
|---|---|---|---|
| A | 3,62% | 6,99% | +3,37% |
| B | 8,60% | 10,60% | +2,00% |
| C | 14,81% | 14,27% | -0,54% |
| D | 19,94% | 18,76% | -1,18% |
| E | 26,34% | 24,71% | -1,63% |
| F | 32,92% | 29,78% | -3,14% |
| G | 35,00% | 30,88% | -4,12% |

**Lecture prudente** : cette "marge apparente" compare un taux d'intérêt
**annuel** à une perte rapportée au **principal** (non annualisée) — une
simplification pédagogique, pas un calcul de rentabilité réelle (qui
nécessiterait d'intégrer la durée du prêt, le coût du capital et les coûts
opérationnels). À ce niveau de lecture simplifié, les grades C à G affichent
une marge apparente négative, qui s'aggrave nettement sur les grades les plus
risqués (F, G) — ce sont les premiers segments à surveiller ou à retarifer en
cas de dégradation macroéconomique confirmée.

## 5. Limites et pistes d'amélioration

- **Biais de sélection** : le dataset ne contient que des prêts déjà accordés
  par Lending Club — aucune information sur les emprunteurs refusés à
  l'octroi.
- **Censure à droite confirmée empiriquement** : le taux de défaut du jeu de
  test (2017, 23,13%) est supérieur à celui du train (19,70%) — les prêts
  encore en cours au moment de la compilation des données sont exclus,
  alors que les défauts précoces des cohortes récentes ont déjà eu le temps
  d'apparaître.
- **Modèle LGD au pouvoir explicatif modeste** : même dans sa version
  enrichie et corrigée du leakage, le pseudo R² reste faible (0,0327) — la
  LGD réelle dépend largement de facteurs non observés dans ce dataset
  (comportement de l'emprunteur après le premier impayé, efficacité du
  recouvrement).
- **EAD par amortissement introduit un nouveau biais** : sous-pondération de
  l'exposition des défauts précoces (détaillé en section 4).
- **Stress test simplifié** : choc arbitraire sur la PD, non calibré sur une
  récession historique réelle, sans impact modélisé sur la LGD (le
  recouvrement est généralement plus difficile en période de ralentissement
  économique).
- **"Marge apparente" simplifiée** : comparaison d'un taux annuel à une perte
  non annualisée, à ne pas interpréter comme un calcul de rentabilité réel.
- **Validation temporelle limitée** : le walk-forward validation confirme la
  stabilité du modèle PD jusqu'en 2017, mais le dataset ne permet pas de
  tester sa performance sur des cohortes postérieures.

## Structure du dépôt

```
data/          # données brutes (non versionnées) et traitées
notebooks/     # 01 exploration, 02 PD, 03 LGD/EAD/EL, 04 stress test
src/           # fonctions réutilisables (nettoyage, features, métriques)
reports/       # figures exportées, modèles sauvegardés
```

## Installation

```bash
pip install -r requirements.txt
```