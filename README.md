# Projet : Credit Risk Scoring (Lending Club)

Portfolio Analytics Engineer / Data Analyst
## 1. Contexte et objectif

Ce projet construit un pipeline complet de scoring de risque de crédit à l'échelle
bancaire : probabilité de défaut (PD), perte en cas de défaut (LGD), exposition
au moment du défaut (EAD), et perte attendue (Expected Loss, EL) complété par
un stress test simplifié et des recommandations d'octroi par segment de risque.

**Changement de dataset par rapport au cahier des charges initial.** Le German
Credit Data (UCI, 1 000 lignes) proposé à l'origine a été écarté au profit du
dataset Lending Club (prêts accordés, 2007-2018) pour deux raisons :
- **Volume** : German Credit Data ne permet pas de démontrer un pipeline à
  l'échelle d'un cas d'usage bancaire réel.
- **Richesse des variables financières** : contrairement à German Credit Data,
  Lending Club fournit les montants réellement remboursés, les recouvrements et
  les frais de recouvrement. Ce qui permet de calculer une **LGD et une EAD
  empiriques réelles**, plutôt que de les poser comme hypothèses forfaitaires
  (l'approche que German Credit Data aurait imposée).

## 2. Données

- **Source** : Kaggle, "All Lending Club loan data" (prêts acceptés, 2007-2018)
- **Périmètre retenu** : prêts émis entre 2007 et 2017, avec une issue connue
  (Fully Paid, Charged Off, Default et les prêts encore en cours, "Current",
  sont exclus). 2018 a été volontairement écarté pour limiter l'effet de
  censure à droite sur les cohortes trop récentes (voir section 5).
- **Chargement** : le fichier brut pèse ~1,6 Go ; il est lu et filtré par blocs
  de 200 000 lignes pour rester dans les limites mémoire d'un poste de travail
  standard, puis mis en cache au format Parquet.
- **Volumétrie finale** : **1 289 032 prêts**
- **Cible** : `default = 1` si le prêt est Charged Off ou Default, `0` si Fully Paid
- **Split train / validation / test** : chronologique en 3 ensembles distincts,
  pour permettre de choisir le modèle et le seuil de décision sans jamais
  regarder le jeu final :
  - **Train (fit)** : 2007-2015, 826 606 prêts, défaut 18,43%
  - **Validation** : 2016, 293 105 prêts, défaut 23,29%
  - **Test (évaluation finale, une seule fois)** : 2017, 169 321 prêts, défaut 23,13%

  Un split chronologique (plutôt qu'aléatoire) est plus réaliste pour un usage
  en production, où un modèle est toujours évalué sur des données futures
  inconnues à l'entraînement.

## 3. Méthodologie

### 3.1 Anti-data-leakage

Les colonnes connues seulement après l'octroi (`total_pymnt`, `recoveries`,
`out_prncp`, dates et montants de paiement, statuts de renégociation) sont
explicitement exclues du jeu de features PD.

**Quatre fuites de données ont été identifiées et corrigées au cours du
projet**, la dernière lors d'un audit méthodologique approfondi :
1. Un proxy du délai avant défaut basé sur `total_pymnt / installment`
   contaminait indirectement le modèle LGD, puisque `total_pymnt` entre aussi
   dans le calcul de la cible LGD il a été remplacé par un rechargement ciblé de la
   vraie variable `last_pymnt_d`.
2. **Le modèle LGD était entraîné ET évalué sur les mêmes prêts en défaut du
   portefeuille test**. Les métriques de validation n'étaient donc pas une
   mesure hors échantillon valide.
3. **Le seuil de décision PD était optimisé directement sur le jeu de test**
   (le même qui servait à présenter la matrice de confusion finale), au lieu
   d'un jeu de validation distinct.
4. **Le délai moyen avant défaut utilisé pour calibrer l'EAD** était lui aussi
   calculé sur les défauts du test, la même circularité que le point 2.

La correction de ces trois derniers points a nécessité l'introduction d'un
véritable split **train / validation / test** (section 2) : le choix du
modèle et du seuil se fait désormais sur la validation, le test n'étant
utilisé qu'une seule fois, à la toute fin, pour chaque pipeline (PD, puis
LGD/EAD).

### 3.2 Modélisation PD

Trois modèles comparés sur trois métriques complémentaires (AUC-ROC, Average
Precision, Brier score), **sur la validation** (2016) :

| Modèle | AUC-ROC | Average Precision | Brier score |
|---|---|---|---|
| Régression logistique | 0,7087 | 0,4145 | 0,1647 |
| Random Forest | 0,7066 | 0,4134 | 0,2113 |
| **XGBoost (retenu)** | **0,7149** | **0,4256** | **0,1629** |

XGBoost l'emporte sur les trois métriques simultanément et est retenu comme
modèle final. Random Forest, malgré un AUC comparable à la régression
logistique, a un Brier score nettement dégradé un effet du paramètre
`class_weight="balanced"`, qui améliore la détection de la classe minoritaire
au prix de la calibration des probabilités.

**Seuil de décision** : optimisé sur un coût métier (coût d'un défaut non
détecté 5 fois supérieur au coût d'un bon client refusé), **déterminé sur la
validation**, donnant un seuil optimal de **0,120**.

**Évaluation finale, une seule fois, sur le test (2017)** modèle et seuil
désormais figés sans avoir jamais regardé le test :

| Métrique | Validation (sélection) | **Test (évaluation finale)** |
|---|---:|---:|
| AUC-ROC | 0,7149 | **0,7073** |
| Average Precision | 0,4256 | **0,4039** |
| Brier score | 0,1629 | **0,1640** |
| Rappel au seuil optimal | — | **85,29%** |
| Précision au seuil optimal | — | **30,42%** |

L'écart entre validation et test reste faible (AUC -0,0076), signe que le
modèle généralise raisonnablement bien et n'était pas fortement surajusté
même si seule cette mesure finale, sur un jeu jamais utilisé pour le choix du
modèle ou du seuil, fait foi.

**Validation temporelle (walk-forward)** : le modèle a été ré-entraîné et
testé sur 4 fenêtres glissantes successives (2014 à 2017). L'AUC reste stable
entre 0,709 et 0,733, sans dérive de performance dans le temps.

**Analyse complémentaire - apport de la politique d'octroi Lending Club** :
en retirant `grade`, `sub_grade` et `int_rate` (variables liées à la décision
d'octroi propriétaire de Lending Club), évalué sur la validation, l'AUC ne
baisse que de 0,7149 à 0,7002 (écart de 0,0147). Les caractéristiques propres
de l'emprunteur (ancienneté, revenu, historique de crédit) portent donc déjà
une grande partie du signal prédictif, indépendamment de toute information du
prêteur.

### 3.3 Modélisation LGD

Régression quasi-binomiale à lien logit (adaptée à une proportion bornée entre
0 et 1), **entraînée sur le train (fit, ≤2015)**, comparée et choisie **sur
la validation** (2016) : jamais sur le test.

- **Version par grade seul (v1)** : pseudo R² de 0,0075 en échantillon (train).
  **Sur la validation (hors échantillon), ce modèle ne bat pas une simple
  moyenne par grade** : MAE 0,2058 contre 0,1995 pour la moyenne naïve, soit
  un résultat **3,11% moins bon**. Un modèle de régression à une seule
  variable catégorielle n'apporte donc rien de plus qu'une moyenne de groupe.
- **Version enrichie (v2)** : ajout du délai avant défaut
  (`months_on_book_default`), de l'ancienneté du crédit, du statut de
  propriété, de l'objectif du prêt, du DTI et du revenu. Pseudo R² de 0,2509
  en échantillon. **Sur la validation, MAE de 0,0976** avec un gain de ~53% par
  rapport à la moyenne naïve et de ~53% par rapport à v1. Les variables les
  plus significatives sont `months_on_book_default` et `term_months`
  (p < 0,001), suivies de `int_rate` (p < 0,001).
- **C'est v2 qui est retenu comme modèle final**, sur la base de cette
  comparaison hors échantillon pas par défaut.

### 3.4 EAD

Deux approches comparées :
- **V1 (référence prudente)** : EAD = montant financé initial (`funded_amnt`,
  moyenne 14 301 $ sur le test) l'hypothèse la plus conservatrice pour un prêt
  amortissable classique, sans risque de tirage supplémentaire contrairement à
  une ligne de crédit renouvelable.
- **V2 (retenue pour les résultats finaux)** : EAD estimée par la formule
  d'amortissement standard, appliquée au délai moyen avant défaut de chaque
  grade. **Ce délai moyen est calibré exclusivement sur les défauts du train**
  (jamais sur le test), puis appliqué au portefeuille test une moyenne de 8 688 $.
  La formule d'amortissement elle-même est validée sur le train, où EAD
  théorique et EAD empirique (calculées sur le délai réel propre à chaque
  prêt en défaut) concordent à moins de 3,2% près sur tous les grades.

### 3.5 Perte attendue et stress test
```
EL = PD × LGD × EAD
```
calculée prêt par prêt sur le portefeuille test (EAD v2, LGD v2, PD du modèle
final), puis agrégée par segment.

**Stress test** : choc simplifié et différencié sur la PD (non calibré sur une
récession historique réelle, noté comme limite) donnant +3 points pour les grades
A à C, +7 points pour D-E, +12 points pour F-G, reflétant une sensibilité
accrue des profils les plus fragiles à une dégradation macroéconomique.

## 4. Résultats

### Perte attendue (EL)
- **EL de base** : 163 627 776 $  soit **6,76%** du portefeuille test
  (2 421 502 464 $)
- **EL sous stress** : 197 005 904 $ (+20,4%, soit +33 378 128 $)
- **Provision supplémentaire nécessaire sous stress** : +1,38% du portefeuille

### Backtesting — l'étape de validation la plus importante

| | Perte attendue (EL) prédite | Perte réellement observée | Écart |
|---|---:|---:|---:|
| Portefeuille test (global) | 163 627 776 $ | 407 594 144 $ | **-59,9%** |

Par grade, l'écart reste remarquablement uniforme :

| Grade | EL prédite | Perte observée | Écart |
|---|---:|---:|---:|
| A | 5 911 439 $ | 17 247 822 $ | -65,7% |
| B | 27 467 177 $ | 67 300 120 $ | -59,2% |
| C | 62 757 125 $ | 146 041 232 $ | -57,0% |
| D | 35 405 006 $ | 91 371 472 $ | -61,3% |
| E | 17 730 792 $ | 48 522 720 $ | -63,5% |
| F | 8 481 512 $ | 21 996 866 $ | -61,4% |
| G | 5 874 726 $ | 15 113 909 $ | -61,1% |

**Cet écart systématique et quasi uniforme n'est pas un défaut de calibration
du modèle, mais un biais de censure à droite, confirmé empiriquement** (détail
en section 5) : le délai réel avant défaut des prêts 2017 (9,84 mois en
moyenne) est près de deux fois plus court que celui calibré sur le train
(18,84 mois), et ce quasi uniformément sur les 7 grades (écart de -8,6 à -9,4
mois partout). Moins de temps s'étant écoulé avant le défaut, moins de
capital a été remboursé. L'exposition réelle au moment du défaut est donc
mécaniquement plus élevée que ce que prédit un modèle calibré sur une
cohorte plus mature. Recalibrer ce délai sur le test aurait réintroduit
exactement la fuite de données corrigée en section 3.1 (point 4). Ce biais
est documenté comme une limite plutôt que masqué par un ajustement
illégitime.

### Sensibilité au stress test, par grade

| Grade | EL de base | EL sous stress | Hausse relative |
|---|---:|---:|---:|
| A | 5 911 439 $ | 9 479 923 $ | +60,4% |
| B | 27 467 177 $ | 33 784 174 $ | +23,0% |
| C | 62 757 125 $ | 71 238 214 $ | +13,5% |
| D | 35 405 006 $ | 43 739 115 $ | +23,5% |
| E | 17 730 792 $ | 20 880 992 $ | +17,8% |
| F | 8 481 512 $ | 10 569 919 $ | +24,6% |
| G | 5 874 726 $ | 7 313 568 $ | +24,5% |

Le grade A affiche la plus forte hausse *relative* : sa PD de base étant très
faible (5,07%), un choc de +3 points la fait presque proportionnellement
exploser, alors que les grades F/G, déjà à PD élevée, absorbent le même choc
en points avec un impact relatif moindre.

### Recommandations d'octroi — EL/montant prêté vs taux d'intérêt facturé

| Grade | EL/montant | Taux d'intérêt moyen | Marge apparente |
|---|---:|---:|---:|
| A | 1,72% | 6,99% | +5,27% |
| B | 4,04% | 10,60% | +6,56% |
| C | 6,91% | 14,27% | +7,36% |
| D | 8,29% | 18,76% | +10,47% |
| E | 9,19% | 24,71% | +15,52% |
| F | 11,54% | 29,78% | +18,24% |
| G | 12,96% | 30,88% | +17,92% |

**Lecture prudente, à deux niveaux.** D'abord, cette "marge apparente" compare
un taux d'intérêt **annuel** à une perte rapportée au **principal** (non
annualisée). Une simplification pédagogique, pas un calcul de rentabilité
réel (qui nécessiterait d'intégrer la durée du prêt, le coût du capital et
les coûts opérationnels).

Ensuite, et surtout : **le backtesting ci-dessus montre que l'EL est
sous-estimée d'environ -60%, de façon quasi uniforme sur tous les grades**, à
cause du biais de censure à droite documenté en section 5. Si l'on corrigeait
grossièrement ce biais (EL × ~2,5), toutes les marges se réduiraient
fortement, et rien ne garantit que leur hiérarchie actuelle (marge croissante
avec le risque du grade) se maintiendrait. **Ce tableau doit donc être lu
comme un ordre de grandeur optimiste, pas comme une recommandation tarifaire
définitive**. C'est en soi un résultat méthodologique important du projet :
un backtesting rigoureux peut révéler qu'une conclusion métier qui semblait
favorable ne l'est peut-être pas une fois le biais de mesure pris en compte.

## 5. Limites et pistes d'amélioration

- **Biais de sélection** : le dataset ne contient que des prêts déjà accordés
  par Lending Club, aucune information sur les emprunteurs refusés à
  l'octroi.
- **Censure à droite, confirmée et quantifiée empiriquement** : le taux de
  défaut du jeu de test (2017, 23,13%) est supérieur à celui du train
  (18,43%), et surtout, le délai moyen avant défaut réellement observé sur le
  test (9,84 mois) est près de deux fois plus court que celui calibré sur le
  train (18,84 mois). Un écart quasi uniforme de -8,6 à -9,4 mois sur les 7
  grades. C'est ce biais qui explique la quasi-totalité de la sous-estimation
  de -59,9% observée lors du backtesting EL (section 4), et qui rend les
  marges apparentes par grade optimistes.
- **Modèle LGD par grade seul (v1) ne bat pas une moyenne naïve** sur la
  validation (-3,11%), seule la version enrichie (v2), intégrant notamment le
  délai avant défaut, apporte un gain réel (MAE divisée par plus de 2).
- **Modèle LGD au pouvoir explicatif modeste** : même dans sa
  version enrichie, le pseudo R² en échantillon reste à 0,2509. La LGD
  réelle dépend largement de facteurs non observés dans ce dataset
  (comportement de l'emprunteur après le premier impayé, efficacité du
  recouvrement).
- **Stress test simplifié** : choc arbitraire sur la PD, non calibré sur une
  récession historique réelle, sans impact modélisé sur la LGD (le
  recouvrement est généralement plus difficile en période de ralentissement
  économique).
- **"Marge apparente" simplifiée et optimiste** : comparaison d'un taux annuel
  à une perte non annualisée, construite sur une EL elle-même sous-estimée
  d'environ 60% (voir ci-dessus) à ne pas interpréter comme une
  recommandation tarifaire réelle.
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

## Auteur

Crespino Marius ADJANINYEDO — [linkedin.com/in/cm-adjaninyedo] — [Email: cmadjaninyedo1@gmail.com]
