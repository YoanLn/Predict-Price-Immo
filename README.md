# Prédiction des Prix Immobiliers

Projet d'apprentissage en machine learning réalisé par Yoan LE NEVEZ dans le cadre de mes études en data science.

## À propos

Ce projet utilise le dataset Kaggle "House Prices" pour apprendre et pratiquer les techniques de machine learning sur des données tabulaires. L'objectif est de prédire le prix de vente des maisons en utilisant différents algorithmes et de comparer leurs performances.

## Technologies utilisées

- **Python 3.x**
- **pandas** - manipulation des données
- **numpy** - calculs numériques
- **scikit-learn** - algorithmes de machine learning
- **xgboost** - gradient boosting
- **matplotlib & seaborn** - visualisations
- **jupyter** - développement interactif

## Dataset

Le dataset contient 1460 maisons d'entraînement et 1459 maisons de test avec 79 variables explicatives décrivant différents aspects des propriétés résidentielles.

Variables principales:
- Surface habitable, nombre de chambres, salles de bain
- Qualité générale et condition de la maison
- Année de construction et rénovations
- Caractéristiques du garage, sous-sol, terrain
- Localisation (quartier)

## Installation

```bash
git clone https://github.com/YoanLeNevez/Predict-Price-Immo.git
cd Predict-Price-Immo
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Utilisation

```bash
python main.py
```

Le script va automatiquement:
1. Charger et explorer les données
2. Nettoyer les valeurs manquantes
3. Créer de nouvelles variables
4. Entraîner 3 modèles différents
5. Comparer les performances
6. Générer les prédictions pour Kaggle

## Pipeline d'analyse

### 1. Exploration des données
- Analyse des types de variables
- Distribution du prix cible
- Détection des valeurs manquantes
- Visualisations exploratoires

### 2. Preprocessing
- Gestion intelligente des valeurs manquantes
- Création de nouvelles features (surface totale, âge de la maison, etc.)
- Encodage des variables catégorielles
- Suppression des outliers
- Standardisation des données

### 3. Modélisation
- **Régression linéaire** - modèle de base
- **Random Forest** - ensemble method
- **XGBoost** - gradient boosting

### 4. Évaluation
- RMSE (Root Mean Square Error)
- R² Score
- Validation croisée
- Analyse de l'importance des variables

## Résultats attendus

D'après mes tests précédents:
- Régression linéaire: ~85% R²
- Random Forest: ~89% R²
- XGBoost: ~90% R²

Les variables les plus importantes sont généralement:
- Surface habitable totale
- Qualité générale de la maison
- Quartier
- Âge de la maison

## Fichiers générés

- `data_exploration.png` - graphiques d'exploration
- `model_comparison.png` - comparaison des performances
- `feature_importance_*.png` - importance des variables
- `submission.csv` - prédictions pour Kaggle

## Ce que j'ai appris

Ce projet m'a permis de pratiquer:
- Le preprocessing complet d'un dataset réel
- La création de nouvelles features pertinentes
- La comparaison de différents algorithmes
- L'évaluation et la visualisation des résultats
- La préparation d'une soumission Kaggle

## Améliorations possibles

- Hyperparameter tuning avec GridSearch
- Feature selection plus poussée
- Ensemble methods (stacking, blending)
- Cross-validation plus robuste
- Analyse des résidus plus approfondie

## Structure du projet

```
Predict-Price-Immo/
├── main.py                 # Script principal
├── train.csv              # Données d'entraînement
├── test.csv               # Données de test
├── data_description.txt   # Description des variables
├── sample_submission.csv  # Format de soumission
└── README.md             # Ce fichier
```

## Contact

Yoan LE NEVEZ - Étudiant en Data Science

N'hésitez pas à me faire des retours ou suggestions pour améliorer ce projet d'apprentissage ! 