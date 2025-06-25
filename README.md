# Prédiction des Prix Immobiliers

Un projet de machine learning pour prédire les prix des maisons en utilisant le dataset Kaggle "House Prices". Ce projet explore différentes techniques d'analyse de données et d'apprentissage automatique sur des données tabulaires.

## Vue d'ensemble

L'objectif est de construire un modèle capable de prédire avec précision le prix de vente des maisons en se basant sur leurs caractéristiques. Le projet couvre l'ensemble du pipeline de data science : exploration, nettoyage, feature engineering, modélisation et évaluation.

## Technologies

- Python 3.x
- pandas, numpy pour la manipulation de données
- scikit-learn pour le machine learning
- XGBoost, LightGBM pour le gradient boosting
- matplotlib, seaborn pour les visualisations
- Jupyter pour l'exploration interactive

## Structure du projet

```
Predict-Price-Immo/
├── data/                   # Données du projet
│   ├── train.csv          # Données d'entraînement
│   ├── test.csv           # Données de test
│   ├── sample_submission.csv
│   └── data_description.txt
├── notebooks/              # Analyses exploratoires
│   └── house_price_exploration.ipynb
├── src/                    # Code source
│   ├── main.py            # Pipeline principal
│   ├── optimize.py        # Version optimisée
│   ├── demo.py            # Démonstration interactive
│   └── utils.py           # Fonctions utilitaires
├── results/                # Résultats et visualisations
│   ├── data_exploration.png
│   ├── model_comparison.png
│   ├── submission.csv
│   └── submission_optimized.csv
├── requirements.txt
└── README.md
```

## Installation

Clonez le repository et installez les dépendances :

```bash
git clone https://github.com/YoanLeNevez/Predict-Price-Immo.git
cd Predict-Price-Immo
pip install -r requirements.txt
```

## Utilisation

### Exploration des données
Commencez par le notebook Jupyter pour une analyse exploratoire interactive :
```bash
jupyter notebook notebooks/house_price_exploration.ipynb
```

### Scripts Python
Trois approches différentes sont disponibles :

**Démonstration simple :**
```bash
python src/demo.py
```

**Pipeline complet :**
```bash
python src/main.py
```

**Version optimisée :**
```bash
python src/optimize.py
```

## Le dataset

Le dataset contient 1460 maisons d'entraînement et 1459 maisons de test avec 79 variables décrivant différents aspects des propriétés résidentielles.

Les variables incluent :
- Caractéristiques physiques : surface, nombre de pièces, garage
- Qualité et condition de la maison
- Informations temporelles : année de construction, rénovations
- Localisation et environnement
- Équipements et finitions

## Approche méthodologique

### 1. Exploration des données
Analyse des distributions, corrélations et valeurs manquantes pour comprendre la structure du dataset.

### 2. Preprocessing
- Traitement intelligent des valeurs manquantes
- Feature engineering : création de nouvelles variables pertinentes
- Encodage des variables catégorielles
- Gestion des outliers
- Transformation des données asymétriques

### 3. Modélisation
Test de plusieurs algorithmes :
- Régression linéaire (baseline)
- Random Forest
- XGBoost
- LightGBM
- Modèles ensemble

### 4. Évaluation
- Validation croisée
- Métriques : RMSE, R²
- Analyse de l'importance des features
- Visualisation des performances

## Résultats

**Version de base :**
- Linear Regression : 88.0% R²
- Random Forest : 87.0% R²
- XGBoost : 87.6% R²

**Version optimisée :**
- Random Forest : 89.0% R²
- XGBoost : 90.2% R²
- LightGBM : 90.1% R²
- Modèle ensemble : 90.6% R²

Les variables les plus prédictives sont généralement la surface habitable, la qualité générale, le quartier et l'âge de la maison.

## Points clés du projet

Ce projet illustre un pipeline complet de data science avec :
- Une exploration approfondie des données
- Des techniques avancées de feature engineering
- La comparaison rigoureuse de différents modèles
- L'optimisation des hyperparamètres
- La création de modèles ensemble

Il démontre l'importance du preprocessing et de la compréhension métier dans la performance des modèles de machine learning.

## Améliorations possibles

- Stacking de modèles plus sophistiqué
- Feature selection automatisée
- Validation temporelle pour les données immobilières
- Intégration de données externes (économiques, géographiques)
- Déploiement du modèle en production 