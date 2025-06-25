"""
Fonctions utilitaires pour le projet de prédiction des prix immobiliers

Ce fichier contient les fonctions communes utilisées dans main.py et optimize.py
pour éviter la duplication de code (principe DRY - Don't Repeat Yourself)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def load_data():
    """
    Charge les donnees depuis les fichiers CSV
    
    Returns:
        tuple: (train_data, test_data) - DataFrames pandas
    """
    try:
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        print(f"Donnees chargees - Train: {train_data.shape}, Test: {test_data.shape}")
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Erreur: Fichier non trouve - {e}")
        return None, None

def analyze_missing_values(df, dataset_name="Dataset"):
    """
    Analyse les valeurs manquantes dans un DataFrame
    
    Args:
        df: DataFrame a analyser
        dataset_name: Nom du dataset pour l'affichage
    
    Returns:
        DataFrame: Table des valeurs manquantes
    """
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    missing_table = pd.DataFrame({
        'missing_count': missing_data,
        'missing_percentage': missing_percent
    })
    missing_table = missing_table[missing_table['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if len(missing_table) > 0:
        print(f"\n{dataset_name} - Valeurs manquantes dans {len(missing_table)} colonnes:")
        print(missing_table.head(10))
    else:
        print(f"\n{dataset_name} - Aucune valeur manquante detectee")
    
    return missing_table

def calculate_model_metrics(y_true, y_pred):
    """
    Calcule les metriques de performance d'un modele
    
    Args:
        y_true: Vraies valeurs
        y_pred: Predictions
    
    Returns:
        dict: Dictionnaire avec RMSE et R2
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'r2_percent': r2 * 100
    }

def save_predictions(predictions, test_ids, filename='submission.csv'):
    """
    Sauvegarde les predictions au format Kaggle
    
    Args:
        predictions: Array des predictions
        test_ids: IDs des maisons de test
        filename: Nom du fichier de sortie
    """
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"Predictions sauvegardees dans '{filename}'")
    print(f"Prix moyen predit: ${predictions.mean():,.0f}")
    print(f"Prix median predit: ${np.median(predictions):,.0f}")

def print_separator(title="", char="=", width=50):
    """
    Affiche un séparateur décoratif
    
    Args:
        title: Titre à afficher au centre
        char: Caractère pour le séparateur
        width: Largeur du séparateur
    """
    if title:
        print(f"\n{title}")
        print(char * width)
    else:
        print(char * width)

def create_feature_importance_plot(model, feature_names, model_name, top_n=20):
    """
    Cree un graphique d'importance des features
    
    Args:
        model: Modele entraine
        feature_names: Noms des features
        model_name: Nom du modele
        top_n: Nombre de features a afficher
    """
    # Recuperation de l'importance selon le type de modele
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        print(f"Impossible d'extraire l'importance des features pour {model_name}")
        return
    
    # Creation du DataFrame pour le tri
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Graphique
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title(f'Top {top_n} - Importance des Variables ({model_name})')
    plt.xlabel('Importance')
    plt.ylabel('Variables')
    plt.tight_layout()
    
    # Sauvegarde
    filename = f'feature_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Graphique sauvegarde: {filename}")

def encode_categorical_variables(train_df, test_df):
    """
    Encode les variables categorielles avec LabelEncoder
    
    Args:
        train_df: DataFrame d'entrainement
        test_df: DataFrame de test
    
    Returns:
        tuple: (train_encoded, test_encoded, label_encoders)
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    categorical_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    print(f"Encodage de {len(categorical_cols)} variables categorielles...")
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # Combinaison des valeurs train et test pour avoir toutes les categories
        combined_values = pd.concat([train_encoded[col], test_encoded[col]], axis=0)
        le.fit(combined_values.astype(str))
        
        train_encoded[col] = le.transform(train_encoded[col].astype(str))
        test_encoded[col] = le.transform(test_encoded[col].astype(str))
        
        label_encoders[col] = le
    
    return train_encoded, test_encoded, label_encoders

def create_basic_features(df):
    """
    Cree des features de base communes aux deux scripts
    
    Args:
        df: DataFrame d'entree
    
    Returns:
        DataFrame: DataFrame avec nouvelles features
    """
    df_enhanced = df.copy()
    
    # Surface totale de la maison
    df_enhanced['total_sqft'] = df_enhanced['GrLivArea'] + df_enhanced['TotalBsmtSF']
    
    # Nombre total de salles de bain
    df_enhanced['total_bathrooms'] = (df_enhanced['FullBath'] + 
                                     0.5 * df_enhanced['HalfBath'] + 
                                     df_enhanced['BsmtFullBath'] + 
                                     0.5 * df_enhanced['BsmtHalfBath'])
    
    # Age de la maison
    current_year = 2023
    df_enhanced['house_age'] = current_year - df_enhanced['YearBuilt']
    df_enhanced['years_since_remod'] = current_year - df_enhanced['YearRemodAdd']
    
    # Age du garage
    df_enhanced['garage_age'] = np.where(df_enhanced['GarageYrBlt'] > 0, 
                                        current_year - df_enhanced['GarageYrBlt'], 0)
    
    # Interactions
    df_enhanced['quality_condition'] = df_enhanced['OverallQual'] * df_enhanced['OverallCond']
    df_enhanced['sqft_per_room'] = df_enhanced['GrLivArea'] / (df_enhanced['TotRmsAbvGrd'] + 1)
    
    return df_enhanced

def handle_common_missing_values(df):
    """
    Gere les valeurs manquantes les plus communes
    
    Args:
        df: DataFrame a nettoyer
    
    Returns:
        DataFrame: DataFrame nettoye
    """
    df_clean = df.copy()
    
    # Variables categorielles ou NA signifie "None"
    categorical_na_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                          'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                          'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
    
    for col in categorical_na_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')
    
    # Variables numeriques a remplacer par 0
    numeric_na_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                      'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                      'BsmtHalfBath', 'MasVnrArea']
    
    for col in numeric_na_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # LotFrontage par quartier
    if 'LotFrontage' in df_clean.columns:
        df_clean['LotFrontage'] = df_clean.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
    
    return df_clean

def validate_data_quality(train_df, test_df):
    """
    Valide la qualite des donnees chargees
    
    Args:
        train_df: DataFrame d'entrainement
        test_df: DataFrame de test
    
    Returns:
        bool: True si les donnees sont valides
    """
    issues = []
    
    # Verification des colonnes essentielles
    if 'SalePrice' not in train_df.columns:
        issues.append("Colonne 'SalePrice' manquante dans les donnees d'entrainement")
    
    if 'Id' not in train_df.columns or 'Id' not in test_df.columns:
        issues.append("Colonne 'Id' manquante")
    
    # Verification des tailles
    if len(train_df) == 0 or len(test_df) == 0:
        issues.append("Dataset vide detecte")
    
    # Verification des doublons d'ID
    if train_df['Id'].duplicated().any():
        issues.append("IDs dupliques dans train")
    
    if test_df['Id'].duplicated().any():
        issues.append("IDs dupliques dans test")
    
    # Verification des valeurs aberrantes dans SalePrice
    if 'SalePrice' in train_df.columns:
        price_stats = train_df['SalePrice'].describe()
        if price_stats['min'] <= 0:
            issues.append("Prix negatifs ou nuls detectes")
        if price_stats['max'] > 1000000:  # Prix superieur a 1M semble suspect
            issues.append("Prix extremement eleves detectes (>1M)")
    
    if issues:
        print("Problemes de qualite detectes:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("Validation des donnees: OK")
        return True

def get_feature_types(df):
    """
    Categorise les features par type pour un meilleur preprocessing
    
    Args:
        df: DataFrame a analyser
    
    Returns:
        dict: Dictionnaire avec les types de features
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Enlever les colonnes speciales
    if 'Id' in numeric_features:
        numeric_features.remove('Id')
    if 'SalePrice' in numeric_features:
        numeric_features.remove('SalePrice')
    
    # Features ordinales (ordre important)
    ordinal_features = [
        'OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 
        'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
        'FireplaceQu', 'GarageQual', 'GarageCond'
    ]
    
    # Features temporelles
    temporal_features = [
        'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold'
    ]
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features,
        'ordinal': [f for f in ordinal_features if f in df.columns],
        'temporal': [f for f in temporal_features if f in df.columns]
    }

if __name__ == "__main__":
    print("Ce fichier contient des fonctions utilitaires.")
    print("Utilisez 'python main.py' ou 'python optimize.py' pour lancer l'analyse.") 