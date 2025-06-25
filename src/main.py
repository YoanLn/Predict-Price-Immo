import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuration de base
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Charge les données depuis les fichiers CSV"""
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, test_data

def quick_data_overview(df, dataset_name):
    """Aperçu rapide des données"""
    print(f"\n--- {dataset_name} ---")
    print(f"Taille: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()[:10]}..." if len(df.columns) > 10 else f"Colonnes: {df.columns.tolist()}")
    print(f"Types de données:\n{df.dtypes.value_counts()}")
    if 'SalePrice' in df.columns:
        print(f"Prix moyen: ${df['SalePrice'].mean():,.0f}")
        print(f"Prix médian: ${df['SalePrice'].median():,.0f}")

def analyze_missing_values(df):
    """Analyse des valeurs manquantes"""
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    missing_table = pd.DataFrame({
        'missing_count': missing_data,
        'missing_percentage': missing_percent
    })
    missing_table = missing_table[missing_table['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if len(missing_table) > 0:
        print(f"\nValeurs manquantes trouvées dans {len(missing_table)} colonnes:")
        print(missing_table.head(10))
    else:
        print("\nAucune valeur manquante détectée")
    
    return missing_table

def create_visualizations(df):
    """Crée quelques graphiques pour comprendre les données"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution du prix
    axes[0, 0].hist(df['SalePrice'], bins=50, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Distribution des Prix de Vente')
    axes[0, 0].set_xlabel('Prix ($)')
    axes[0, 0].set_ylabel('Fréquence')
    
    # Prix vs Surface habitable
    axes[0, 1].scatter(df['GrLivArea'], df['SalePrice'], alpha=0.6, color='lightcoral')
    axes[0, 1].set_title('Prix vs Surface Habitable')
    axes[0, 1].set_xlabel('Surface Habitable (sq ft)')
    axes[0, 1].set_ylabel('Prix ($)')
    
    # Prix par quartier (top 10)
    neighborhood_prices = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).head(10)
    axes[1, 0].bar(range(len(neighborhood_prices)), neighborhood_prices.values, color='lightgreen')
    axes[1, 0].set_title('Prix Médian par Quartier (Top 10)')
    axes[1, 0].set_xlabel('Quartiers')
    axes[1, 0].set_ylabel('Prix Médian ($)')
    axes[1, 0].set_xticks(range(len(neighborhood_prices)))
    axes[1, 0].set_xticklabels(neighborhood_prices.index, rotation=45)
    
    # Qualité générale vs Prix
    quality_prices = df.groupby('OverallQual')['SalePrice'].mean()
    axes[1, 1].plot(quality_prices.index, quality_prices.values, marker='o', color='purple', linewidth=2)
    axes[1, 1].set_title('Prix Moyen par Qualité Générale')
    axes[1, 1].set_xlabel('Qualité Générale (1-10)')
    axes[1, 1].set_ylabel('Prix Moyen ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()

def handle_missing_values(df):
    """Gère les valeurs manquantes de façon intelligente"""
    df_clean = df.copy()
    
    # Pour les variables catégorielles, on remplace par 'None' ou 'No'
    categorical_na_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                          'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                          'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    
    for col in categorical_na_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')
    
    # Pour les variables numériques liées au garage/basement, on met 0
    numeric_na_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                      'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    
    for col in numeric_na_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # LotFrontage: on utilise la médiane par quartier
    if 'LotFrontage' in df_clean.columns:
        df_clean['LotFrontage'] = df_clean.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
    
    # MasVnrArea: 0 si pas de masonry veneer
    if 'MasVnrArea' in df_clean.columns:
        df_clean['MasVnrArea'] = df_clean['MasVnrArea'].fillna(0)
    
    # MasVnrType: None si pas de masonry veneer
    if 'MasVnrType' in df_clean.columns:
        df_clean['MasVnrType'] = df_clean['MasVnrType'].fillna('None')
    
    # Pour les autres, on utilise le mode ou la médiane
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

def create_new_features(df):
    """Crée de nouvelles variables qui peuvent être utiles"""
    df_enhanced = df.copy()
    
    # Surface totale de la maison
    df_enhanced['total_sqft'] = df_enhanced['GrLivArea'] + df_enhanced['TotalBsmtSF']
    
    # Nombre total de salles de bain
    df_enhanced['total_bathrooms'] = (df_enhanced['FullBath'] + 
                                     0.5 * df_enhanced['HalfBath'] + 
                                     df_enhanced['BsmtFullBath'] + 
                                     0.5 * df_enhanced['BsmtHalfBath'])
    
    # Age de la maison
    df_enhanced['house_age'] = 2023 - df_enhanced['YearBuilt']
    
    # Années depuis la rénovation
    df_enhanced['years_since_remod'] = 2023 - df_enhanced['YearRemodAdd']
    
    # Age du garage
    df_enhanced['garage_age'] = np.where(df_enhanced['GarageYrBlt'] > 0, 
                                        2023 - df_enhanced['GarageYrBlt'], 0)
    
    # Interaction qualité x condition
    df_enhanced['quality_condition'] = df_enhanced['OverallQual'] * df_enhanced['OverallCond']
    
    # Surface par chambre
    df_enhanced['sqft_per_room'] = df_enhanced['GrLivArea'] / (df_enhanced['TotRmsAbvGrd'] + 1)
    
    return df_enhanced

def encode_categorical_variables(train_df, test_df):
    """Encode les variables catégorielles"""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # On identifie les colonnes catégorielles
    categorical_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # On utilise LabelEncoder pour chaque colonne
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        
        # On combine train et test pour avoir toutes les catégories
        combined_values = pd.concat([train_encoded[col], test_encoded[col]], axis=0)
        le.fit(combined_values.astype(str))
        
        train_encoded[col] = le.transform(train_encoded[col].astype(str))
        test_encoded[col] = le.transform(test_encoded[col].astype(str))
        
        label_encoders[col] = le
    
    return train_encoded, test_encoded, label_encoders

def remove_outliers(df, target_col='SalePrice'):
    """Supprime les outliers évidents"""
    df_clean = df.copy()
    
    # On supprime les maisons avec une surface habitable énorme mais prix bas
    df_clean = df_clean.drop(df_clean[(df_clean['GrLivArea'] > 4000) & 
                                     (df_clean[target_col] < 300000)].index)
    
    # On supprime les maisons avec des prix extrêmes
    Q1 = df_clean[target_col].quantile(0.25)
    Q3 = df_clean[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_removed = len(df_clean) - len(df_clean[(df_clean[target_col] >= lower_bound) & 
                                                   (df_clean[target_col] <= upper_bound)])
    
    df_clean = df_clean[(df_clean[target_col] >= lower_bound) & 
                       (df_clean[target_col] <= upper_bound)]
    
    print(f"Outliers supprimés: {outliers_removed}")
    
    return df_clean

def train_models(X_train, X_val, y_train, y_val):
    """Entraîne différents modèles et compare leurs performances"""
    models = {}
    results = {}
    
    print("\nEntraînement des modèles...")
    
    # Régression linéaire
    print("- Régression linéaire")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_val)
    models['Linear Regression'] = lr
    results['Linear Regression'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, lr_pred)),
        'r2': r2_score(y_val, lr_pred)
    }
    
    # Random Forest
    print("- Random Forest")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, rf_pred)),
        'r2': r2_score(y_val, rf_pred)
    }
    
    # XGBoost
    print("- XGBoost")
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, xgb_pred)),
        'r2': r2_score(y_val, xgb_pred)
    }
    
    return models, results

def evaluate_models(results):
    """Affiche et compare les résultats des modèles"""
    print("\n--- Résultats des Modèles ---")
    
    results_df = pd.DataFrame(results).T
    results_df['rmse'] = results_df['rmse'].round(0)
    results_df['r2_percent'] = (results_df['r2'] * 100).round(1)
    
    print(results_df[['rmse', 'r2_percent']])
    
    # Trouve le meilleur modèle
    best_model_name = results_df['r2'].idxmax()
    print(f"\nMeilleur modèle: {best_model_name}")
    print(f"R² Score: {results_df.loc[best_model_name, 'r2_percent']:.1f}%")
    print(f"RMSE: ${results_df.loc[best_model_name, 'rmse']:,.0f}")
    
    return best_model_name

def create_model_comparison_plot(results):
    """Crée un graphique de comparaison des modèles"""
    models = list(results.keys())
    rmse_values = [results[model]['rmse'] for model in models]
    r2_values = [results[model]['r2'] * 100 for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # RMSE comparison
    bars1 = ax1.bar(models, rmse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Comparaison RMSE')
    ax1.set_ylabel('RMSE ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars1, rmse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
                f'${value:,.0f}', ha='center', va='bottom')
    
    # R² comparison
    bars2 = ax2.bar(models, r2_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('Comparaison R² Score')
    ax2.set_ylabel('R² Score (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_feature_importance_plot(model, feature_names, model_name):
    """Crée un graphique d'importance des features"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Importance des Variables - {model_name}')
        plt.bar(range(15), importances[indices])
        plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

def make_predictions(model, X_test, test_ids):
    """Fait les prédictions finales et crée le fichier de soumission"""
    predictions = model.predict(X_test)
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': predictions
    })
    
    submission.to_csv('submission.csv', index=False)
    print(f"\nPrédictions sauvegardées dans 'submission.csv'")
    print(f"Nombre de prédictions: {len(predictions)}")
    print(f"Prix moyen prédit: ${predictions.mean():,.0f}")
    print(f"Prix médian prédit: ${np.median(predictions):,.0f}")
    
    return submission

def main():
    """Fonction principale qui orchestre tout le processus"""
    print("Démarrage de l'analyse des prix immobiliers")
    print("=" * 50)
    
    # 1. Chargement des données
    print("\n1. Chargement des données...")
    train_data, test_data = load_data()
    quick_data_overview(train_data, "Données d'entraînement")
    quick_data_overview(test_data, "Données de test")
    
    # 2. Analyse exploratoire
    print("\n2. Analyse exploratoire...")
    missing_analysis = analyze_missing_values(train_data)
    create_visualizations(train_data)
    
    # 3. Nettoyage des données
    print("\n3. Nettoyage des données...")
    train_clean = handle_missing_values(train_data)
    test_clean = handle_missing_values(test_data)
    print("Valeurs manquantes traitées")
    
    # 4. Création de nouvelles variables
    print("\n4. Ingénierie des features...")
    train_enhanced = create_new_features(train_clean)
    test_enhanced = create_new_features(test_clean)
    print("Nouvelles variables créées")
    
    # 5. Suppression des outliers
    print("\n5. Suppression des outliers...")
    train_no_outliers = remove_outliers(train_enhanced)
    
    # 6. Encodage des variables catégorielles
    print("\n6. Encodage des variables...")
    # On sépare la variable cible
    y = train_no_outliers['SalePrice']
    X_train_features = train_no_outliers.drop(['Id', 'SalePrice'], axis=1)
    X_test_features = test_enhanced.drop(['Id'], axis=1)
    test_ids = test_enhanced['Id']
    
    X_train_encoded, X_test_encoded, encoders = encode_categorical_variables(X_train_features, X_test_features)
    
    # 7. Division train/validation
    print("\n7. Division des données...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_encoded, y, test_size=0.2, random_state=42
    )
    
    # 8. Standardisation
    print("\n8. Standardisation...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    # 9. Entraînement des modèles
    print("\n9. Entraînement des modèles...")
    models, results = train_models(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # 10. Évaluation
    print("\n10. Évaluation des modèles...")
    best_model_name = evaluate_models(results)
    create_model_comparison_plot(results)
    
    # 11. Analyse d'importance des features
    print("\n11. Analyse des variables importantes...")
    best_model = models[best_model_name]
    feature_names = X_train_encoded.columns.tolist()
    create_feature_importance_plot(best_model, feature_names, best_model_name)
    
    # 12. Prédictions finales
    print("\n12. Prédictions finales...")
    # On réentraîne le meilleur modèle sur toutes les données d'entraînement
    X_full_scaled = scaler.fit_transform(X_train_encoded)
    X_test_final_scaled = scaler.transform(X_test_encoded)
    best_model.fit(X_full_scaled, y)
    
    submission = make_predictions(best_model, X_test_final_scaled, test_ids)
    
    print("\n" + "=" * 50)
    print("Analyse terminée !")
    print(f"Meilleur modèle: {best_model_name}")
    print("Fichiers générés:")
    print("- data_exploration.png")
    print("- model_comparison.png") 
    print("- feature_importance_*.png")
    print("- submission.csv")

if __name__ == "__main__":
    main()