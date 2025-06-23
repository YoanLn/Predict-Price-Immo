import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.special import boxcox1p
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data():
    """Charge et prepare les donnees avec preprocessing avance"""
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    print(f"Donnees chargees - Train: {train_data.shape}, Test: {test_data.shape}")
    
    return train_data, test_data

def advanced_feature_engineering(df):
    """Creation de features plus sophistiquees"""
    df_enhanced = df.copy()
    
    # Features de base
    df_enhanced['total_sqft'] = df_enhanced['GrLivArea'] + df_enhanced['TotalBsmtSF']
    df_enhanced['total_bathrooms'] = (df_enhanced['FullBath'] + 
                                     0.5 * df_enhanced['HalfBath'] + 
                                     df_enhanced['BsmtFullBath'] + 
                                     0.5 * df_enhanced['BsmtHalfBath'])
    
    # Features temporelles
    df_enhanced['house_age'] = 2023 - df_enhanced['YearBuilt']
    df_enhanced['years_since_remod'] = 2023 - df_enhanced['YearRemodAdd']
    df_enhanced['garage_age'] = np.where(df_enhanced['GarageYrBlt'] > 0, 
                                        2023 - df_enhanced['GarageYrBlt'], 0)
    
    # Features d'interaction
    df_enhanced['quality_condition'] = df_enhanced['OverallQual'] * df_enhanced['OverallCond']
    df_enhanced['total_sqft_per_room'] = df_enhanced['total_sqft'] / (df_enhanced['TotRmsAbvGrd'] + 1)
    df_enhanced['price_per_sqft'] = df_enhanced['total_sqft'] * df_enhanced['OverallQual']
    
    # Features de surface
    df_enhanced['total_porch_sf'] = (df_enhanced['OpenPorchSF'] + df_enhanced['3SsnPorch'] + 
                                    df_enhanced['EnclosedPorch'] + df_enhanced['ScreenPorch'])
    
    # Features binaires
    df_enhanced['has_pool'] = (df_enhanced['PoolArea'] > 0).astype(int)
    df_enhanced['has_garage'] = (df_enhanced['GarageArea'] > 0).astype(int)
    df_enhanced['has_basement'] = (df_enhanced['TotalBsmtSF'] > 0).astype(int)
    df_enhanced['has_fireplace'] = (df_enhanced['Fireplaces'] > 0).astype(int)
    df_enhanced['is_new'] = (df_enhanced['YearBuilt'] >= 2000).astype(int)
    
    # Features de qualite moyennee
    quality_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                       'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    # On encode temporairement pour calculer la moyenne
    temp_df = df_enhanced.copy()
    quality_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, 'NA': 0}
    
    for feature in quality_features:
        if feature in temp_df.columns:
            temp_df[feature + '_encoded'] = temp_df[feature].map(quality_mapping).fillna(0)
    
    encoded_quality_cols = [col for col in temp_df.columns if col.endswith('_encoded')]
    if encoded_quality_cols:
        df_enhanced['avg_quality'] = temp_df[encoded_quality_cols].mean(axis=1)
    
    return df_enhanced

def handle_missing_values_advanced(df):
    """Gestion avancee des valeurs manquantes"""
    df_clean = df.copy()
    
    # Valeurs manquantes qui signifient "None"
    none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
                'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
    
    for col in none_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')
    
    # Valeurs numeriques a remplacer par 0
    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                'BsmtHalfBath', 'MasVnrArea']
    
    for col in zero_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # LotFrontage par quartier
    if 'LotFrontage' in df_clean.columns:
        df_clean['LotFrontage'] = df_clean.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))
    
    # Autres valeurs manquantes
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean

def detect_and_handle_skewness(df, target_col=None):
    """Detecte et corrige l'asymetrie des variables numeriques"""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_features:
        numeric_features.remove(target_col)
    
    skewed_features = []
    for feature in numeric_features:
        skewness = stats.skew(df[feature])
        if abs(skewness) > 0.75:
            skewed_features.append(feature)
    
    print(f"Variables asymetriques detectees: {len(skewed_features)}")
    
    # Application de Box-Cox
    df_transformed = df.copy()
    for feature in skewed_features:
        df_transformed[feature] = boxcox1p(df_transformed[feature], 0.15)
    
    return df_transformed, skewed_features

def advanced_outlier_removal(df, target_col='SalePrice'):
    """Suppression d'outliers plus sophistiquee"""
    df_clean = df.copy()
    
    # Outliers specifiques au domaine
    df_clean = df_clean.drop(df_clean[(df_clean['GrLivArea'] > 4000) & 
                                     (df_clean[target_col] < 300000)].index)
    
    # Outliers multivaries avec Isolation Forest
    from sklearn.ensemble import IsolationForest
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numeric_cols:
        numeric_cols.remove('Id')
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(df_clean[numeric_cols])
    
    outliers_removed = sum(outliers == -1)
    df_clean = df_clean[outliers == 1]
    
    print(f"Outliers supprimes: {outliers_removed}")
    
    return df_clean

def encode_categorical_advanced(train_df, test_df, target_col='SalePrice'):
    """Encodage avance des variables categorielles"""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    categorical_cols = train_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Label encoding pour toutes les variables categorielles
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined_values = pd.concat([train_encoded[col], test_encoded[col]], axis=0)
        le.fit(combined_values.astype(str))
        
        train_encoded[col] = le.transform(train_encoded[col].astype(str))
        test_encoded[col] = le.transform(test_encoded[col].astype(str))
        label_encoders[col] = le
    
    return train_encoded, test_encoded, label_encoders

def feature_selection(X, y, k=50):
    """Selection des meilleures features"""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Features selectionnees: {len(selected_features)} sur {X.shape[1]}")
    
    return X_selected, selected_features, selector

def train_advanced_models(X_train, X_val, y_train, y_val):
    """Entrainement de modeles avances avec hyperparameter tuning"""
    models = {}
    results = {}
    
    print("\nEntrainement des modeles avances...")
    
    # Ridge avec hyperparameter tuning
    print("- Ridge Regression avec GridSearch")
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2', n_jobs=-1)
    ridge_grid.fit(X_train, y_train)
    ridge_pred = ridge_grid.predict(X_val)
    models['Ridge'] = ridge_grid.best_estimator_
    results['Ridge'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, ridge_pred)),
        'r2': r2_score(y_val, ridge_pred),
        'best_params': ridge_grid.best_params_
    }
    
    # Random Forest optimise
    print("- Random Forest optimise")
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_pred = rf_grid.predict(X_val)
    models['Random Forest'] = rf_grid.best_estimator_
    results['Random Forest'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, rf_pred)),
        'r2': r2_score(y_val, rf_pred),
        'best_params': rf_grid.best_params_
    }
    
    # XGBoost optimise
    print("- XGBoost optimise")
    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2]
    }
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='r2', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_pred = xgb_grid.predict(X_val)
    models['XGBoost'] = xgb_grid.best_estimator_
    results['XGBoost'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, xgb_pred)),
        'r2': r2_score(y_val, xgb_pred),
        'best_params': xgb_grid.best_params_
    }
    
    # LightGBM
    print("- LightGBM")
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    models['LightGBM'] = lgb_model
    results['LightGBM'] = {
        'rmse': np.sqrt(mean_squared_error(y_val, lgb_pred)),
        'r2': r2_score(y_val, lgb_pred)
    }
    
    return models, results

def create_ensemble_model(models, X_val, y_val):
    """Cree un modele ensemble par moyenne ponderee"""
    predictions = {}
    weights = {}
    
    for name, model in models.items():
        pred = model.predict(X_val)
        r2 = r2_score(y_val, pred)
        predictions[name] = pred
        weights[name] = r2
    
    # Normalisation des poids
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # Prediction ensemble
    ensemble_pred = np.zeros(len(y_val))
    for name, pred in predictions.items():
        ensemble_pred += weights[name] * pred
    
    ensemble_r2 = r2_score(y_val, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
    
    print(f"\nEnsemble Model - R²: {ensemble_r2:.3f}, RMSE: {ensemble_rmse:.0f}")
    print("Poids des modeles:", {k: f"{v:.3f}" for k, v in weights.items()})
    
    return weights, ensemble_r2, ensemble_rmse

def evaluate_advanced_models(results):
    """Evaluation detaillee des modeles"""
    print("\n--- Resultats des Modeles Avances ---")
    
    results_df = pd.DataFrame(results).T
    results_df['rmse'] = results_df['rmse'].round(0)
    results_df['r2_percent'] = (results_df['r2'] * 100).round(1)
    
    print(results_df[['rmse', 'r2_percent']])
    
    # Affichage des meilleurs parametres
    print("\nMeilleurs parametres:")
    for model_name, result in results.items():
        if 'best_params' in result:
            print(f"{model_name}: {result['best_params']}")
    
    best_model_name = results_df['r2'].idxmax()
    print(f"\nMeilleur modele: {best_model_name}")
    print(f"R² Score: {results_df.loc[best_model_name, 'r2_percent']:.1f}%")
    
    return best_model_name

def make_ensemble_predictions(models, weights, X_test, test_ids):
    """Fait les predictions avec le modele ensemble"""
    ensemble_pred = np.zeros(len(X_test))
    
    for name, model in models.items():
        pred = model.predict(X_test)
        ensemble_pred += weights[name] * pred
    
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': ensemble_pred
    })
    
    submission.to_csv('submission_optimized.csv', index=False)
    print(f"\nPredictions ensemble sauvegardees dans 'submission_optimized.csv'")
    print(f"Prix moyen predit: ${ensemble_pred.mean():,.0f}")
    
    return submission

def main():
    """Pipeline d'optimisation avance"""
    print("Demarrage de l'optimisation avancee")
    print("=" * 50)
    
    # 1. Chargement
    train_data, test_data = load_and_prepare_data()
    
    # 2. Feature engineering avance
    print("\n2. Feature engineering avance...")
    train_enhanced = advanced_feature_engineering(train_data)
    test_enhanced = advanced_feature_engineering(test_data)
    
    # 3. Gestion des valeurs manquantes
    print("\n3. Gestion avancee des valeurs manquantes...")
    train_clean = handle_missing_values_advanced(train_enhanced)
    test_clean = handle_missing_values_advanced(test_enhanced)
    
    # 4. Correction de l'asymetrie
    print("\n4. Correction de l'asymetrie...")
    train_transformed, skewed_features = detect_and_handle_skewness(train_clean, 'SalePrice')
    test_transformed, _ = detect_and_handle_skewness(test_clean)
    
    # 5. Suppression d'outliers avancee
    print("\n5. Suppression d'outliers avancee...")
    train_no_outliers = advanced_outlier_removal(train_transformed)
    
    # 6. Encodage avance
    print("\n6. Encodage avance...")
    y = train_no_outliers['SalePrice']
    X_train_features = train_no_outliers.drop(['Id', 'SalePrice'], axis=1)
    X_test_features = test_transformed.drop(['Id'], axis=1)
    test_ids = test_transformed['Id']
    
    X_train_encoded, X_test_encoded, encoders = encode_categorical_advanced(
        X_train_features, X_test_features, 'SalePrice')
    
    # 7. Selection de features
    print("\n7. Selection de features...")
    X_train_selected, selected_features, selector = feature_selection(X_train_encoded, y, k=60)
    X_test_selected = selector.transform(X_test_encoded)
    
    # 8. Division et standardisation
    print("\n8. Division et standardisation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_selected, y, test_size=0.2, random_state=42)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # 9. Entrainement des modeles avances
    print("\n9. Entrainement des modeles avances...")
    models, results = train_advanced_models(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # 10. Evaluation
    print("\n10. Evaluation...")
    best_model_name = evaluate_advanced_models(results)
    
    # 11. Modele ensemble
    print("\n11. Creation du modele ensemble...")
    weights, ensemble_r2, ensemble_rmse = create_ensemble_model(models, X_val_scaled, y_val)
    
    # 12. Predictions finales
    print("\n12. Predictions finales avec ensemble...")
    X_full_scaled = scaler.fit_transform(X_train_selected)
    
    # Reentrainement sur toutes les donnees
    final_models = {}
    for name, model in models.items():
        model.fit(X_full_scaled, y)
        final_models[name] = model
    
    submission = make_ensemble_predictions(final_models, weights, X_test_scaled, test_ids)
    
    print("\n" + "=" * 50)
    print("Optimisation terminee !")
    print(f"Meilleur modele individuel: {best_model_name}")
    print(f"Ensemble R² Score: {ensemble_r2:.1%}")
    print(f"Ensemble RMSE: ${ensemble_rmse:,.0f}")
    print("Fichier genere: submission_optimized.csv")

if __name__ == "__main__":
    main()