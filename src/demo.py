"""
Script de démonstration pour le projet de prédiction des prix immobiliers
"""

import pandas as pd
import numpy as np
from utils import load_data, analyze_missing_values, print_separator

def demo_exploration():
    """Demonstration de l'exploration de donnees"""
    print_separator("DEMO - EXPLORATION DES DONNEES")
    
    # Chargement des donnees
    train_data, test_data = load_data()
    if train_data is None:
        return
    
    # Apercu rapide
    print("\nApercu du dataset d'entrainement:")
    print(f"   - Taille: {train_data.shape}")
    print(f"   - Variables numeriques: {train_data.select_dtypes(include=[np.number]).shape[1]}")
    print(f"   - Variables categorielles: {train_data.select_dtypes(include=['object']).shape[1]}")
    
    # Prix
    print(f"\nStatistiques des prix:")
    print(f"   - Prix moyen: ${train_data['SalePrice'].mean():,.0f}")
    print(f"   - Prix median: ${train_data['SalePrice'].median():,.0f}")
    print(f"   - Prix min: ${train_data['SalePrice'].min():,.0f}")
    print(f"   - Prix max: ${train_data['SalePrice'].max():,.0f}")
    
    # Variables manquantes
    analyze_missing_values(train_data, "Train")
    analyze_missing_values(test_data, "Test")

def demo_features():
    """Demonstration des features importantes"""
    print_separator("DEMO - FEATURES IMPORTANTES")
    
    train_data, _ = load_data()
    if train_data is None:
        return
        
    # Top 10 des correlations avec le prix
    numeric_features = train_data.select_dtypes(include=[np.number])
    correlations = numeric_features.corr()['SalePrice'].abs().sort_values(ascending=False)
    
    print("\nTop 10 des variables les plus correlees au prix:")
    for i, (feature, corr) in enumerate(correlations.head(11).items(), 1):
        if feature != 'SalePrice':  # On exclut SalePrice elle-meme
            print(f"   {i-1:2d}. {feature:<20} : {corr:.3f}")

def demo_neighborhoods():
    """Démonstration des quartiers"""
    print_separator("DEMO - ANALYSE PAR QUARTIER")
    
    train_data, _ = load_data()
    if train_data is None:
        return
    
    # Prix médian par quartier
    neighborhood_prices = train_data.groupby('Neighborhood')['SalePrice'].agg(['median', 'mean', 'count']).round(0)
    neighborhood_prices = neighborhood_prices.sort_values('median', ascending=False)
    
    print("\nTop 10 des quartiers les plus chers (prix median):")
    for i, (neighborhood, row) in enumerate(neighborhood_prices.head(10).iterrows(), 1):
        print(f"   {i:2d}. {neighborhood:<12} : ${row['median']:>8,.0f} (moy: ${row['mean']:>8,.0f}, n={int(row['count'])})")
    
    print("\nTop 5 des quartiers les moins chers:")
    for i, (neighborhood, row) in enumerate(neighborhood_prices.tail(5).iterrows(), 1):
        print(f"   {i:2d}. {neighborhood:<12} : ${row['median']:>8,.0f} (moy: ${row['mean']:>8,.0f}, n={int(row['count'])})")

def demo_quality_impact():
    """Demonstration de l'impact de la qualite"""
    print_separator("DEMO - IMPACT DE LA QUALITE")
    
    train_data, _ = load_data()
    if train_data is None:
        return
    
    # Prix par qualite generale
    quality_prices = train_data.groupby('OverallQual')['SalePrice'].agg(['median', 'count']).round(0)
    
    print("\nPrix median par qualite generale (1=Tres Pauvre, 10=Excellent):")
    for quality, row in quality_prices.iterrows():
        stars = "*" * min(quality, 5)  # Max 5 etoiles pour l'affichage
        print(f"   Qualite {quality:2d} {stars:<5} : ${row['median']:>8,.0f} ({int(row['count'])} maisons)")
    

def main():
    """Fonction principale du script de démo"""
    print("Bienvenue dans la démo du projet Predict-Price-Immo!")
    print("=" * 60)
    
    try:
        # Vérification des données
        train_data, test_data = load_data()
        if train_data is None:
            print("Impossible de charger les données. Vérifiez que train.csv et test.csv sont présents.")
            return
        
        print("\nDémonstrations disponibles:")
        print("   1. Exploration générale des données")
        print("   2. Features importantes")
        print("   3. Analyse par quartier")
        print("   4. Impact de la qualité")
        print("   5. Tout lancer")
        
        choice = input("\nChoisissez une option (1-5) ou Entrée pour tout: ").strip()
        
        if choice == '1':
            demo_exploration()
        elif choice == '2':
            demo_features()
        elif choice == '3':
            demo_neighborhoods()
        elif choice == '4':
            demo_quality_impact()
        else:
            # Lancer toutes les démos
            demo_exploration()
            demo_features()
            demo_neighborhoods()
            demo_quality_impact()
        
        print_separator("DEMO TERMINÉE")
        print("Pour une analyse complète, utilisez:")
        print("   • python main.py (version de base)")
        print("   • python optimize.py (version avancée)")
        
    except KeyboardInterrupt:
        print("\n\nDémo interrompue par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur durant la démo: {e}")

if __name__ == "__main__":
    main() 