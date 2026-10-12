import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def analyze_drift():
    # 1. Donnees de reference (entrainement)
    iris = load_iris()
    df_ref = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    print("=== 1. Statistiques du jeu d'entrainement (Reference) ===")
    print(df_ref.describe())
    
    # 2. Simulation de nouvelles donnees de production avec derive (Data Drift)
    # Exemple : le capteur de sepal length derive de +35%
    np.random.seed(42)
    df_prod = df_ref.copy()
    df_prod['sepal length (cm)'] = df_prod['sepal length (cm)'] * 1.35 + np.random.normal(0, 0.2, len(df_prod))
    
    print("\n=== 2. Statistiques des donnees en Production ===")
    print(df_prod.describe())
    
    # 3. Detection de derive (Challenge 11 : Alerte simple de drift)
    threshold = 0.20 # Seuil d'alerte : 20% de variation de moyenne
    drift_detected = False
    
    print("\n=== 3. Rapport d'analyse de Data Drift ===")
    for col in iris.feature_names:
        mean_ref = df_ref[col].mean()
        mean_prod = df_prod[col].mean()
        delta_pct = abs(mean_prod - mean_ref) / mean_ref
        
        print(f"Feature: {col}")
        print(f"  Moyenne Ref: {mean_ref:.2f} | Moyenne Prod: {mean_prod:.2f} | Ecart: {delta_pct*100:.1f}%")
        
        if delta_pct > threshold:
            print(f"  [ALERTE DRIFT] Derive significative detectee sur '{col}' (> {threshold*100:.0f}%) !")
            drift_detected = True
            
    if drift_detected:
        print("\n=> RECOMMANDATION MLOps : Declencher le re-entrainement automatique du modele !")
    else:
        print("\n=> Pas de derive majeure : Le modele reste valide.")

if __name__ == '__main__':
    analyze_drift()
