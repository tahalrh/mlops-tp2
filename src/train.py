import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train_model(n_estimators=100):
    mlflow.set_experiment('Iris_MLOps')
    iris = load_iris()
    X, y = iris.data, iris.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with mlflow.start_run(run_name=f"RF_{n_estimators}_estimators"):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        
        # Enregistrement dans MLflow
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metric('accuracy', acc)
        
        # Enregistrement du modele avec cloudpickle pour compatibilite
        try:
            mlflow.sklearn.log_model(model, 'model', serialization_format='cloudpickle')
        except Exception:
            mlflow.sklearn.log_model(model, 'model')
        
        # Sauvegarde du modele serialise
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/iris_model.pkl')
        print(f"Run MLflow termine [n_estimators={n_estimators}] -> Accuracy: {acc:.4f}")
        return acc

if __name__ == '__main__':
    print("--- Demarrage de l'entrainement avec MLflow ---")
    # Execute les 3 runs demandes par le TP (50, 100, 200)
    for n in [50, 100, 200]:
        train_model(n)
    print("--- Entrainement termine avec succes ! Modeles enregistres dans models/ et MLflow ---")
