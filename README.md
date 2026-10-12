# TP 2 MLOps : Du modèle ML à la production

**Institution :** EMSI  
**Professeur :** BAKKALI YEDRI OTHMAN  
**Module :** DevOps / MLOps  
**Environnement :** Windows 11 / PowerShell / Docker  

---

## 🎯 Objectif du TP
Construire la chaîne MLOps complète du modèle d'Intelligence Artificielle jusqu'à la production :
$$\mathbf{Donn\acute{e}es} \longrightarrow \mathbf{Training} \longrightarrow \mathbf{MLflow} \longrightarrow \mathbf{Mod\grave{e}le} \longrightarrow \mathbf{API} \longrightarrow \mathbf{Docker} \longrightarrow \mathbf{Monitoring} \longrightarrow \mathbf{Drift}$$

---

## 📁 Structure du projet

```text
mlops-tp2/
├── .github/
│   └── workflows/
│       └── ci.yml             # Pipeline CI GitHub Actions (Entraînement + Tests)
├── data/                      # Stockage des jeux de données
├── models/
│   └── iris_model.pkl         # Modèle RandomForest sérialisé
├── src/
│   ├── train.py               # Entraînement avec suivi d'expériences MLflow
│   ├── predict.py             # Test local de prédiction
│   ├── api.py                 # API REST Flask (/predict et /health)
│   └── drift.py               # Analyse statistique et alerte de Data Drift
├── tests/
│   └── test_api.py            # Tests unitaires de validation
├── .gitignore                 # Exclusion des artefacts temporaires
├── Dockerfile                 # Image Docker basée sur python:3.12-slim
├── requirements.txt           # Dépendances (scikit-learn, mlflow, flask, etc.)
└── README.md                  # Documentation et compte-rendu complet
```

---

## 🚀 Guide d'exécution sous Windows

### 1. Installation de l'environnement
```powershell
cd C:\Users\HP\Desktop\mlops-tp2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Entraînement avec MLflow (3 runs : 50, 100, 200 estimateurs)
```powershell
python src/train.py
```

### 3. Visualisation dans MLflow UI
```powershell
mlflow ui
```
Ouvrir dans le navigateur : `http://127.0.0.1:5000` pour comparer les métriques (`accuracy`) des différents runs.

### 4. Test de prédiction locale
```powershell
python src/predict.py
```

### 5. Démarrage de l'API REST Flask
```powershell
python src/api.py
```
* **Vérification de l'état (Healthcheck) :** `http://127.0.0.1:5000/health`
* **Test de prédiction via POST :**
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -ContentType "application/json" -Body '{"features":[5.1, 3.5, 1.4, 0.2]}'
```

### 6. Conteneurisation avec Docker
```powershell
docker build -t mlops-tp2:1.0 .
docker run -d --name mlops-tp2 -p 5000:5000 mlops-tp2:1.0
```

### 7. Analyse du Monitoring et Data Drift
```powershell
python src/drift.py
```

---

## 📝 Réponses aux Questions du TP & Challenge

### Question 9 : Monitoring / Data Drift
1. **Quelle variable change ?**
   Dans notre simulation (`src/drift.py`), la variable `sepal length (cm)` a subi une dérive moyenne de +35% due à un changement de distribution ou à un capteur mal étalonné.
2. **Quel impact ?**
   Les performances du modèle vont chuter en production car il a été entraîné sur une distribution différente de celle qu'il observe en temps réel (perte de précision des prédictions).
3. **Quand réentraîner ?**
   Dès qu'un seuil statistique d'alerte (ex: variation de moyenne supérieure à 20% ou test de Kolmogorov-Smirnov avec p-value < 0.05) est franchi.

### Comparaison DevOps vs MLOps (Section 12)
* **DevOps :** Se concentre sur le cycle de vie du **code** :
  $$\text{Code} + \text{Tests} + \text{Git} + \text{CI/CD} + \text{Docker} + \text{Monitoring}$$
* **MLOps :** Intègre la gestion des **données** et des **modèles** en plus du code :
  $$\text{Code} + \text{Donn\acute{e}es} + \text{Mod\grave{e}le} + \text{Training} + \text{MLflow} + \text{CI/CD/CT} + \text{Monitoring} + \text{Drift} + \text{Retraining}$$
  *(CT = Continuous Training / Entraînement continu).*
