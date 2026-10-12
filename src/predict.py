import joblib
from sklearn.datasets import load_iris

# Chargement du modele serialise
model = joblib.load('models/iris_model.pkl')
iris = load_iris()

# Test de prediction sur le premier echantillon du dataset Iris
sample = [iris.data[0]]
prediction = model.predict(sample)
predicted_class = prediction[0]
class_name = iris.target_names[predicted_class]

print(f"Echantillon teste : {sample}")
print(f"Classe predite    : {predicted_class} ({class_name})")
