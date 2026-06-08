from sklearn.datasets import load_iris

from sklearn.ensemble import RandomForestClassifier
import joblib

#let's load the iris dataset
iris=load_iris()

X=iris.data
y=iris.target

#train the model
model = RandomForestClassifier()
model.fit(X,y)

#save the model
joblib.dump(model, "save_model/iris_model.pkl")

print("model saved!")
