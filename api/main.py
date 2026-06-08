from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app=FastAPI()


model = joblib.load("save_model/iris_model.pkl")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_iris(input_data: IrisInput):

    features = [[input_data.sepal_length, 
                input_data.sepal_width, 
                input_data.petal_length, 
                input_data.petal_width]]
    
    prediction = model.predict(features)
    
    flower_names = {
        0:"Setosa",
        1:"versicolor",
        2:"Virginica"
    }

    return{
        "prediction": flower_names[int(prediction)]}
    