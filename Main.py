from flask import Flask,jsonify, request
from Model.Dataset import tasks
from Model.Dataset import Iris
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np


app = Flask(__name__)
Swagger(app)
# CORS(app)

@app.route("/")
def helloWorld():
    return "Hello World"


@app.route("/get/iris")
def getTask():
    """
    Ini adalah coba coba
    ---
    tags:
        - Rest Controller
    parameter:
    responses:
        200:
            description: success bro
    """
    return jsonify({'iris':tasks})


@app.route("/input/iris",methods=['POST'])
def inputIris():
    newIris = request.get_json()

    p1 = newIris["petalLength"]
    p2 = newIris["petalWidth"]
    s1 = newIris["sepalLength"]
    s2 = newIris["sepalWidth"]

    newPetal = Iris(p1,p2,s1,s2)

    tasks.append(newPetal.__dict__)

    return "Success Add Iris"

@app.route("/edit/iris/<int:id>",methods=['PUT'])
def editIris():
    newIris = request.get_json()

    p1 = newIris["petalLength"]
    p2 = newIris["petalWidth"]
    s1 = newIris["sepalLength"]
    s2 = newIris["sepalWidth"]

    newPetal = Iris(p1, p2, s1, s2)

    tasks[id] = newPetal.__dict__

    return "Success Edit Iris"


@app.route("/delete/iris/<int:id>",methods=['DELETE'])
def deleteIris():
    del tasks[id]
    return "Success Delete Iris"


@app.route('/predict/task', methods=['POST'])
def predict():
    """
    Ini Adalah Endpoint Untuk Memprediksi IRIS
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Petal
          required:
            - petalLength
            - petalWidth
            - sepalLength
            - sepalWidth
          properties:
            petalLength:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            petalWidth:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            sepalLength:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
            sepalWidth:
              type: int
              description: Please input with valid Sepal and Petal Length-Width.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    petalLength = new_task['petalLength']
    petalWidth = new_task['petalWidth']
    sepalLength = new_task['sepalLength']
    sepalWidth = new_task['sepalWidth']


    X_New = np.array([[petalLength,petalWidth,sepalLength,sepalWidth]])

    clf = joblib.load('knnClassifier.pkl')

    resultPredict = clf[0].predict(X_New)

    return jsonify({'message': format(clf[1].target_names[resultPredict])})


if __name__ == '__main__':
    app.run()