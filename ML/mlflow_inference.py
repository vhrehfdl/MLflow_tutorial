import mlflow
import pandas as pd


if __name__ == '__main__':
    logged_model = 'runs:/72558aa709ab4d898c54c8b6f2a2d2ff/titanic_model'

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    test_x = pd.DataFrame({"Pclass": [2, 1], "Sex": [0, 1], "Fare": [3.3211, 3.3211], "SibSp": [3, 3], "Parch":[3, 3]})
    print(loaded_model.predict(test_x))