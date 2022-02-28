import mlflow
import pandas as pd


if __name__ == '__main__':
    logged_model = 'runs:/1c89c8eed61e423a8e7fea45f0d24db6/titanic_model'

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    test_x = pd.DataFrame({"Pclass": [2, 1], "Sex": [0, 1], "Fare": [3.3211, 3.3211], "SibSp": [3, 3], "Parch":[3, 3]})
    print(loaded_model.predict(test_x))

    # mlflow models serve -m ~/mlruns/0/your_uuid/artifacts/model -h 0.0.0.0 -p 8001