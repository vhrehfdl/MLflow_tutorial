
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir, index_col=["PassengerId"])
    test = pd.read_csv(test_dir, index_col=["PassengerId"])

    return train, test


def pre_processing(train, test):
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    test.loc[test["Sex"] == "male","Sex"]=0
    test.loc[test["Sex"] == "female", "Sex"] = 1
    
    feature_names = ["Pclass", "Sex", "Fare", "SibSp", "Parch"]
    train_x, train_y = train[feature_names], train["Survived"]
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1)

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    # Directory
    train_dir = "train.csv"
    test_dir = "test.csv"

    # Flow
    train, test = load_data(train_dir, test_dir)
    train_x, train_y, test_x, test_y = pre_processing(train, test)
    
    logged_model = 'runs:/ee58184aba224541b4f0dd8fabf5da0a/intent_classification'
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    test_x = pd.DataFrame({"Pclass":[2], "Sex":[0], "Fare": [3.3211], "SibSp": [3], "Parch":[3]})
    print(loaded_model.predict(test_x))