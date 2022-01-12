import mlflow
import collections
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
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


def build_model(train_x, train_y):
    model = DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


def evaluation(model, test_x, test_y):
    pred_y = model.predict(test_x)
    score = f1_score(test_y, pred_y, average='weighted')
    return score


if __name__ == '__main__':
    # MLflow
    # mlflow.set_tracking_uri("http://10.20.81.77:5000")
    mlflow.set_experiment('titanic')
    mlflow.log_param("env", "local")

    # Directory
    train_dir = "train.csv"
    test_dir = "test.csv"

    # Flow
    train, test = load_data(train_dir, test_dir)
    train_x, train_y, test_x, test_y = pre_processing(train, test)
    model = build_model(train_x, train_y)
    score = evaluation(model, test_x, test_y)

    pred_x = model.predict(test_x)

    mlflow.log_param("train", train_dir)
    mlflow.log_param("train num", len(train_x))
    mlflow.log_param("class", collections.Counter(train_y))
    mlflow.log_param("class num", len(set(train_y)))

    mlflow.log_metric("f1 score", score)
    mlflow.log_artifact(train_dir)
    mlflow.sklearn.log_model(model, "titanic_model")