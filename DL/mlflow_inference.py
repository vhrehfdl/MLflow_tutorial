import mlflow
import pickle
import numpy as np


if __name__ == "__main__":
    max_len = 30
    test_x = "좋은 영화 입니다"

    with open('vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)

    test_x = test_x.split(" ")
    text_pipeline = lambda x: vocab(x)
    test_x = text_pipeline(test_x)
    
    if len(test_x) > max_len:
        test_x = test_x[0:max_len]
    else:
        test_x = test_x[0:] + ([0]*(max_len-len(test_x)))

    print(test_x)
    test_x = np.array([test_x])
    loaded_model = mlflow.pyfunc.load_model('runs:/eb143e5b147e464d8d94b75178307609/model')
    print(loaded_model.predict(test_x))