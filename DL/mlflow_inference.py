import torch
import mlflow
import numpy as np
import pickle


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
    
    test_x = np.array([test_x])
    loaded_model = mlflow.pyfunc.load_model('runs:/ec1037ef033e4a4cbc32647356644020/model')
    print(loaded_model.predict(test_x))