# 소개
MLflow 튜토리얼 소개를 위한 페이지입니다.
1. MLflow Tracking
2. MLflow Registry
3. MLflow Models

## MLflow Tracking
실험 결과 자동 저장 및 관리 UI 화면 보기
```
$ cd ML (현재 경로:ML)
$ python mlflow_tracking.py
$ mlflow ui
$ cd ../DL (현재 경로:DL)
$ python mlflow_tracking.py
$ mlflow ui -h 0.0.0.0 -p 1010
```

## MLflow Registry
MLflow Registry 같은 경우에는 터미널 2개 띄운 후 실행한다.
```
[터미널 A]
$ cd .. (현재 경로:mlflow_tutorial)
$ mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root artifacts --host 0.0.0.0
```

mlflow_tracking.py에서 "mlflow.set_tracking_uri("http://IP주소:5000")" 코드의 주석을 풀어준다.  
```
[터미널 B]
$ cd ML (현재 경로:ML)
$ python mlflow_tracking.py
$ cd ../DL (현재 경로:DL)
$ python mlflow_tracking.py
```
IP주소:5000번 주소로 접속하면 결과를 확인할 수 있다.


## MLflow Inference
저장된 모델에 추론 하는 코드
```
$ cd ../ML (현재 경로:ML)
$ python mlflow_inference.py
$ cd ../DL (현재 경로:DL)
$ python mlflow_inference.py
```

```
$ mlflow models serve -m runs:/4ec92189f0b646dcb5d1a8ba0d6c878f/titanic_model --no-conda
$ mlflow models serve -m runs:/4ec92189f0b646dcb5d1a8ba0d6c878f/titanic_model
$ curl http://IP주소:1012/invocations -H 'Content-Type: application/json' -d '{"columns": ["Pclass", "Sex", "Fare", "SibSp", "Parch"], "data": [[1, 2, 3, 2 ,2], [1, 2, 4, 5, 6]]}'

```