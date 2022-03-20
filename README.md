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
$ mlflow ui (로컬 접속)
$ mlflow ui -h 0.0.0.0 -p 1000 (서버 접속)
```


## MLflow Inference
저장된 모델에 추론 하는 코드
```
$ cd ../ML (현재 경로:ML)
$ python mlflow_inference.py
$ cd ../DL (현재 경로:DL)
$ python mlflow_inference.py
```

모델 API 서버 띄우기
```
$ mlflow models serve -m runs:/72558aa709ab4d898c54c8b6f2a2d2ff/titanic_model --no-conda
$ curl http://IP주소:포트/invocations -H 'Content-Type: application/json' -d '{"columns": ["Pclass", "Sex", "Fare", "SibSp", "Parch"], "data": [[1, 2, 3, 2 ,2], [1, 2, 4, 5, 6]]}'
```


## MLflow Registry
MLflow Registry 같은 경우에는 터미널 2개 띄운 후 실행한다.
```
[터미널 A]
$ cd .. (현재 경로:mlflow_tutorial)
$ mkdir tracking_server
$ cd tracking_server
$ mlflow server --backend-store-uri file:/<경로>/MLflow_tutorial/tracking_server --default-artifact-root /<경로>/MLflow_tutorial/tracking_server --host 0.0.0.0 --port 1000
```

mlflow_tracking.py에서 "mlflow.set_tracking_uri("http://IP주소:포트")" 코드의 주석을 풀어준다.  
```
[터미널 B]
$ cd ML (현재 경로:ML)
$ python mlflow_tracking.py
$ cd ../DL (현재 경로:DL)
$ python mlflow_tracking.py
```
IP주소: 주소로 접속하면 결과를 확인할 수 있다.

