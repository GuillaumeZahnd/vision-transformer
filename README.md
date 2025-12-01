# Vision Transformer

## Python environment

```sh
python -m pip install --upgrade setuptools pip
mkdir .venv
pipenv install -d --python 3.12
```

## Hyperparameters

All hyperparameters are defined in the YAML files contained in the `config` folder and subfolders.

## Running the main training script

```sh
pipenv shell
python vit.py
```

## Monitor runs and logs with MLFlow Tracking

1. Start a local tracking server:

```sh
mlflow ui --port 8080 --backend-store-uri sqlite:///mlflow.db
```

2. Access the  interface via a web browser:

```sh
http://127.0.0.1:8080
```
