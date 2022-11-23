from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import mlflow
import os
from random import random


from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_mlflow_uri)


for ws in ml_client.workspaces.list():
    print(ws.name, ":", ws.location, ":", ws.description)

experiment_name = 'first_experiment'
mlflow.set_experiment(experiment_name)


with mlflow.start_run() as mlflow_run:
    mlflow.log_param("hello_param", "world")
    mlflow.log_metric("hello_metric", random())
    os.system(f"echo 'hello world' > helloworld.txt")
    mlflow.log_artifact("helloworld.txt")