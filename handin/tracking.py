import pandas as pd
import mlflow

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(azureml_mlflow_uri)

# TODO: Set the experiment name
mlflow.set_experiment("sinie - wind_tracking")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler,OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer



# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="RFR_250"):
    df = pd.read_json("dataset.json", orient="split")

    metrics = [
        ("MAE", mean_absolute_error, []),
        ("RSE", mean_squared_error,[]),
        ("R2",r2_score,[])
    ]
    
    df = df.dropna()


    
    X = df[["Speed","Direction"]]
    y = df["Total"]
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse = False)
    ct = ColumnTransformer([('encoder transformer', encoder, ['Direction'])], remainder="passthrough")
    model = RandomForestRegressor(n_estimators=250)
    scaler = MaxAbsScaler()
    
    
    pipeline = Pipeline([
         ('column_transformer', ct),
         ("scaler", scaler),
         ('model',model)
    ])

    number_of_splits = 5
    i = 0
    mlflow.log_params({"splits":number_of_splits,"model":model,"scaler":scaler,"encoder":encoder})
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        model = pipeline.fit(X.iloc[train],y.iloc[train])
        #pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = model.predict(X.iloc[test])
        truth = y.iloc[test]
        from matplotlib import pyplot as plt 
        plt.plot(truth.index, truth.values, label="Truth")
        plt.plot(truth.index, predictions, label="Predictions")
        plt.savefig("curve{}.png".format(i))
        plt.show()
        mlflow.log_artifact("curve{}.png".format(i))
        i += 1
        
        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    
    # Log a summary of the metrics
    for name, _, scores in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/number_of_splits
            mlflow.log_metric(f"mean_{name}", mean_score)
            
    mlflow.sklearn.log_model(sk_model = model,
                             artifact_path = 'power_gen-pyfile-model',
                             registered_model_name = 'power_gen-pyfile-model')
