# Import required packages
from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
import shutil
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score

# #Parse input arguments
parser = argparse.ArgumentParser("Register new model and generate forecast")
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--target_column', type=str, required=True)
parser.add_argument('--forecast_results_data', dest='forecast_results_data', required=True)

args, _ = parser.parse_known_args()
model_name = args.model_name
target_column = args.target_column
forecast_results_data = args.forecast_results_data

#Get current run
current_run = Run.get_context()

#Get parent run
parent_run = current_run.parent

#Get associated AML workspace
ws = current_run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

#Get testing dataset
forecast_datset = current_run.input_datasets['Forecast_Data']
forecast_df = forecast_datset.to_pandas_dataframe()

#Separate inputs from outputs (actuals). Create separate dataframes for testing champion and challenger.

#Get new AutoML model
for c in parent_run.get_children():
    if 'AutoML' in c.name:
        best_child_run_id = c.tags['automl_best_child_run_id']
        automl_run = ws.get_run(best_child_run_id)
        automl_run.download_files('outputs', output_directory='outputs', append_prefix=False)
        model = joblib.load('outputs/model.pkl')
        print(best_child_run_id)
        print()
        model_tags = {'Parent Run ID': parent_run.id, 'AutoML Run ID': best_child_run_id}
        
#Generate new forecast
preds = model.predict(forecast_df)
updated_df = forecast_df
updated_df['PREDICTIONS'] = preds

Model.register(model_path="outputs",
        model_name=model_name,
        tags=model_tags,
        description=model_name,
        workspace=ws)

# Make directory on mounted storage for output dataset
os.makedirs(forecast_results_data, exist_ok=True)

# Save modified dataframe
updated_df.to_csv(os.path.join(forecast_results_data, 'result_data.csv'), index=False)