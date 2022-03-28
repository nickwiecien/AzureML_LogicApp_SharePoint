# Step 1. Get Data
# Sample Python script designed to load data from a target data source,
# and export as a tabular dataset

from azureml.core import Run, Workspace, Datastore, Dataset
import pandas as pd
import os
import argparse
import numpy as np

# Parse input arguments
parser = argparse.ArgumentParser("Get raw data from a selected datastore and register in AML workspace")
parser.add_argument('--training_data', dest='training_data', required=True)
parser.add_argument('--forecasting_data', dest='forecasting_data', required=True)
parser.add_argument('--input_data', dest='input_data',  required=True)
parser.add_argument('--file_path',  type=str, required=True)
parser.add_argument('--timestamp_column',  type=str, required=True)

args, _ = parser.parse_known_args()
forecasting_data = args.forecasting_data
training_data = args.training_data
input_data = args.input_data
file_path = args.file_path
timestamp_column  = args.timestamp_column

# Get current run
current_run = Run.get_context()

# Get associated AML workspace
ws = current_run.experiment.workspace

# Connect to default blob datastore
ds = ws.get_default_datastore()

# Get file and save to input dataset
os.makedirs(input_data, exist_ok=True)
ds.download(input_data, prefix=file_path)

#Load input  file data
raw_df = None
if '.xls' in file_path:
    raw_df = pd.read_excel(os.path.join(input_data, file_path))
else:
    raw_df = pd.read_csv(os.path.join(input_data, file_path))

#TRANSFORM DATA & APPLY LOGIC BELOW
forecasting_df = raw_df.tail(10)
cutoff_date = forecasting_df.iloc[0][timestamp_column]
training_df = raw_df[raw_df[timestamp_column]<cutoff_date]


# Make directory on mounted storage for output dataset
os.makedirs(training_data, exist_ok=True)
os.makedirs(forecasting_data, exist_ok=True)

# Save modified dataframes
training_df.to_csv(os.path.join(training_data, 'training_data.csv'), index=False)
forecasting_df.to_csv(os.path.join(forecasting_data, 'forecasting_data.csv'), index=False)