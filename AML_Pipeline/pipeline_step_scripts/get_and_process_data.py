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
parser.add_argument('--output_data', dest='output_data', required=True)
parser.add_argument('--input_data', dest='input_data',  required=True)
parser.add_argument('--file_path',  type=str, required=True)

args, _ = parser.parse_known_args()
output_data = args.output_data
input_data = args.input_data
file_path = args.file_path

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


# Make directory on mounted storage for output dataset
os.makedirs(output_data, exist_ok=True)

# Save modified dataframe
raw_df.to_csv(os.path.join(output_data, 'output_data.csv'), index=False)