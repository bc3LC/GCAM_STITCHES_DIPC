"""
This file manages which scripts are used for each job
"""

# from pangeo import basd_pangeo
# from downloaded import basd_downloaded
# from stitched import basd_stitches
from stitched_ba import ba_stitches
from stitched_sd import sd_stitches

import os
import socket
import sys

import argparse
import dask
from dask.distributed import (Client, LocalCluster)
import numpy as np
import pandas as pd
import warnings
import shutil  # Utility functions script


if __name__ == "__main__":

    # Set high recursion limit so Dask is able to do things like find size of objects
    # sys.setrecursionlimit(3000)

# Paths =======================================================================================================
    root_dir = "/scratch/bc3lc/[project_name]/climate_integration_metarepo/"    
    stitches_dir = "/scratch/bc3lc/[project_name]/results/stitches/"    
    intermediate_path = 'intermediate'
    input_path = 'input'

    # root_dir = "C:/GCAM/Theo/[project_name]/python/climate_integration_metarepo"
    # run_name = "Impacts-stitches"
    # task_id = 0

# Get Run Details =============================================================================================

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_id', type=int, help='The number of the current task (row of the run_manager_explicit_list.csv file)')
    parser.add_argument('run_name', type=str, help='name of your experiment directory')
    
    parser.add_argument('--warn', action='store_const', dest='warn',
                        const=True, default=False,
                        help='flag to print warnings in log .out file')
    args = parser.parse_args()

    # Task index from SLURM array to run specific variable and model combinations
    task_id = args.task_id
    # Name of run directory
    run_name = args.run_name
    
    # Extract task details
    task_details = pd.read_csv(os.path.join(root_dir, intermediate_path, run_name, 'run_manager_explicit_list.csv')).iloc[task_id]
    # Extract Dask settings
    dask_settings = pd.read_csv(os.path.join(root_dir, input_path, run_name, 'dask_parameters.csv')).iloc[0]
    # Set names of dask temp folder
    dask_tmp = dask_settings.dask_temp_directory
    task_str = str(task_id)

    # Ignore non-helpful warnings
    if not args.warn:
        dask.config.set({'logging.distributed': 'error'})
        warnings.filterwarnings('ignore')

# Check if using Pangeo =======================================================================================

    # Boolean will be true when no input location is given
    using_pangeo = pd.isna(task_details.ESM_Input_Location) & ~(task_details.Variable in ['tasrange', 'tasskew'])
    # Boolean will be true when using STITCHED data
    using_stitches = task_details.stitched
    # When trying to use pangeo for tasrange/tasskew, data will actually be saved in intermediate
    if pd.isna(task_details.ESM_Input_Location) & (task_details.Variable in ['tasrange', 'tasskew']):
        task_details.ESM_Input_Location = os.path.join(intermediate_path, run_name, 'tasrange_tasskew')

# BIAS ADJUSTMENT =======================================================================================

    # Writing task details to log
    print(f'======================================================', flush=True)
    print(f'Task Details - BIAS ADJUSMENT:', flush=True)
    print(f'ESM: {task_details.ESM}', flush=True)
    print(f'Variable: {task_details.Variable}', flush=True)
    print(f'Scenario: {task_details.Scenario}', flush=True)
    try:
        print(f'Ensemble Member: {task_details.Ensemble}', flush=True)
    except AttributeError:
        pass
    print(f'Reference Period: {task_details.target_period}', flush=True)
    print(f'Application Period: {task_details.application_period}', flush=True)
    if using_pangeo:
        print('Getting Data From Pangeo', flush=True)
    elif using_stitches:
        print('Using STITCHED Data', flush=True)
    else: 
        print(f'Retrieving Data From {task_details.Reference_Input_Location}', flush=True)
    print(f'======================================================')

    # Create a subfolder for the task for Dask temp files 
    dask_tmp_task = os.path.join(dask_tmp, f'BA_{task_str}')
    print(dask_tmp_task)
    isExist = os.path.exists(dask_tmp_task)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dask_tmp_task)
        
    # Check to see if a non-default dask temporary directory is requested
    # If so, set it using dask config
    if not pd.isna(dask_settings.dask_temp_directory):
        # dask.config.set({'temporary_directory': f'{dask_settings.dask_temp_directory}'})
        dask.config.set({'temporary_directory': f'{dask_tmp_task}'})

    with LocalCluster(processes=True, threads_per_worker=1, n_workers = 30) as cluster, Client(cluster) as client:
        # Setting up dask.Client so that I can ssh into the dashboard
        port = client.scheduler_info()['services']['dashboard']
        host = client.run_on_scheduler(socket.gethostname)
        print("If running remotely use the below command to ssh into dashboard from a local terminal session")
        print(f"ssh -N -L 8000:{host}:{port} <username>@<remote name>", flush=True)
        print("Then use a browser to visit localhost:8000/ to view the dashboard.")
        print("If running locally, just visit the below link")
        print({client.dashboard_link})

        if using_pangeo:
            pass
            # Run pangeo script
            # basd_pangeo(task_details, run_name)
        elif using_stitches:
            # Run stitches script
            ba_stitches(task_details, run_name)
        else:
            pass
            # Run downloaded data script
            # basd_downloaded(task_details, run_name)

        client.close()
        cluster.close()
        
# STATISTICAL DOWNSCALING =======================================================================================
    
    
    # Writing task details to log
    print(f'======================================================', flush=True)
    print(f'Task Details - STATISTICAL DOWNSCALING:', flush=True)
    print(f'ESM: {task_details.ESM}', flush=True)
    print(f'Variable: {task_details.Variable}', flush=True)
    print(f'Scenario: {task_details.Scenario}', flush=True)
    try:
        print(f'Ensemble Member: {task_details.Ensemble}', flush=True)
    except AttributeError:
        pass
    print(f'Reference Period: {task_details.target_period}', flush=True)
    print(f'Application Period: {task_details.application_period}', flush=True)
    if using_pangeo:
        print('Getting Data From Pangeo', flush=True)
    elif using_stitches:
        print('Using STITCHED Data', flush=True)
    else: 
        print(f'Retrieving Data From {task_details.Reference_Input_Location}', flush=True)
    print(f'======================================================')
    
    # Create a subfolder for the task for Dask temp files 
    dask_tmp_task = os.path.join(dask_tmp, f'SD_{task_str}')
    print(dask_tmp_task)
    isExist = os.path.exists(dask_tmp_task)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dask_tmp_task)
        
    # Check to see if a non-default dask temporary directory is requested
    # If so, set it using dask config
    if not pd.isna(dask_settings.dask_temp_directory):
        # dask.config.set({'temporary_directory': f'{dask_settings.dask_temp_directory}'})
        dask.config.set({'temporary_directory': f'{dask_tmp_task}'})
    
    with LocalCluster(processes=True, threads_per_worker=1, n_workers = 30) as cluster, Client(cluster) as client:
        # Setting up dask.Client so that I can ssh into the dashboard
        port = client.scheduler_info()['services']['dashboard']
        host = client.run_on_scheduler(socket.gethostname)
        print("If running remotely use the below command to ssh into dashboard from a local terminal session")
        print(f"ssh -N -L 8000:{host}:{port} <username>@<remote name>", flush=True)
        print("Then use a browser to visit localhost:8000/ to view the dashboard.")
        print("If running locally, just visit the below link")
        print({client.dashboard_link})
    
        if using_pangeo:
            pass
            # Run pangeo script
            # basd_pangeo(task_details, run_name)
        elif using_stitches:
            # Run stitches script
            sd_stitches(task_details, run_name)
        else:
            pass
            # Run downloaded data script
            # basd_downloaded(task_details, run_name)
    
        client.close()
        cluster.close()

# Delete the STITCHES file =============================================================================================

    
    # Delete the BASD files once the Climate indicators are computed
    stitches_file = f'stitched_{task_details.ESM}_{task_details.Variable}_{task_details.Scenario}~~1.nc'
    stitches_path = os.path.join(stitches_dir, stitches_file)
    print(f'STITCHES Results for {task_details.ESM} with scenario {task_details.Scenario} and variable {task_details.Variable} being deleted from /scratch')
    try:
        shutil.rmtree(stitches_path)
    except OSError as e:
        print(f"Error removing STITCHES directory: {e}")

    print("STITCHES Deleted")
















