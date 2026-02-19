def set_names(run_object):
    """
    Function for setting paths and file names based on the run details
    """
    # We want to edit global variables
    global temp_intermediate_dir, output_ba_path, output_basd_path
    global output_day_ba_file_name, output_mon_ba_file_name, output_day_basd_file_name, output_mon_basd_file_name
    global input_ref_data_path, input_sim_data_path
    # Temporary intermediate results directory
    temp_intermediate_dir = os.path.join(run_object.Output_Location, run_object.Reference_Dataset, 
                                         run_object.ESM, run_object.Scenario, 
                                         f'{run_object.Variable}_STITCHES_temp_intermediate')
    # Full output path for bias adjusted data
    output_ba_path = os.path.join(run_object.Output_Location, run_object.Reference_Dataset,
                                  run_object.ESM, run_object.Scenario, 'ba')
    # Start and End years
    start, end = str.split(run_object.application_period, '-')
    # Output file name for daily and monthly bias adjusted data
    output_day_ba_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_daily_{start}_{end}.nc'
    output_mon_ba_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_monthly_{start}_{end}.nc'
    # Full output path for downscaled data
    output_basd_path = os.path.join(run_object.Output_Location, run_object.Reference_Dataset,
                                    run_object.ESM, run_object.Scenario, 'basd')
    # Output file name for daily and monthly downscaled data
    output_day_basd_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_daily_{start}_{end}.nc'
    output_mon_basd_file_name = f'{run_object.ESM}_STITCHES_{run_object.Reference_Dataset}_{run_object.Scenario}_{run_object.Variable}_global_monthly_{start}_{end}.nc'
    # Input location for observational reference dataset
    # input_ref_data_path = os.path.join(run_object.Reference_Input_Location, run_object.Variable)
    input_ref_data_path = os.path.join(run_object.Reference_Input_Location)
    # Input location for simulated datasets
    input_sim_data_path = run_object.ESM_Input_Location
    
# Function that creates new directories
def create_directories():
    """
    Try to create any new directories. If already exist, do nothing.
    """
    try:
        os.makedirs(temp_intermediate_dir)
    except FileExistsError:
        pass
    try:
        os.makedirs(output_ba_path)
    except FileExistsError:
        pass
    try:
        os.makedirs(output_basd_path)
    except FileExistsError:
        pass











##############################################################
##############################################################
#  IF?basd?main
##############################################################
##############################################################
root_dir = IF_PATH    
stitches_dir = os.path.join(IF_PATH, "output/stitches/")    
intermediate_path = os.path.join(IF_PATH, 'intermediate')
input_path = os.path.join(IF_PATH, 'input')

    # root_dir = "C:/GCAM/Theo/GCAM_7.2_Impacts/python/climate_integration_metarepo"
    # run_name = "heat_ineq"
    # task_id = 0

# Get Run Details =============================================================================================

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('task_id', type=int, help='The number of the current task (row of the run_manager_explicit_list.csv file)')
parser.add_argument('run_name', type=str, help='name of your experiment directory')

parser.add_argument('--warn', action='store_const', dest='warn',
                    const=True, default=False,
                    help='flag to print warnings in log .out file')
# args = parser.parse_args()
class Args:
    task_id = 0
    run_name = 'heat_ineq'

args = Args()

    # Task index from SLURM array to run specific variable and model combinations
task_id = args.task_id
# Name of run directory
run_name = args.run_name

# Extract task details
task_details = pd.read_csv(os.path.join(intermediate_path, run_name, 'run_manager_explicit_list.csv')).iloc[task_id]
# Extract Dask settings
dask_settings = pd.read_csv(os.path.join(input_path, run_name, 'dask_parameters.csv')).iloc[0]
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

# # Writing task details to log
# print(f'======================================================', flush=True)
# print(f'Task Details - BIAS ADJUSMENT:', flush=True)
# print(f'ESM: {task_details.ESM}', flush=True)
# print(f'Variable: {task_details.Variable}', flush=True)
# print(f'Scenario: {task_details.Scenario}', flush=True)
# try:
#     print(f'Ensemble Member: {task_details.Ensemble}', flush=True)
# except AttributeError:
#     pass
# print(f'Reference Period: {task_details.target_period}', flush=True)
# print(f'Application Period: {task_details.application_period}', flush=True)
# if using_pangeo:
#     print('Getting Data From Pangeo', flush=True)
# elif using_stitches:
#     print('Using STITCHED Data', flush=True)
# else: 
#     print(f'Retrieving Data From {task_details.Reference_Input_Location}', flush=True)
# print(f'======================================================')

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
