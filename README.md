# GCAM-STITCHES-BASD Integration (DIPC)

The purpose of the repo is to be an easy yet comprehensive way to interact with [basd](https://github.com/JGCRI/basd), the Python package that allows you to **bias adjust** and **statistically downscale** arbitrary climate data, <u>without having to write a single line of code</u>, in a way that interacts with other climate tools [STITCHES](https://github.com/JGCRI/stitches) and [Hector](https://github.com/JGCRI/hector).

# Installation

Start by cloning this repository:

```
git clone https://github.com/bc3LC/GCAM_STITCHES_DIPC
```

## Option 1. Activate existing conda environment from the GCAM server 

```
conda activate xesmf
```

The conda environment has all the required packages installed and can be used readily.

## Option 2. Fresh install 

Set up a Conda virtual environment ([Conda user guide](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)), and activate it. **Specifically use Conda to manage this environment as we will use a Python package which is only available through Conda.**

Install the Conda dependent package,

```
conda install -c conda-forge xesmf
```

Next, install two Python packages, BASD and STITCHES. You can do this in multiple ways:

### Clone GitHub locally

This is the preferred way, as both BASD and STITCHES are new and quickly developing packages. This allows you to easily pull recent updates, switch branches, and even edit the source if needed. You may even want to default to the `dev` branch. Do this by cloning,

```
git clone https://github.com/JGCRI/basd.git
git clone https://github.com/JGCRI/stitches.git
```

and then installing in your virtual environment using develop mode by navigating to each respective repo and running, 

```
python -m pip install -e .
```


### GitHub Remote
You can also just install from github, and you could still specify the branch you want to use,

```
pip install git+https://github.com/JGCRI/basd.git
pip install git+https://github.com/JGCRI/stitches.git
```


# Usage

To use this repo you'll create a folder for an experiment you want to run that contains 6 input files that define your experiment details, and how you want the software to run. The `input/stitches-experiment` directory is an example of such a folder, which you can duplicate and then edit the configuration for other experiments.

## Editing Input Files

For a new experiment, duplicate the `stitches-experiment` folder in `input` and name it something informative to your experiment. Then you can edit the input files inside.

1. Open `run_manager_dipc.csv` and edit the available columns:
    * ESM
        * List the names of all the ESMs you want to use
        * Ex. CanESM5, MRI-ESM2-0
    * ESM_Input_Location
        * Set the path to where the data for the ESM is stored.
        * For usage in the DIPC, the recommended option is to follow the folder structure that you can find below. 
        * Leave the path empty if you want to use Pangeo to access the ESM data from cloud storage
    * Output_Location
        * Set the path where the output data should be stored
    * Reference Dataset
        * List the names of all the reference dataset you want to use (normally just one)
        * Most likely you will need to use W5E5v2, which is the standard for impact studies: https://data.isimip.org/search/simulation_round/ISIMIP3a/product/SecondaryInputData/climate_forcing/w5e5v2.0/ 
    * Reference_Input_Location
        * Set the path where the reference dataset is stored
        * This data must be stored locally on the DIPC to be used directly by BASD
    * Variable
        * List the climate variable short names that you want to use 
        * Ex. tas, pr
    * Scenario
        * List the scenarios that you want to use. They should be the same as the GCAM scenarios that you want to emulate with STITCHES
        * Ex. ssp245, ssp370, rcp245, GCAM6-SSP5-Hector, etc.
    * Ensemble
        * List the ensembles that you want to use
        * Ex. r1i1p1f1, r1i1p2f1
        * If using STITCHES, you do not need to provide ensemble names since the algorithm will look across all ensembles to look for the closest temperature trajectory.
    * target_period
        * List the year ranges for the reference data that you want to use (normally just use one, the range of the reference data)
        * The recommended range varies, and 20 years is usually seen as a minimum range and 30 years as the standard for climate modelling
        * Use the format start_year-end_year
        * Ex. 1990-2010
    * application_period
        * List the year ranges for which you want to adjust
        * You may use the "future period" of the CMIP6 archive for example, 2015-2100. 
        * You generally don't want to use the exact period as the target to avoid over-fitting. For example if the target period is 1980-2014, and you want to adjust a historical period, perhaps use 1950-2014 as the application period.
        * Use the format start_year-end_year
    * daily
        * Whether to save output data at the daily resolution
        * True or False
    * monthly
        * Whether to save output at the monthly resolution
        * True or False
    * stitched
        * Whether input ESM data was created using STITCHES
        * True or False

2. Open `slurm_parameters.csv` and enter details for the slurm scheduler that will be used for your jobs. The values provided by default work fine for the HPC/DIPC and can be adapted depending on the uses.   

4. The `attributes.csv` file allows you to specify the metadata in the output NetCDF files, both global, and variable-specific attributes. The file as found in the repo give examples of what might be included. However, there is great flexibility here. To add a new tag, add a column with the right name, and assign its value in any row you want it included in.

5. The file `encoding.csv` describes how the output NetCDF files will be encoded when saved. Mostly the defaults should be good for most applications. You may in particular want to change: 
    * `complevel`, which will change the level of compression applied to your data
    * `time_chunk`, `lat_chunk`, `lon_chunk`, which changes the chunk sizes of each dimension in the output NetCDF. This can effect how other programs interact and read in the data consequently. You can either enter an integer, or "max", which will use the full size of the that dimension in the given data.

6. The file `dask_parameters.csv` changes how [Dask](https://www.dask.org/), the Python package responsible for the parallelization in these processes, will split up (i.e. "chunk") the data. For machines with smaller RAM, you may want to lower from the defaults. The `dask_temp_directory` option gives you a chance to change where Dask stores intermediate files. For example, some computing clusters have a `/scratch/` directory where it is ideal to store temporary files that we don't want to be accidentally stored long term.

7. The file `variable_parameters.csv` may be edited, though the values set in the repo will be good for most cases, and more details are given in the file itself.

### STITCHES Integration
To use data generated by `STITCHES` you will then need to modify input file in your experiment folder called `temp_gcam_interface.csv`.
   * Copy-paste the temperature trajectories of the GCAM scenarios directly as new rows of the file. 
   * Make sure the scenario column includes the same names provided in the `run_manager_dipc.csv` file described above.
   * Create a column for each of your temperature trajectories, with the column headers matching the names you listed in `run_manager.csv` under "Scenario"

## Data inputs and folder structure (DIPC)

Once the inputs are ready, you will need to create on the DIPC a project folder where the data can be uploaded and where the results will be generated.
   * In the `scratch` directory of the `bc3lc` user of the DIPC, create a folder for your project (e.g. "Impacts-stitches")
   * Create in the folder `results` and within it the folders `stitches` and `basd`
   * Create a folder `data` and within it a folder `ISIMIP` (assuming you data inputs for the BASD process will be W5E5v2.0 from ISIMIP). The data folder can be used to store any other files required for follow-up modelling or tools.
   * Upload all the required data in the `data/ISIMIP` folder and ensure the path provided in `run_manager_dipc.csv` matches this path.
 

## Running

First run locally (on the GCAM server) `job-script-generation.py` from the root repository level, passing the name of your experiment folder as an argument, and making sure that your conda environment `xesmf` is activated. For example:

```
python code/python/job-script-generation.py stitches-experiment
```

After, you should see a new directory with the name of your experiment folder in the `intermediate` directory. It will contain 8 files when using STITCHES:

1. `run_manager_explicit_list.csv`
    * This will list out the details of each run that you requested explicitly. It lists one row per scenario and variable, and will be used by `basd.sh` to create one array per variable. 
    * Note that if you requested either `tasmin` and/or `tasmax`, these will be replaced by the variables `tasrange` and `tasskew`, which are used as an intermediate step for generating the `tasmin`/`tasmax` variables.
2. `run_manager_explicit_list_stitches.csv`
    * This will list out the details of each run that you requested explicitly. It lists on row per scenario, and will be used by `stitch_array.sh` 
    * Note that if you requested either `tasmin` and/or `tasmax`, these will be replaced by the variables `tasrange` and `tasskew`, which are used as an intermediate step for generating the `tasmin`/`tasmax` variables.
3. `basd.job`
    * This is a bash script responsible for submitting each of your requested tasks to the slurm scheduler.
4. `tasrange_tasskew.job`
    * This is a bash script which is responsible for submitting a script to generate the `tasrange` and `tasskew` variables from `tasmin` and `tasmax`, in the frequent case where `tasrange` and `tasskew` are not already generated. If they are already present, this script will do nothing.
5. `tasmin_tasmax.job`
    * This is a bash script which is responsible for submitting a script to generate the `tasmin` and `tasmax` variables, after `tasrange` and `tasskew` have gone through the bias adjustment and downscaling process.
6. `manager.job`
    * This is a bash script responsible for calling the above scripts in the correct order, `stitch_array.job` -> `tasrange_tasskew.job` -> `basd.job` -> `tasmin_tasmax.job` if using `STITCHES`.
7. `stitch.job` 
    * This is a bash script responsible for submitting a job to generate your STITCHED data. It runs one array at a time for all the scenarios (not the recommended approach, see `stitch_array.sh`)
8. `stitch_array` 
    * This is a bash script responsible for submitting a job to generate your STITCHED data. It runs one array per scenario and hence runs faster and more efficiently. 

It's good to check that these files were generated as you expected. For example that the `.job` files include all the slurm metadata that you input, and check the explicit list file to see the tasks you've requested, and how many there are.

Then, you're ready to run the STITCHES-BASD scripts for your GCAM scenarios, following these steps: 

1. Upload all the `GCAM_STITCHES_DIPC` folder from the GCAM server to your DIPC `scratch/bc3lc/[project_name]` (e.g. using the `server2dipc.sh` file provided in the repository)

2. Submit you running the `manager.job` script from the root repository level on the DIPC session. For example:
```
sbatch intermediate/test_run/manager.job
```
will submit all of your jobs, creating the `tasrange`/`tasskew` data along the way, and `tasmin`/`tasmax` at the end (if required). 

Alternatively you can run each of the `.job` scripts by hand. This may be especially useful if you either:

* Haven't requested `tasmin` or `tasmax`, in which case you can skip the two scripts responsible for those variables. Though running them in this case is fine, the scripts will do nothing.
* Or have lots of tasks you want to run, but maybe not all at once, or want to test the run with just one task. In this case you can run the `stitch_array.job` and then `basd.job` individually. In `basd.job` edit the `--array` flag like `--array=0` and run

    ```
    sbatch intermediate/test_run/basd.job
    ```

    which will submit just the first run, and the array argument can of course be set to your liking.

## Monitoring Job Progress

There is a hidden directory in this repo `.out`, which stores the files generated by the slurm scheduler. As each step runs, check out the logs in these files to check progress, and use `squeue` to see how long jobs have been running.

When you submit the `basd` portion of your jobs, you can view their progress, resource usage, etc. This is because the processes use [Dask](https://www.dask.org/), which give access the Dask dashboard. Once each `basd.job` script starts, find the respective log file in `.out`, and you'll quickly see two lines printed near the top of the file with commands that you can copy to access the Dask dashboard. Which one you will use varies depending on if you are running locally or remote. Generally you will use the remote command, which needs to be slightly edited manually, where you enter your remote details. You can then use your preferred browser to open the specified port and monitor the progress.

## Output

Navigate to the output paths that you set in the run manager file. This should be populated with NetCDF files as the run progresses. You can use software like NCO, with the `ncdump` command to view metadata, or you can use software like [Panopoly](https://www.giss.nasa.gov/tools/panoply/) to open and view the data plotted.
