#! /bin/sh


######################################### METAREPO #########################

# ALL REPOSITORY (Modify paths accordingly)
scp -r python/climate_integration_metarepo bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/GCAM_7.2_Impacts/


########################################s DATA   ##########################
# ISIMIP (Modify paths accordingly)
scp -r data/ISIMIP/tas_W5E5v2.0_1990-2010.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/GCAM_7.2_Impacts/data/ISIMIP/
scp -r data/ISIMIP/pr_W5E5v2.0_1990-2010.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/GCAM_7.2_Impacts/data/ISIMIP/
scp -r data/ISIMIP/hurs_W5E5v2.0_1990-2010.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/GCAM_7.2_Impacts/data/ISIMIP/
scp -r data/ISIMIP/sfcWind_W5E5v2.0_1990-2010.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/GCAM_7.2_Impacts/data/ISIMIP/
scp -r data/GPP/SSR/rsds_W5E5v2.0_1990-2010.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/GCAM_7.2_Impacts/data/ISIMIP/
scp -r data/ISIMIP/rsds_W5E5v2.0_1990-2010.nc bc3lc@atlas-edr.sw.ehu.es:/scratch/bc3lc/GCAM_7.2_Impacts/data/ISIMIP/

