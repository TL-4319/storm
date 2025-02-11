import numpy as np
from netCDF4 import Dataset

# 
granules = "20231101_151645"

science_file_path = "dataset/ISS_LIS_SC_V2.2_" + granules + "_FIN.nc"
#background_file_path = "dataset/ISS_LIS_BG_V2.1_" + granules + "_FIN.nc"

datafile = Dataset(science_file_path, 'r')

print(datafile.variables['bg_summary_TAI93_time'][:].data)