# Tuan Luong
# 02/19/2025

# Geolocation and transformation script and ISS LIS constant obtained from Timuthy Lang

import numpy as np
from netCDF4 import Dataset
import cv2 as cv
import matplotlib.pyplot as plt
import os
import pickle

##### Input params here
# Input orbit granule
granules = "20231101_151645"
dt = 1
video_size = (300, 300) # Video is resized for clarity. Measurements vectors are kept in 128x128 size

##### The rest of the scrip
img_size = (128,128)
if not os.path.isdir("dataset/"+granules):
    # Create dir for pre process data if it didn't exist
    os.mkdir("dataset/"+granules)

# Pre construct output video name
vid_out_name = "dataset/" + granules + "/events.avi"
video = cv.VideoWriter(vid_out_name, cv.VideoWriter_fourcc(*'XVID'), int(5/dt), video_size) # Set to 5 time real time

# Pickle file name
pickle_name = "dataset/" + granules + "/events.pik"

# Read the needed file
science_file_path = "dataset/ISS_LIS_SC_V2.2_" + granules + "_FIN.nc"

science_data = Dataset(science_file_path, 'r')

# Get relevant variables
lightning_event_TAI93_time_s = science_data.variables['lightning_event_TAI93_time'][:].data
lightning_event_pix_x = science_data.variables['lightning_event_x_pixel'][:].data
lightning_event_pix_y = science_data.variables['lightning_event_y_pixel'][:].data

# sort time (including duplicate)
time_sorted_ind = np.argsort(lightning_event_TAI93_time_s, stable=True)

sorted_lightning_event_TAI93_time_s = lightning_event_TAI93_time_s[time_sorted_ind]
elapsed_time_vec_s = sorted_lightning_event_TAI93_time_s - sorted_lightning_event_TAI93_time_s[0]

sorted_lightning_event_pix_x = lightning_event_pix_x[time_sorted_ind]
sorted_lightning_event_pix_y = lightning_event_pix_y[time_sorted_ind]

obs_time_s = sorted_lightning_event_TAI93_time_s[-1] - sorted_lightning_event_TAI93_time_s[0]



meas_time_vec = np.arange(0,obs_time_s + dt, dt)
meas_time_vec_TAI93 = meas_time_vec + sorted_lightning_event_TAI93_time_s[0]
#exit()
# Construct measurement table
meas_table = []
time = "TAI: {val:.2f} s"
for tt in meas_time_vec:
    # Find ind of entries within the time window
    sorted_ind_in_timerange = np.nonzero(np.logical_and(elapsed_time_vec_s >= tt,elapsed_time_vec_s < (tt + dt)))

    # Find pixel location within time window, if any
    if sorted_ind_in_timerange[0].size == 0:
        pix_loc_in_time_range = np.nan
        
    else:   
        pix_x_in_timerange = sorted_lightning_event_pix_x[sorted_ind_in_timerange[0]]
        pix_y_in_timerange = sorted_lightning_event_pix_y[sorted_ind_in_timerange[0]]
        pix_loc_in_time_range = np.unique(np.vstack((pix_x_in_timerange, pix_y_in_timerange)),axis=1).T
        
    meas_table.append(pix_loc_in_time_range)
    # Make video
    img = np.zeros(img_size, dtype='uint8')

    #print(pix_loc_in_time_range)
    if not np.any(np.isnan(pix_loc_in_time_range)):
        for pix in pix_loc_in_time_range:
            img[pix[1],pix[0]] = 255
    img = cv.resize(img, video_size)
    cv.putText(img,time.format(val=tt+sorted_lightning_event_TAI93_time_s[0]), (3, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    
    video.write(cv.cvtColor(img,cv.COLOR_GRAY2BGR))

# Release video object
video.release

# Create pickle file for post processing
meas_data = {"TAI93_time_s": meas_time_vec_TAI93, "elapsed_time_s": meas_time_vec, "meas_tab":meas_table}
pickle_file = open(pickle_name,'wb')
pickle.dump(meas_data, pickle_file)
pickle_file.close()

    

    
