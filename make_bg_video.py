import numpy as np
from netCDF4 import Dataset
import cv2 as cv
import os

# 
granules = "20231101_195517"
video_size = (300,300) # Video is resized for clarity. Measurements vectors are kept in 128x128 size
dt = 10

##### The rest of the scrip
#
if not os.path.isdir("dataset/"+granules):
    # Create dir for pre process data if it didn't exist
    os.mkdir("dataset/"+granules)

# Pre construct output video name
vid_out_name = "dataset/" + granules + "/background.avi"
video = cv.VideoWriter(vid_out_name, cv.VideoWriter_fourcc(*'XVID'), 10, video_size) # Set to 5 time real time

# Read datafile
background_file_path = "dataset/ISS_LIS_BG_V2.2_" + granules + "_FIN.nc"
background_file = Dataset(background_file_path, 'r')

# Extract background image data
bg_img = background_file.variables['bg_data'][:].data

TAI93_time = background_file.variables['bg_data_summary_TAI93_time'][:].data
elapsed_time_vec_s = TAI93_time - TAI93_time[0]

obs_time_s = elapsed_time_vec_s[-1] - elapsed_time_vec_s[0]

meas_time_vec = np.arange(0,obs_time_s + dt, dt)

width, height = bg_img[1,:,:].shape
time = "TAI: {val:.0f} s"
i = 0
for tt in meas_time_vec:
    if tt > elapsed_time_vec_s[i+1]:
        i+=1

    cur_bg_img = bg_img[i,:,:]
    bg_8bit = ((cur_bg_img - cur_bg_img.min()) / (cur_bg_img.max() - cur_bg_img.min()) * 255).astype(np.uint8)# Rescale to 8bit using AGC
    #bg_8bit = ((cur_bg_img - 222.0) / (600 - 222) * 255).astype(np.uint8)# Rescale to 8bit using fixed gain

    bg_8bit = cv.cvtColor(bg_8bit,cv.COLOR_GRAY2BGR)

    bg_8bit = cv.resize(bg_8bit, video_size)
    cv.putText(bg_8bit,time.format(val=tt+TAI93_time[0]), (3, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    video.write(bg_8bit)

video.release()