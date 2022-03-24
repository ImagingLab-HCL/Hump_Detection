

In Scope 
Detection of Hump with / without markings at Nighttime in urban area, streets, inside cities road etc., 

Installation steps:
Install anaconda prompt in your system.
Anaconda link : https://docs.anaconda.com/anaconda/install/windows/

While installing anaconda,Select Instalation only for current user.

Open the anaconda prompt,Create the virtual enviroment.
   conda create -n HumpDetection python=3.7
   conda activate HumpDetection
   cd /Downloade/Library/
Makesure that folder contains requirements.txt file.
   pip install -r requirements.txt
   
Setting up environemntal:
Open Anaconda prompt.
Activate the HumpDetection environment.(conda activate HumpDetection)
Change the directory to HumpDetection.(cd {HUMP_DETECTION_PATH})

How to execute:
1) Change the directory to Calibration folder.(cd {HUMP_DETECTION_PATH}/Calibration/)
2) Calibartion (python calibration_set.py --input video_name)
Example : python calibration_set.py --input ./../Input/Sample_video.avi
3) Change the directory to HumpDetection.(cd {HUMP_DETECTION_PATH})
4) Hump detection (python hump_detection.py --input video_name --display 1)
Example : python hump_detection.py --input ./Input/Sample_video.avi --display 1

Note: Make sure calibration folder contains "calibration_mask.jpg" and "calibration_mask.png" image file.