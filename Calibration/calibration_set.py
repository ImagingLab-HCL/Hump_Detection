
"""
File Name: calibration_set.py
Script to set calibration

Steps involved -
1. Take the input video
2. Reading calibration.txt file
3. calculating  height, width, starting and ending position to get ROI
5. displaying the ROI on screen



Author  : Dhinakaran k
Date    : 21-10-2020

"""
import numpy as np
import cv2
import argparse



calib_param=[]
filerea_d = open("calibration.txt", "r")
for param_idx in range (0,5):
 data = filerea_d.readline()
 actual_val = int(data[-3:])
 calib_param.append(actual_val)



display = True
scale_percent = calib_param[0]
factor_to_start_height_mask = calib_param[1] / 100
factor_to_end_height_mask = calib_param[2] / 100
factor_to_start_width_mask = calib_param[3] / 100
factor_to_end_width_mask = calib_param[4] / 100

#print("ROI" , scale_percent,factor_to_start_height_mask,factor_to_end_height_mask,factor_to_start_width_mask,factor_to_end_width_mask )


if __name__ == "__main__":
    """    Main Program of this file 

            Parameters
             ----------
            filename   : input video filename given by user in cmd
           

            Returns
             -------

            function  will display the ROI  

            Notes
            -----
            This is function to display ROI, calibrated from user inputs 

    """
    # Reading Input from cmd Prompt
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input file path with filename ")
    args = parser.parse_args()
    input = args.input
    cap = cv2.VideoCapture(input)
    # mask image reading
    mask = cv2.imread("calibration_mask.png", 0)
    mask_line = cv2.imread("calibration_mask.jpg", 0)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask_line, kernel, iterations=5)
    mask_inv = cv2.bitwise_not(dilation)
    itr_frame_no = 0
    # main loop to fetch frames from video
    while (cap.isOpened()):
        image_format = '.jpg'
        itr_frame_no = itr_frame_no + 1
        name_img = str(itr_frame_no)
        imageName = name_img + image_format
        ret_frame, frame = cap.read()
        # Invalid or End of Video indication message
        if np.shape(frame) == ():
            print(" Hump Detection Execution Done ")
            break
        # Image cropping based the calibrated parameter to get ROI
        resized_width = int(frame.shape[1] * scale_percent / 100)
        resized_height = int(frame.shape[0] * scale_percent / 100)
        dim = (resized_width, resized_height)
        resized_width = resized_width
        resized_height = resized_height
        tmpImg = np.zeros((resized_height, resized_width), np.uint8)
        resized_img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        oriimage_copy = resized_img.copy()
        gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        actualHeight = int(resized_height * factor_to_start_height_mask)
        actualStartWidth = int(resized_width * factor_to_start_width_mask)
        actualEndWidth = int(resized_width * factor_to_end_width_mask)
        actualEndHeght = int(resized_height * factor_to_end_height_mask)
        start_y = actualHeight + 1
        start_x = actualStartWidth + 1
        Heightactual_crop = ((resized_height - actualEndHeght) - start_y)
        widthactual_crop = ((resized_width - actualEndWidth) - start_x)
        crop_img = gray_image[start_y:start_y + Heightactual_crop, start_x:start_x + widthactual_crop]
        crop_img = cv2.bitwise_or(crop_img, crop_img, mask=mask)
        crop_img_dublicate = oriimage_copy[start_y:start_y + Heightactual_crop, start_x:start_x + widthactual_crop]
        start_point = (start_x, start_y)
        end_point = (start_x + widthactual_crop, start_y + Heightactual_crop)
        image = cv2.rectangle(oriimage_copy, start_point, end_point, (0, 0, 255), 4)
        cv2.imshow(" Hump_Detection_v1.0 ", oriimage_copy)
        cv2.waitKey(1)