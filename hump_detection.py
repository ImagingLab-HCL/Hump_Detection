"""
File Name: hump_detecting.py
Script to detect Hump from given input videos

Steps involved -
1. Take the input video
2. Calculate ROI based on calibration parameters
3. calculate Sobel gradient
5. Finding Horizontal line using countours
6. classification by Condition checking with Hump properties


Author  : Dhinakaran k
Date    : 21-10-2020

"""

import numpy as np
import cv2
import argparse
from scipy.spatial import distance
import time

# Reading calibration file to set ROI 
tmp_ary=[]
filerea_d = open(r"./Calibration/calibration.txt", "r")
for x in range (0,5):
 data = filerea_d.readline()
 actual_val = int(data[-3:])
 tmp_ary.append(actual_val)



display = True
scale_percent = tmp_ary[0]
factor_to_start_height_mask = tmp_ary[1] / 100
factor_to_end_height_mask = tmp_ary[2] / 100
factor_to_start_width_mask = tmp_ary[3] / 100
factor_to_end_width_mask = tmp_ary[4] / 100

#Enable below line of code to see ROI Region in console window 
#print("ROI" , scale_percent,factor_to_start_height_mask,factor_to_end_height_mask,factor_to_start_width_mask,factor_to_end_width_mask )


if __name__ == "__main__":
    """    Main Program of this file 

            Parameters
             ----------
            filename   : input video filename given by user in cmd
            roidisplay :  display the processed ROI if this flag enables

            Returns
             -------

            function  will display the results 

            Notes
            -----
            This is function to detect the Hump by implementing the above steps

    """
    # Reading Input from cmd Prompt
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input file path with filename ")
    parser.add_argument("--display", required=True, help="Do you want to see ROI please enter 1 else 0 ")
    args = parser.parse_args()
    input = args.input
    roidisplay = int(args.display)
    cap = cv2.VideoCapture(input)
    # mask image reading
    mask = cv2.imread("./Calibration/calibration_mask.png", 0)
    mask_line = cv2.imread("./Calibration/calibration_mask.jpg", 0)
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
		# Frame  resizing to optimize further process 
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
		# Image cropped to execute algorithm on ROI 
        crop_img = gray_image[start_y:start_y + Heightactual_crop, start_x:start_x + widthactual_crop]
        crop_img = cv2.bitwise_or(crop_img, crop_img, mask=mask)
        crop_img_dublicate = oriimage_copy[start_y:start_y + Heightactual_crop, start_x:start_x + widthactual_crop]
        width_crop = int(crop_img.shape[1])
        height_crop = int(crop_img.shape[0])
        offset_center = 20
        cneter_x = int(width_crop / 2)
        center_y = int(height_crop / 2)
        leftCenter_x = cneter_x - offset_center
        rightCenter_x = cneter_x + offset_center
        scale_percent_start_width = 70
        offset_width = 50
        width_start = width_crop - int(width_crop * scale_percent_start_width / 100)
        width_end = width_crop - width_start + offset_width
        # Sobel applied on ROI
        sobel_imagey = cv2.Sobel(crop_img, cv2.CV_8UC1, 0, 1)
        sobel_imagex = cv2.Sobel(crop_img, cv2.CV_8UC1, 1, 0)
        sobel_image = cv2.add(sobel_imagex, sobel_imagey)
        sobel_image = cv2.bitwise_or(sobel_image, sobel_image, mask=mask_inv)
        horizontal_size = 5
        ret, sobel_bw = cv2.threshold(sobel_image, 80, 255, cv2.THRESH_BINARY)
        ori_bw = sobel_bw.copy()
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 2))
        # Apply morphology operations on sobel outputs
        horizontal = cv2.dilate(sobel_bw, horizontalStructure)
        ret, horz_bw = cv2.threshold(horizontal, 80, 255, cv2.THRESH_BINARY)
        connectivity = 8
        output_horz_bw = cv2.connectedComponentsWithStats(horz_bw, connectivity, cv2.CV_32S)
        num_stats = output_horz_bw[0]
        labels = output_horz_bw[1]
        stats = output_horz_bw[2]
        new_image = horz_bw.copy()
        # entry time to calculate execution time
        t0 = time.time()
        # Hump classify by comparing with Hump parameters
		
		# Features to implement ML - TBD 
		   # CC_STAT_HEIGHT
		   # CC_STAT_WIDTH
		   # CC_STAT_TOP
		   # CC_STAT_LEFT
		   # cenX cenY
        for label in range(num_stats):
            cenX = int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] / 2)
            cenY = int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] / 2)
            if stats[label, cv2.CC_STAT_HEIGHT] >= 15 and stats[label, cv2.CC_STAT_WIDTH] <= 30:
                new_image[labels == label] = 0
            elif stats[label, cv2.CC_STAT_HEIGHT] >= 30:
                new_image[labels == label] = 0
            elif stats[label, cv2.CC_STAT_WIDTH] <= 60:
                new_image[labels == label] = 0
            elif stats[label, cv2.CC_STAT_TOP] <= 10:
                new_image[labels == label] = 0
            elif stats[label, cv2.CC_STAT_LEFT] <= 10 and stats[label, cv2.CC_STAT_WIDTH] <= 50:
                new_image[labels == label] = 0
            elif stats[label, cv2.CC_STAT_LEFT] >= int(width_crop / 2) + 20 and stats[label, cv2.CC_STAT_LEFT] + stats[
                label, cv2.CC_STAT_WIDTH] >= width_crop - 10:
                val = stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH]
                new_image[labels == label] = 0
            elif cenX <= 150 or cenX >= rightCenter_x:
                new_image[labels == label] = 0
            else:
                new_image = cv2.circle(new_image, (cenX, cenY), 2, (255, 255, 255), -1)
        output_new_image = cv2.connectedComponentsWithStats(new_image, connectivity, cv2.CV_32S)
		# output_new_image will have many white dots along with Hump 
		# condition to eliminate other than hump labels 
        num_stats = output_new_image[0] 
        labels = output_new_image[1]
        stats = output_new_image[2]
        for label in range(num_stats):
            contour_area = stats[label, cv2.CC_STAT_AREA]
            if contour_area > 60000:
                continue
            cenX = int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] / 2)
            cenY = int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] / 2)
            distance_x = distance.euclidean(cenX, cneter_x)
            distance_y = distance.euclidean(cenY, height_crop - 5)
            if distance_y > 100:
                new_image[labels == label] = 0
        count = 0
		# Merging cropped region on original image 
        for i in range(0, height_crop):
            for j in range(0, width_crop):
                if new_image.item(i, j) == 255:
                    crop_img_dublicate.itemset(i, j, 1, 255);
                    count = count + 1
        oriimage_copy[start_y:start_y + Heightactual_crop, start_x:start_x + widthactual_crop, :] = crop_img_dublicate
        # exit time to calculate execution time
        t1 = time.time()
        total = t1 - t0
        # FPS calculation
        FPS = 1 / total
        if FPS > 60:
            FPS = 60
        start_point = (start_x, start_y)
        end_point = (start_x + widthactual_crop, start_y + Heightactual_crop)

        # overlay results 
        image = cv2.putText(oriimage_copy, 'FPS', (5, 30), 1,
                            1, (128, 128, 255), 2, cv2.LINE_AA)
        image = cv2.putText(oriimage_copy, str(int(FPS)), (45, 30), 1,
                            1, (128, 128, 255), 2, cv2.LINE_AA)
        if roidisplay == 1:
            image = cv2.putText(oriimage_copy, ' Processed ROI ', (start_x, start_y - 20), 2,
                            2, (255, 0, 0), 1, cv2.LINE_AA)
            image = cv2.rectangle(oriimage_copy, start_point, end_point, (0, 0, 255), 4)

        if count > 0:
            image = cv2.putText(oriimage_copy, ' HUMP DETECED ', (100, 100), 2,
                                2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow(" Hump_Detection_v1.0 ", oriimage_copy)
        cv2.waitKey(1)

	# Memory Free up
    cap.release()
    cv2.destroyAllWindows()


