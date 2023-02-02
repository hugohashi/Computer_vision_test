#!/usr/bin/env python3

import numpy as np
import cv2

#define red upper and lower boundaries
red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)


#other values
# red_lower = np.array([0, 100, 100], np.uint8)
# red_upper = np.array([10, 255, 255], np.uint8)

# red_lower = np.array([160, 100, 100], np.uint8)
# red_upper = np.array([180, 255, 255], np.uint8)

#define yellow upper and lower boundaries
yellow_lower = np.array([5, 50, 50], np.uint8)
yellow_upper = np.array([15, 255, 255], np.uint8)

#other values
# yellow_lower = np.array([15,100,100], np.uint8)
# yellow_upper = np.array([35,255,255], np.uint8)

# yellow_lower = np.array([15, 150, 150], np.uint8)
# yellow_upper = np.array([35, 255, 255], np.uint8)

#turn on camera
video = cv2.VideoCapture(0)

camera_on = True
while camera_on:

    #colors not detected
    red_detected = False
    yellow_detected = False

    #capture camera frame
    ret, frame = video.read()

    #convert rgb to hsv
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv", hsv_frame)

    #create mask to identify the desired colors
    
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    #cv2.imshow("mask", red_mask)
    kernal = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(frame, frame, mask=red_mask)
    #cv2.imshow("mask2", red_mask)
    #cv2.imshow("result", res_red)


    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    kernal = np.ones((5, 5), "uint8")
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(frame, frame, mask=yellow_mask)

    #identify region of interests 
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_detected = True

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))


    contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_detected = True

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(frame, "Yellow Detected", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255))

    #show video
    cv2.imshow("Color Detection", frame)

    #stop program
    if cv2.waitKey(10) & 0xFF == ord(' '):
        camera_on = False
