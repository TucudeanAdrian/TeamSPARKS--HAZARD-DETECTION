import cv2
import numpy as np
def canny_edge_detector(image): 
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  
    #blur = cv2.GaussianBlur(gray_image, (3, 3), 0)  
    canny = cv2.Canny(gray_image, 100, 150) 
    return canny
def region_of_interest(image,width,height): 
    height = image.shape[0] 
    polygons = np.array([ 
        [(150, height), (300, int(height-100)), (width-300, int(height-100)),(width-150, height)]]) 
    mask = np.zeros_like(image) 
    cv2.fillPoly(mask, polygons, 255)  
    masked_image = cv2.bitwise_and(image, mask)  
    return masked_image
def average_slope_intercept(image, crop):
    lines = cv2.HoughLinesP(crop, 1, np.pi/180, 30, maxLineGap=200)
    left_fit = [] 
    right_fit = []
    if(lines):
        for line in lines: 
            x1, y1, x2, y2 = line.reshape(4)     
            parameters = np.polyfit((x1, x2), (y1, y2), 1)  
            slope = parameters[0] 
            intercept = parameters[1] 
            if slope < 0: 
                left_fit.append((slope, intercept)) 
            else: 
                right_fit.append((slope, intercept)) 
              
    left_fit_average = np.average(left_fit, axis = 0) 
    right_fit_average = np.average(right_fit, axis = 0) 
    left_line = create_coordinates(image, left_fit_average) 
    right_line = create_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line])
def display_lines(image, lines): 
    line_image = np.zeros_like(image) 
    if lines is not None: 
        for x1, y1, x2, y2 in lines: 
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) 
    return line_image 
