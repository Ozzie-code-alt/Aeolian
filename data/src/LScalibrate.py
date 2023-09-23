import cv2
import numpy as np
import LSsharedmodules
import math
import tkinter as tk
from tkinter import simpledialog


def selectPoints():  # function for user to define corners of screen for warping

    global points  # initialise points var - stores co-ords of points for use during warping
    points = []

    cap = cv2.VideoCapture(1)  # init webcam capture
    LSsharedmodules.popUp("Select points","To calibrate, please select the corners of your screen \n\nPress 'ENTER' to save config or 'R' to reset points", 1)

    check, frame = cap.read()
    cv2.imshow("Calibration", frame)
    set_top = True
    while True:  # cv2 loop
        check, frame = cap.read()
        if not check:  # checks frames are being recieved
            break
        
        cv2.imshow("Calibration", frame)  # displays current frame

        if set_top:  # opens the window in front of all windows
            cv2.setWindowProperty("Calibration",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty("Calibration",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
            set_top = False

        cv2.setMouseCallback("Calibration", click)  # mouse event callback func

        displayPoints(frame)  # displays selected points on the screen
        
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # exits calibration if esc pressed
            break
        elif key == ord('r') and len(points) != 0:  # clears all points and resets frame
            points.clear()
            displayPoints(frame)
            cv2.imshow("Calibration", frame)
        elif key == 13 and len(points) == 4:  # moves onto warping if 4 points chosen and ENTER is pressed
            cv2.destroyWindow('Calibration')
            matrix = warpImage(cap, points)  # calls functions and returns matrix and mask parameters
            maskparams = maskImage(cap, matrix)
            if maskparams == False:
                return (False, False)
            confirm = LSsharedmodules.popUp("Save Profile", "Do you want to save this profile?", 2)
            cv2.destroyAllWindows()
            return (points, maskparams) if confirm else (False, False)
        #hover function


    cap.release()
    cv2.destroyWindow('Calibration')
    return (False, False)



def click(event, x, y, flags, params,):  # event function to detect user calibration clicks
        global point
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:  # appends point to points var. only allows 4 points
            point = [x,y]
            print(point)
            points.append(point)  # adds point to list of points


def displayPoints(frame):
    for i in range(len(points)):  # draws all points on frame
        cv2.circle(frame, points[i], 5, (0, 0, 255), -1)

    if len(points) == 4:
        for point in points:
            distances = [math.sqrt((point[0]-other_p[0])**2+(point[1]-other_p[1])**2) for other_p in points]  # formula for distance between 2 points
            sorted_d = sorted(distances, reverse=True)
            sorted_d.pop()
            for i in range(2):  # connects each point to the nearest two points
                connected_d = sorted_d.pop()
                connected_p = points[distances.index(connected_d)]
                cv2.line(frame, point, connected_p, (0, 255, 0), 1)

    cv2.imshow("Calibration", frame)


def warpImage(cap, points):
    left, right = sorted(points)[:2], sorted(points)[2:]  # calculates which points are for which corner of the screen
    tl, bl = sorted(left, key=lambda x: x[1])             # + allows point selection in any order
    tr, br = sorted(right, key=lambda x: x[1])

    pts1 = np.float32([tl, tr, bl, br])  # creates source matrix of points on orignial frame
    pts2 = np.float32([[0, 0], [1000, 0], [0, 1000], [1000, 1000]])  # creates destination matrix of points on warped frame
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # creates perspective transform matrix for warping below
    return matrix
    

def maskImage(cap, mat):
    saved = False
    selected = False
    
    def get_hsv_value(color_name):
        color_values = {
            "Red": ([0, 101, 101], [10, 255, 255]),
            "Blue": ([101, 142, 88], [108, 255, 255]),
            "Green": ([36, 50, 69], [90, 255, 255]),
            "Purple": ([129, 50, 69], [158, 255, 255]),
            "Orange": ([10, 50, 69], [24, 255, 255])
        }
        return color_values.get(color_name, ([0, 0, 0], [0, 0, 0]))
    
    while True:
        check, frame = cap.read()
        if not check:
            break

        frame = cv2.warpPerspective(frame, mat, (1000, 1000))
        hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # converts img to hsv for masking

        if not selected:
            auto = LSsharedmodules.popUp("Select a masking method", "Would you like to automatically generate a mask?\nPress 'No' to manually create a mask.\n\nIf unsure, automatic mask generation is recommended",2 )  # branches to auto or manual masking
            if not auto:

                color_name = simpledialog.askstring("Color Selection", "Select a color (Red, Blue, Green, Purple, Orange):")
                if color_name is None:
                    n = "Create Mask"  # Creates trackbar menu
                    cv2.namedWindow(n)
                    cv2.createTrackbar("Lower H", n, 0, 255, noFunc)
                    cv2.createTrackbar("Lower S", n, 0, 255, noFunc)
                    cv2.createTrackbar("Lower V", n, 0, 255, noFunc)

                    cv2.createTrackbar("Upper H", n, 255, 255, noFunc)
                    cv2.createTrackbar("Upper S", n, 255, 255, noFunc)
                    cv2.createTrackbar("Upper V", n, 255, 255, noFunc)
                    imageHEHE = cv2.imread('data\images\image.jpg')

                    # Define the text to be added
                    text = 'RED Value:  Lower Hue: 0 , Lower Saturation: 101 or 99 , Lower Value: 101 or 99'
                    text1_0 = 'Upper Hue: 10,  Upper Saturation: 255, Upper Value: 255 '

                    text2 = "Blue Value: Lower Hue: 101 , Lower Saturation: 142 , Lower Value: 88"
                    text2_0 = 'Upper Hue: 108,  Upper Saturation: 255, Upper Value: 255 '

                    text3 = "Green Value: Lower Hue: 36 , Lower Saturation: 50 , Lower Value: 69 or 71"
                    text3_0 = "Green Value: Upper Hue: 90 , Upper Saturation: 255 , Upper Value: 255"

                    text4 = "Purple Value: Lower Hue: 129 , Lower Saturation: 50 , Lower Value: 69 or 71"
                    text4_0 = "Purple Value: Upper Hue: 158 , Upper Saturation: 255 , Upper Value: 255"

                    text5 = "Orange Value: Lower Hue: 10 , Lower Saturation: 50 , Lower Value: 69 or 71"
                    text5_0 = "Orange Value: Upper Hue: 24 , Upper Saturation: 255 , Upper Value: 255"

                    # Define the font settings (font type, size, color, etc.)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5  # Font scale factor
                    font_color1 = (000, 000, 255)  # font color Red
                    font_color = (255, 51, 51)  # Font color Blue
                    font_color3 = (0, 128, 0)  # Font color green
                    font_color4 = (128,0,128) # font color purple
                    font_color5 =(0,165,255)


                    thickness = 2  # Thickness of the text

                    # Specify the position where you want to add the text (x, y)
                    position = (0,40) #Red
                    position1_0 = (0,80)
                    position2 = (0,150) #Blue
                    position2_0 = (0,190) 

                    position3 = (0,260)#Green
                    position3_0 = (0, 300)
                    
                    position4 = (0,370) #Purple
                    position4_0 = (0,410)

                    position5 = (0,480) #Orange
                    position5_0 = (0, 520)
                    # Use the putText function to add text to the imageHEHE
                    cv2.putText(imageHEHE, text, position, font, font_scale, font_color1, thickness)
                    cv2.putText(imageHEHE, text1_0, position1_0, font, font_scale, font_color1, thickness)

                    cv2.putText(imageHEHE, text2, position2, font, font_scale, font_color, thickness)
                    cv2.putText(imageHEHE, text2_0, position2_0, font, font_scale, font_color, thickness)
                    
                    cv2.putText(imageHEHE, text3, position3, font, font_scale, font_color3, thickness)
                    cv2.putText(imageHEHE, text3_0, position3_0, font, font_scale, font_color3, thickness)

                    cv2.putText(imageHEHE, text4, position4, font, font_scale, font_color4, thickness)
                    cv2.putText(imageHEHE, text4_0, position4_0, font, font_scale, font_color4, thickness)

                    cv2.putText(imageHEHE, text5, position5, font, font_scale, font_color5, thickness)
                    cv2.putText(imageHEHE, text5_0, position5_0, font, font_scale, font_color5, thickness)
                    # Save or display the imageHEHE with the added text
                    cv2.imwrite('output_image.jpg', imageHEHE)
                    # Display the imageHEHE (optional)
                    cv2.imshow('Sample BGR Values', imageHEHE)

   
            selected = True

        if not saved:  # exits once maskparams created and saved
            if auto:
                maskparams = automaticMaskParams(frame, hsvimg)
                saved = True
            elif not auto and color_name is not None:
                lower_hsv, upper_hsv = get_hsv_value(color_name)
                maskparams = [(lower_hsv[0], lower_hsv[1], lower_hsv[2]), (upper_hsv[0], upper_hsv[1], upper_hsv[2])]
                showMaskCreation(maskparams, frame, hsvimg, saved)  # displays image and trackbar menu
                saved = True
            else:
                maskparams = manualMaskParams(frame, hsvimg)
                if len(maskparams) == 3:  # checks returned values to see if values saved
                    maskparams = maskparams[:-1]
                    saved = True
                showMaskCreation(maskparams, frame, hsvimg, saved)  # displays image and trackbar menu
                
        if saved:
            return maskparams

        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) == 13:
            cv2.destroyWindow(n)
            saved = True
        

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyWindow("Sample BGR Values")
    return False


def noFunc(x):  # dummy function
    pass


def manualMaskParams(img, hsv):
    pos = []
    n = "Create Mask"
    tbn = {  # trackbar names
        1: "Lower H",
        2: "Lower S",
        3: "Lower V",
        4: "Upper H",
        5: "Upper S",
        6: "Upper V",
    }
    
    for i in range(1, 7):
        pos.append(cv2.getTrackbarPos(tbn[i], n))  # adds trackbar names to interface

    # creates lower and upper bound masking arrays from current positions of trackbar
    lower = np.array([pos[0], pos[1], pos[2]])
    upper = np.array([pos[3], pos[4], pos[5]])

    if cv2.waitKey(1) == 13:
        cv2.destroyWindow(n)
        return [lower, upper, "saved"]  # returns the params and check str once saved

    return [lower, upper]


def automaticMaskParams(img, hsv):  # Automatic Settings (ALPHA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use the cv2.threshold function to create a binary image based on the detected light
    _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the light source)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the centroid of the largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Convert the centroid pixel color to HSV
        target_color_hsv = cv2.cvtColor(np.uint8([[img[cy, cx]]]), cv2.COLOR_BGR2HSV)[0][0]

        # Define a range for detecting the color (you may need to adjust this range)
        color_range = np.array([20, 50, 50])  # Adjust these values as needed

        # Calculate the lower and upper bounds for the color detection
        lower = target_color_hsv - color_range
        upper = target_color_hsv + color_range

        # Ensure that the values are within the valid HSV range (0-180 for H, 0-255 for S and V)
        lower = np.clip(lower, [0, 0, 0], [180, 255, 255])
        upper = np.clip(upper, [0, 0, 0], [180, 255, 255])

        return [lower, upper]

    # If no contours are found, return a default value
    return [np.array([0, 0, 0]), np.array([180, 255, 255])]




def showMaskCreation(maskparams, frame, hsv, saved):
    mask = cv2.inRange(hsv, maskparams[0], maskparams[1])  # creates mask using hsv image, upper and lower bound
    img = cv2.bitwise_and(frame, frame, mask=mask)  # creates original image with mask using bitwise and
    cv2.imshow("Windows", img)
    if saved:
        cv2.destroyAllWindows()
