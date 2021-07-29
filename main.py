# Imports
import cv2
from helper_functions.model import load_model
from helper_functions.object_detection import object_detection
from helper_functions.counting import counting
from helper_functions import drawing

# Getting key global detection variables
print("Specify if the program should use default values for the detection variables (y,n): ")
choice = input()
if choice == 'n':
    ROI_ORIENTATION = input("Enter desired ROI orientation (horizontal or vertical): \n")
    while ROI_ORIENTATION != 'vertical' and ROI_ORIENTATION != 'horizontal':
        ROI_ORIENTATION = input("Please enter a valid ROI orientation(horizontal or vertical): \n")
    while True:
        try:
            vehicle_sensitivity = float(input("Enter custom vehicle sensitivity value: \n"))
        except ValueError:
            print("Vehicle sensitivity value must be a float between 0 and 1! \n")
            vehicle_sensitivity = 2
        if 0 <= vehicle_sensitivity <= 1:
            break
    while True:
        try:
            pedestrian_sensitivity = float(input("Enter custom pedestrian sensitivity value: \n"))
        except ValueError:
            print("Pedestrian sensitivity value must be a float between 0 and 1! \n")
            pedestrian_sensitivity = 2
        if 0 <= pedestrian_sensitivity <= 1:
            break
else:
    ROI_ORIENTATION = 'vertical'
    vehicle_sensitivity = 0.02
    pedestrian_sensitivity = 0.004

model, labels = load_model()

# input video
source_video = 'test.mp4'
cap = cv2.VideoCapture(source_video)

# Variables
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ticker = 0
vehicle_counter = 0
pedestrian_counter = 0
is_vehicle_detected = False

if ROI_ORIENTATION == 'horizontal':
    parameter = height
else:
    parameter = width

# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(source_video.split(".")[0] + '_output.mp4', fourcc, fps, (width, height))
# output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))


while cap.isOpened():
    ret, current_frame = cap.read()
    if ret is True:
        ticker += 1
        # progress ticker
        print(f"Processing frame {ticker} of {num_frames}")
        # object detection helper function
        (num_detect, classes, boxes, box_centers) = \
            object_detection(current_frame, model, ROI_ORIENTATION, height, width)

        # counting helper function
        if ticker % 2 == 0:
            (vehicle_counter, pedestrian_counter, is_vehicle_detected) = \
                counting(box_centers, classes, parameter, vehicle_counter, pedestrian_counter,
                         vehicle_sensitivity, pedestrian_sensitivity)

        # drawing helper functions
        drawing.draw_roi(current_frame, width, height, is_vehicle_detected, ROI_ORIENTATION)
        drawing.draw_detection_boxes(current_frame, num_detect, boxes, labels, classes)
        drawing.draw_counter(current_frame, vehicle_counter, pedestrian_counter)

        output.write(current_frame)
        # uncomment for debugging or sensitivity testing
        # cv2.imshow('debugging', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap.release()
        output.release()
        break
