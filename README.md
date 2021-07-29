# <center> Vehicle-and-Human-Detection

## General Approach

Uses a cv2 while ret is True loop to process entire video
- call object detection function (creates bounding boxes)
- call counter function (check if bounding box intersects ROI)
- call drawing functions (ROI line, bounding boxes, counter text)
- return output video

### Command Line Interface

In the main directory, run main.py then answer all prompts.
```python3 main.py```
You can rename the source video in main.py to your own input.

### Input Video Requirements

Input must be 30 FPS for accurate results. Lower resolution videos process signficantly faster.

## **License Information**

Licensed under GNU General Public License v3.0
  
Please credit Jose Martinez and Wren Priest if you found this repository helpful!
