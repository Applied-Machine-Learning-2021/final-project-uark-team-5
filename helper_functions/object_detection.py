import tensorflow as tf


def object_detection(current_frame, model, ROI_line, height, width):
    # convert to tensor for model
    frame_tensor = tf.convert_to_tensor([current_frame], dtype=tf.uint8)
    # run frame_tensor through model
    detections = model(frame_tensor)

    # extract information from model detections
    num_detect = int(detections[0].numpy()[0])
    classes = detections[1].numpy()[0, :num_detect]
    # scores = detections[2].numpy()[0, :num_detect]
    boxes = detections[3].numpy()[0, :num_detect]

    box_centers = []

    if ROI_line == 'vertical':
        for x in range(num_detect):
            box = boxes[x]
            y1, x1, y2, x2 = box
            x1 *= width
            x2 *= width
            middle = int((x1 + x2) / 2)
            box_centers.append(middle)
    else:
        for x in range(num_detect):
            box = boxes[x]
            y1, x1, y2, x2 = box
            y1 *= height
            y2 *= height
            middle = int((y1 + y2) / 2)
            box_centers.append(middle)

    return num_detect, classes, boxes, box_centers
