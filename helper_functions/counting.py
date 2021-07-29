def counting(box_centers, classes, parameter, vehicle_counter, pedestrian_counter,
             vehicle_sensitivity, pedestrian_sensitivity):
    is_vehicle_detected = False
    # print(classes)
    for i in range(len(box_centers)):
        # print(classes[i])
        if classes[i] in [3, 6, 8]:
            if box_centers[i] in range(int(parameter / 2 - vehicle_sensitivity * parameter),
                                       int(parameter / 2 + vehicle_sensitivity * parameter)):
                vehicle_counter += 1
                is_vehicle_detected = True
            else:
                is_vehicle_detected = False
        elif classes[i] == 1:
            if box_centers[i] in range(int(parameter / 2 - pedestrian_sensitivity * parameter),
                                       int(parameter / 2 + pedestrian_sensitivity * parameter)):
                pedestrian_counter += 1
                is_vehicle_detected = True
            else:
                is_vehicle_detected = False

    return vehicle_counter, pedestrian_counter, is_vehicle_detected
