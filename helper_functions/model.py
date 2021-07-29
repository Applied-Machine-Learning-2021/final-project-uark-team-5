def load_model():
    import urllib.request
    import os

    base_url = 'http://download.tensorflow.org/models/object_detection/'
    file_name = 'ssd_mobilenet_v1_coco_2018_01_28.tar.gz'

    url = base_url + file_name

    urllib.request.urlretrieve(url, file_name)

    import tarfile
    import shutil

    dir_name = file_name[0:-len('.tar.gz')]

    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    tarfile.open(file_name, 'r:gz').extractall('./')

    os.listdir(dir_name)

    import tensorflow as tf

    frozen_graph = os.path.join(dir_name, 'frozen_inference_graph.pb')

    with tf.io.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    outputs = (
        'num_detections:0',
        'detection_classes:0',
        'detection_scores:0',
        'detection_boxes:0',
    )

    def wrap_graph(graph_def, inputs, outputs, print_graph=False):
        wrapped = tf.compat.v1.wrap_function(
            lambda: tf.compat.v1.import_graph_def(graph_def, name=""), [])

        return wrapped.prune(
            tf.nest.map_structure(wrapped.graph.as_graph_element, inputs),
            tf.nest.map_structure(wrapped.graph.as_graph_element, outputs))

    model = wrap_graph(graph_def=graph_def,
                       inputs=["image_tensor:0"],
                       outputs=outputs)

    # Object detection dictionary
    labels = {
        0: "background",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "trafficlight",
        11: "firehydrant",
        12: "unknown",
        13: "stopsign",
        14: "parkingmeter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        26: "unknown",
        27: "backpack",
        28: "umbrella",
        29: "unknown",
        30: "unknown",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sportsball",
        38: "kite",
        39: "baseballbat",
        40: "baseballglove",
        41: "skateboard",
        42: "surfboard",
        43: "tennisracket",
        44: "bottle",
        45: "unknown",
        46: "wineglass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hotdog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "pottedplant",
        65: "bed",
        66: "unknown",
        67: "diningtable",
        68: "unknown",
        69: "unknown",
        70: "toilet",
        71: "unknown",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cellphone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        83: "unknown",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddybear",
        89: "hairdrier",
        90: "toothbrush"
    }

    return model, labels