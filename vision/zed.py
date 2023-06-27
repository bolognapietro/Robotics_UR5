#!/usr/bin/env python

# Standard library modules
import os
import inspect
import math
from statistics import mean
import subprocess
from time import time

import numpy as np
import torch
import tf

import cv2
from cv_bridge import CvBridge

# ROS modules
import rospy as ros
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from motion_plan_pkg.msg import legoMessage
from motion_plan_pkg.msg import legoArray
from std_msgs.msg import Bool

# Local imports
import zed_params
import image_utils
import geometric_utils

clear = lambda: subprocess.run("cls" if os.name == "nt" else "clear", shell=True)

def truncate(number: float, digits: int) -> float:
    """
    Truncate float number to n digits.

    Args:
        number (float): Number to be truncated.
        digits (int): Number of digits to keep.

    Returns:
        float: Truncated number.
    """

    index = str(number).find(".")

    if index == -1:
        return number
    
    number = f"{str(number)[:index]}{str(number)[index:digits+2]}"
    number = float(number)

    return number

def send_objects(objects: dict, topic: str = "/objects_info", max_queue: int = 10) -> None:
    """
    Publish each object.

    Args:
        objects (dict): Detected objects.
        topic (str, optional): Where to publish the informations. Defaults to "/objects_info".

    Returns:
        None.
    """
    if not len(objects):
        return

    pub = ros.Publisher(topic, legoArray, queue_size=max_queue)

    share_msg = legoArray()

    msgArray = []

    for obj in objects:

        position = obj["position"]

        pose = Pose()

        pose.position.x = position["x"]
        pose.position.y = position["y"]
        pose.position.z = position["z"]

        angles = tf.transformations.quaternion_from_euler(position["roll"], position["pitch"], position["yaw"])

        pose.orientation.x = angles[0]
        pose.orientation.y = angles[1]
        pose.orientation.z = angles[2]
        pose.orientation.w = angles[3]

        msg = legoMessage()
        msg.pose = pose
        msg.model = str(int(obj["label_index"]))

        msgArray.append(msg)

    share_msg.lego_array = msgArray
    ros.sleep(1)
    pub.publish(share_msg)
    
    ros.sleep(1)
    pub.publish(share_msg)

def convert_to_gazebo_world_frame(points: list, precision: int = 5) -> list:
    """
    Converts 2D coordinates into gazebo world frame coordinates

    Args:
        points (list): Points to be converted.
        precision (int, optional): Number of decimal digits for the returned coordinates. Defaults to 10.

    Returns:
        list: 3D points (x,y,z).
    """

    point_cloud2_msg = ros.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    zed_points = []
    multiple_points = all(isinstance(item, list) or isinstance(item, tuple) for item in points)

    if multiple_points:

        for point in list(points):
            zed_points.append([int(coordinate) for coordinate in point])
    
    else:
        zed_points = list(points)
        zed_points = [int(coordinate) for coordinate in zed_points]

    # get the 3d point (x,y,z)
    points = point_cloud2.read_points(point_cloud2_msg, field_names=['x','y','z'], skip_nans=False, uvs=zed_points if multiple_points else [zed_points])

    zed_points = []

    for point in points:
        zed_points.append(point[:3])

    Ry = np.array([[ 0.     , -0.49948,  0.86632],[-1.     ,  0.     ,  0.     ],[-0.     , -0.86632, -0.49948]])
    pos_zed = np.array([-0.4 ,  0.59,  1.4 ])

    data_world = []

    for point in zed_points:
        point = Ry.dot(point) + pos_zed
        point = np.array(point)
        point = [round(point[0],precision), round(point[1],precision), round(point[2],precision)]

        data_world.append(point)

    return data_world if len(data_world) > 1 else data_world[0]

def detect_objects(img: np.ndarray, threshold: float = 0.7, render: bool = False, yolo_path: str = "yolov5", model_path: str = "best.pt") -> tuple:
    """
    Detects objects inside a given image.

    Args:
        img (np.ndarray): Image to be processed.
        threshold (float, optional): Specifies the minimum score that a detected object must have in order to be considered "valid". Defaults to 0.7 (70%).
        render (bool, optional): Specifies if the function has to return the rendered image with the detected objects. Defaults to False.
        yolo_path (str, optional): Yolo folder path. Defaults to "yolov5".
        model_path (str, optional): Yolo .pt file path. Defaults to "best.pt".

    Returns:
        dict: Detected objects.
        np.ndarray | None: Rendered image if render param is set on True.
    """
    # check if the threshold is valid
    assert threshold > 0 and threshold <= 1, f"{threshold} is not a valid threshold."

    # get current function as object
    function = eval(inspect.stack()[0][3])

    # in order to reduce the prediction time, the model is loaded once and then saved in .model variable
    try:
        # check if the model has already been loaded
        function.model
    except:
        # load the model (only once)
        function.model = torch.hub.load(yolo_path, 'custom', path=model_path, source='local')

        # apply the threshold
        function.model.conf = threshold
    
    image = img.copy()

    if np.all(image == 0):
        return [], None

    #image = image_utils.extract_objects(image)

    # process input image
    result = function.model([image], size = 640)
    
    # create objects list
    objects = []

    for item in result.xyxy[0]:

        item = item.tolist()
        
        box = item[0:4]
        box = [int(b) for b in box]
        box = [(box[0],box[1]),(box[0],box[3]),(box[2],box[1]),(box[2],box[3])]

        objects.append({
            "label_name": result.names[item[5]],
            "label_index": item[5],
            "score": round(item[4]*100,2),
            "position": {
                "x": None,
                "y": None,
                "z": None,
                "roll": 0,
                "pitch": 0,
                "yaw": 0
            },
            "box": box
        })

    # if specified, render the result image
    frame = None

    if render:
        result.render()
        frame = result.ims[0]

    return objects, frame

def process_objects(img: np.ndarray, objects: dict) -> dict:
    """
    Process the detected objects

    Args:
        frame (np.ndarray): Captured frame. It has a "clean" frame without any previous manipulation.
        objects (dict): Detected objects.

    Returns:
        dict: Updated objects.
    """

    clear()

    image = img.copy()
    image = image_utils.extract_objects(img=image)

    processed_objects = []

    for obj in objects:
        
        start = time()

        # process object box
        box = obj["box"]

        adj_box = 5

        min_x = min(box, key=lambda x: x[0])[0] - adj_box
        max_x = max(box, key=lambda x: x[0])[0] + adj_box

        min_y = min(box, key=lambda x: x[1])[1] - adj_box
        max_y = max(box, key=lambda x: x[1])[1] + adj_box

        clean_image = image.copy()
        clean_box = image_utils.remove_noise_by_color(img=image[min_y : max_y+1, min_x : max_x+1])
        clean_image[min_y : max_y+1, min_x : max_x+1] = clean_box

        points_2D = []

        # extract colored pixels except for the black ones
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                
                if clean_image[y,x].tolist() == [0,0,0]:
                    continue
                
                points_2D.append([x,y])
        
        if not len(points_2D):
            continue

        # convert the extracted pixels (2d points) in 3d points
        points_3D = convert_to_gazebo_world_frame(points=points_2D)
        
        # final points array
        points = [[points_2D[i],points_3D[i]] for i in range(len(points_2D))]

        # get rows of the image
        rows = []

        for y in range(min_y, max_y+1):
            row = [point for point in points if point[0][1] == y]

            if not len(row):
                continue

            rows.append(row)
        
        # get columns of the image
        columns = []

        for x in range(min_x, max_x+1):
            column = [point for point in points if point[0][0] == x]

            if not len(column):
                continue

            columns.append(column)

        #? CENTER
        center_max_x = max(points, key=lambda x: x[1][0])
        center_min_x = min(points, key=lambda x: x[1][0])

        center_max_y = max(points, key=lambda x: x[1][1])
        center_min_y = min(points, key=lambda x: x[1][1])

        center_max_z = max(points, key=lambda x: x[1][2])
        center_min_z = min(points, key=lambda x: x[1][2])

        a = (center_min_x[1][0], center_min_y[1][1])
        b = (center_min_x[1][0], center_max_y[1][1])
        c = (center_max_x[1][0], center_max_y[1][1])
        d = (center_max_x[1][0], center_min_y[1][1])

        m_bd, q_bd = geometric_utils.calculate_line(b, d)
        m_ac, q_ac = geometric_utils.calculate_line(a,c)

        center_x = (q_bd - q_ac) / (m_ac - m_bd)
        center_y = m_ac * center_x + q_ac
        center_z = center_max_z[1][2]

        center = (center_x, center_y, center_z)
        center = min(points, key=lambda x: math.dist(x[1], center))

        center_2D, center_3D = center

        #? BOTTOM
        # the bottom will be the last row (approx), the one with the highest y

        base = [rows[-1][0], rows[-1][-1]]

        if rows[-2][-1][0][0] - rows[-2][0][0][0] > rows[-1][-1][0][0] - rows[-1][0][0][0]:
            base = [rows[-2][0], rows[-2][-1]]
        
        if rows[-3][-1][0][0] - rows[-3][0][0][0] > rows[-2][-1][0][0] - rows[-2][0][0][0]:
            base = [rows[-3][0], rows[-3][-1]]
        
        #? LEFT
        threshold = 3

        left = [sorted(column, key=lambda x: x[0][1]) for column in columns if column[0][0][0] <= base[0][0][0]]
        left = [column[-1] for column in left]

        if not len(left):
            left = base[0]
        
        else:

            tmp = []

            error = 0
            factor = 0.01

            while not len(tmp):

                tmp = [point for point in left if point[1][2] <= base[0][1][2] + error]

                if not len(tmp):
                    error = error + factor

            left_diff = [math.dist(tmp[i+1][0],tmp[i][0]) for i in range(len(tmp)-1)]

            if len(left_diff):
                left_diff_mean = math.ceil(mean(left_diff))

                left = []

                for i in range(len(tmp) - 1):

                    if math.dist(tmp[i+1][0],tmp[i][0]) > left_diff_mean:

                        if len(left):
                            break
                        else:
                            continue
                    
                    left.append(tmp[i])

                left = max(left, key=lambda x: x[1][0])

                if abs(left[0][0] - base[0][0][0]) <= threshold:
                    left = base[0]
            
            else:
                left = base[0]

        #? RIGHT
        right = [sorted(column, key=lambda x: x[0][1]) for column in columns if column[0][0][0] >= base[-1][0][0]]
        right = [column[-1] for column in right]

        if not len(right):
            right = base[-1]
        
        else:

            tmp = []

            error = 0
            factor = 0.01

            while not len(tmp):

                tmp = [point for point in right if point[1][2] <= base[-1][1][2] + error]

                if not len(tmp):
                    error = error + factor
            
            right_diff = [math.dist(tmp[i+1][0],tmp[i][0]) for i in range(len(tmp)-1)]

            if len(right_diff):
                right_diff_mean = math.ceil(mean(right_diff))

                right = []

                for i in range(len(tmp) - 1):

                    if math.dist(tmp[i+1][0],tmp[i][0]) > right_diff_mean:
                        
                        if len(right):
                            break
                        else:
                            continue
                    
                    right.append(tmp[i])

                right = max(right, key=lambda x: x[1][0])

                if abs(right[0][0] - base[-1][0][0]) <= threshold:
                    right = base[-1]
            
            else:
                right = base[-1]
        
        threshold = 3

        if abs(left[0][1] - base[0][0][1]) <= threshold:
            base[0] = left
        
        if abs(right[0][1] - base[-1][0][1]) <= threshold:
            base[-1] = right
            
        #? COLORS
        thickness = 2

        green = (0,255,0)
        blue = (255, 0, 0)
        red = (0, 0, 255)

        cv2.line(image, left[0], base[0][0], blue, thickness)
        cv2.line(image, base[0][0], base[-1][0], green, thickness)
        cv2.line(image, right[0], base[-1][0], red, thickness)

        #? YAW
        if left[1][0] != base[0][1][0]:
            angle_rad = math.atan((left[1][1] - base[0][1][1]) / (left[1][0] - base[0][1][0]))
        elif right[1][0] == base[-1][1][0]:
            angle_rad = 0
        else:
            angle_rad = math.pi / 2 - abs(math.atan((right[1][1] - base[-1][1][1]) / (right[1][0] - base[-1][1][0])))

        if math.dist(left[1], base[0][1]) > math.dist(right[1], base[-1][1]):
            angle_rad = angle_rad + math.pi / 2

        angle_deg = np.rad2deg(angle_rad)

        angle_rad = round(angle_rad,4)
        angle_deg = round(angle_deg,4)

        # save object
        processed_objects.append(obj)
        processed_objects[processed_objects.index(obj)]["position"] = {
            "x": center_3D[0],
            "y": center_3D[1],
            "z": center_3D[2],
            "roll": 0,
            "pitch": 0,
            "yaw": angle_rad
        }

        kpi_1_1 = round(time() - start,2)

        print(f"Model: {obj['label_name']} \nPosition: {(center_3D[0], center_3D[1], center_3D[2])} \nOrientation (rad): (None, None, {angle_rad})\nOrientation (deg): (None, None, {angle_deg})\nKPI 1-1: {kpi_1_1} second(s)\n")

    if len(objects):
        cv2.imshow(f"Debug",image)

    return processed_objects, img

def process_image(img: np.ndarray, render: bool = False) -> np.ndarray:
    """
    Process the image captured by the zed camera

    Args:
        img (np.ndarray): Captured image.
        render (bool, optional): Specifies if the image with the bounding box of the detected objects has to be rendered. Defaults to False.
    Returns:
        dict: Updated objects.
    """

    function = eval(inspect.stack()[0][3])
    
    try:
        function.cold_start
        function.image_detected_objects
    except:
        function.cold_start = True
        function.image_detected_objects = img.copy()

    if not function.cold_start:

        try:
            ros.wait_for_message("/start_zed", Bool, timeout=1)
        except:
            return function.image_detected_objects

    else:
        function.cold_start = False

    image = img.copy()

    # extract table from image
    image = image_utils.extract_table(img = image)

    # half table
    for y in range(int(image.shape[0]*zed_params.TABLE_PERCENTAGE)):
        image[y] = 0

    # detect objects
    objects, image_detected_objects = detect_objects(img = image, render = render)
    
    # process objects
    objects, image_processed_objects = process_objects(img = image, objects = objects)

    # send detected objects
    send_objects(objects = objects)

    function.image_detected_objects = image_detected_objects

    return image_detected_objects

def background_detection(msg: Image) -> None:
    """
    Receives and elaborates zed images in background (no display)

    Args:
        msg (Image): Image received from the zed camera.

    Returns:
        None.

    Example:
        >>> import rospy as ros
        >>> from sensor_msgs.msg import Image
        >>> if __name__ == "__main__":
        >>>     ros.init_node('mynode', anonymous=True)
        >>>     ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = background_detection, queue_size=1)
        >>>     loop_rate = ros.Rate(1.)
        >>>     while True:
        >>>         loop_rate.sleep()
    """

    # convert received image (bgr8 format) to a cv2 image
    img = CvBridge().imgmsg_to_cv2(msg, "bgr8")

    # process image
    process_image(img = img)

def live_detection(msg: Image) -> None:
    """
    Receives, elaborates zed images and display them (useful for debugging).

    Args:
        msg (Image): Image received from the zed camera.

    Returns:
        None.

    Example:
        >>> import rospy as ros
        >>> from sensor_msgs.msg import Image
        >>> if __name__ == "__main__":
        >>>     ros.init_node('mynode', anonymous=True)
        >>>     ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = live_detection, queue_size=1)
        >>>     loop_rate = ros.Rate(1.)
        >>>     while True:
        >>>         loop_rate.sleep()
    """

    # convert received image (bgr8 format) to a cv2 image
    img = CvBridge().imgmsg_to_cv2(msg, "bgr8")

    # process image
    frame = process_image(img = img, render = True)
    
    # display processed image
    cv2.imshow("Live detection",frame)

    # q for exit
    if cv2.waitKey(1) == ord('q'):
        exit(0)

if __name__ == '__main__':

    # dummy invocation used to load yolo
    detect_objects(img = np.zeros((100, 100), dtype=np.uint8), render = False)

    ros.init_node("vision", anonymous=True)

    #ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = background_detection, queue_size=1)
    ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = live_detection, queue_size=1)
    
    loop_rate = ros.Rate(1.)

    while True:
        loop_rate.sleep()
        pass
