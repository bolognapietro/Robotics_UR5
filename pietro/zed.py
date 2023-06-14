#!/usr/bin/env python

# Standard library modules
import os
from os.path import join
import subprocess
import json
import inspect
import math
import xml.etree.ElementTree as ET

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
from std_msgs.msg import String

# Local imports
import zed_params
import image_utils
import geometric_utils

def send_objects(objects: dict, topic: str = "/objects_info", max_queue: int = 10) -> None:
    """
    Publish each object.

    Args:
        objects (dict): Detected objects.
        topic (str, optional): Where to publish the informations. Defaults to "/objects_info".

    Returns:
        None.
    """

    pub = ros.Publisher(topic, Pose, queue_size=max_queue)

    for obj in objects:

        position = obj["position"]

        msg = Pose()

        msg.position.x = position["x"]
        msg.position.y = position["y"]
        msg.position.z = position["z"]

        angles = tf.transformations.quaternion_from_euler(position["roll"], position["pitch"], position["yaw"])

        msg.orientation.x = angles[0]
        msg.orientation.y = angles[1]
        msg.orientation.z = angles[2]
        msg.orientation.w = angles[3]

        pub.publish(msg)

def get_base_link_position(base_link_folder: str = "ros_ws/src/locosim/robot_control/lab_exercises/lab_palopoli/") -> dict:
    """
    Retrieves the following information about the base link:
    - Position: (x,y,z)

    Args:
        base_link_folder (str): The relative path of the params.py file inside ros_ws folder

    Returns:
        dict: Base link position.
    """

    import sys
    sys.path.insert(0,join(os.path.expanduser("~"),base_link_folder))
    import params as locosim_params
    sys.path.pop(0)

    robot_params = locosim_params.robot_params["ur5"]

    return {
        "spawn_x": robot_params["spawn_x"],
        "spawn_y": robot_params["spawn_y"],
        "spawn_z": robot_params["spawn_z"]
    }

def get_zed_position(zed_path: str = "ros_ws/install/share/ur_description/sensors/zed2/real/zed2.launch") -> dict:
    """
    Retrieves the following information about the zed camera:
    - Position: (x,y,z)
    - Rotation (roll, pitch, yaw)

    Args:
        zed_path (str): The relative path of the zed file inside ros_ws folder

    Returns:
        dict: Zed position.
    """

    xml = ET.parse(join(os.path.expanduser("~"),zed_path))

    zed = {
        "cam_pos_x": None,
        "cam_pos_y": None,
        "cam_pos_z": None,
        "cam_roll": None,
        "cam_pitch": None,
        "cam_yaw": None
    }

    for item in xml.findall("./arg"):
        
        if "name" not in item.attrib:
            continue

        if "default" not in item.attrib:
            continue

        name = item.attrib["name"]
        default = item.attrib["default"]

        if name not in zed.keys():
            continue

        zed[name] = float(default)

        if all([zed[key] != None for key in zed.keys()]):
            break

    return zed

def convert_to_gazebo_world_frame(point: tuple, precision: int = 2) -> tuple:
    """
    Converts 2D coordinates into gazebo world frame coordinates

    Args:
        point (tuple): Point to be converted.
        precision (int, optional): Number of decimal digits for the returned coordinates. Defaults to 2.

    Returns:
        tuple: 3D point (x,y,z).
    """
    # get current function as object
    function = eval(inspect.stack()[0][3])

    # in order to reduce the loading time, the zed position and base link position are loaded once 
    # and then saved in .zed_position and .base_link_position variables
    try:
        # check if the zed position has already been loaded
        function.zed_position
    except:
        # load the zed position (only once)
        function.zed_position = get_zed_position()

    try:
        # check if the base link position has already been loaded
        function.base_link_position
    except:
        # load the zed position (only once)
        function.base_link_position = get_base_link_position()

    zed_position = function.zed_position
    base_link_position = function.base_link_position

    point_cloud2_msg = ros.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    # retrieve the center of the detected object
    zed_coordinates = list(point)
    zed_coordinates = [int(coordinate) for coordinate in zed_coordinates]

    # get the 3d point (x,y,z)
    points = point_cloud2.read_points(point_cloud2_msg, field_names=['x','y','z'], skip_nans=False, uvs=[zed_coordinates])
    
    for point in points:
        zed_point = point[:3]

    #Ry = np.matrix([[ 0., -0.49948, 0.86632],[-1., 0., 0.],[-0., -0.86632, -0.49948]])
    #pos_zed = np.array([zed_position["cam_pos_x"], zed_position["cam_pos_y"], zed_position["cam_pos_z"]])
    #pos_base_link = np.array([base_link_position["spawn_x"],base_link_position["spawn_y"],base_link_position["spawn_z"]])

    Ry = np.array([[ 0.     , -0.49948,  0.86632],[-1.     ,  0.     ,  0.     ],[-0.     , -0.86632, -0.49948]])
    pos_zed = np.array([-0.9 ,  0.24, -0.35])
    transl_offset = np.array([0.01, 0.00, 0.1])

    data_world = Ry.dot(zed_point) + pos_zed + transl_offset #+ pos_base_link
    data_world = np.array(data_world)
    
    #data_world = data_world.tolist()[0]
    #data_world = tuple(data_world)

    return (round(data_world[0],precision), round(data_world[1],precision), round(data_world[2],precision))

def detect_objects(img: np.ndarray, threshold: float = 0.8, render: bool = False, yolo_path: str = "/home/pietro/ros_ws/src/locosim/robot_control/lab_exercises/lab_palopoli/zed/yolov5", model_path: str = "/home/pietro/ros_ws/src/locosim/robot_control/lab_exercises/lab_palopoli/zed/best_v6.pt") -> tuple:
    """
    Detects objects inside a given image.

    Args:
        img (np.ndarray): Image to be processed.
        threshold (float, optional): Specifies the minimum score that a detected object must have in order to be considered "valid". Defaults to 1.
        render (bool, optional): Specifies if the function has to return the rendered image with the detected objects. Defaults to False.
        yolo_path (str, optional): Yolo folder path. Defaults to "./yolov5".
        model_path (str, optional): Yolo .pt file path. Defaults to "best_v6.pt".

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

    image = img.copy()

    # process input image
    result = function.model([image], size = 640)

    # keep valid prediction(s) (e.g. score >= threshold)
    thresholded_result = torch.Tensor([item.tolist() for item in result.xyxy[0] if item[-2] >= threshold])
    result.xyxy[0] = thresholded_result
    
    # create objects list
    objects = []

    for item in result.xyxy[0]:

        item = item.tolist()

        box = item[0:4]
        box = [int(b) for b in box]
        box = [(box[0],box[1]),(box[0],box[3]),(box[2],box[1]),(box[2],box[3])]

        center = (int((box[0][0] + box[2][0])/2), int((box[0][1] + box[1][1])/2))
        center = convert_to_gazebo_world_frame(point = center)

        objects.append({
            "label_name": result.names[item[5]],
            "label_index": item[5],
            "score": round(item[4]*100,2),
            "position": {
                "x": center[0],
                "y": center[1],
                "z": center[2],
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

    frame = img.copy()

    image = None

    for obj in objects:
        
        box = obj["box"]

        image = image_utils.extract_obj(img = frame, box = box)
        
        if image_utils.all_black(image):
            continue 
        
        # check
        left_points = image_utils.left_side(img = image)
        right_points = image_utils.right_side(img = image)

        # vertices
        v1 = left_points[1]
        v3 = right_points[1]

        if left_points == right_points:
            v2 = left_points[0]
        else:
            m1, q1 = geometric_utils.calculate_line(left_points[0], left_points[1])
            m2, q2 = geometric_utils.calculate_line(right_points[0], right_points[1])

            v2 = ((q2-q1) / (m1 - m2), m1*((q2-q1) / (m1 - m2)) + q1)
            v2 = [int(v2[0]),int(v2[1])]

        #v2[1] = left_points[0][1] #v2[1] if v2[1] < image.shape[0] else image.shape[0]

        # calculate the height of the object (in pixel)
        if v1 == v3:
            v4 = v1
        else:
            m, q = geometric_utils.calculate_line(v1,v3)
            v4 = (int(v2[0]),int(v2[0]*m + q))

        height = abs(v4[1] - v2[1])

        # convert points coordinates to frame coordinates
        min_x = min(box, key=lambda x: x[0])[0]
        max_x = max(box, key=lambda x: x[0])[0]

        min_y = min(box, key=lambda x: x[1])[1]
        max_y = max(box, key=lambda x: x[1])[1]

        v1[0], v1[1] = v1[0] + min_x, v1[1] + min_y
        v2[0], v2[1] = v2[0] + min_x, v2[1] + min_y
        v3[0], v3[1] = v3[0] + min_x, v3[1] + min_y

        v1, v2, v3 = tuple(v1), tuple(v2), tuple(v3)
        v1_2D, v2_2D, v3_2D = v1, v2, v3

        # convert points
        v1 = convert_to_gazebo_world_frame(point = v1)
        v2 = convert_to_gazebo_world_frame(point = v2)
        v3 = convert_to_gazebo_world_frame(point = v3)

        # calculate angle
        if v2[0] - v1[0] != 0:
            angle = math.atan((v1[1]-v2[1]) / (v1[0]-v2[0]))
        else:
            angle = 0
        
        d12 = math.dist(v1,v2)
        d23 = math.dist(v2,v3)

        if(d12 > d23):
            angle = angle #+ math.pi/2
        
        if(d12 < d23):
            angle = angle - math.pi/2

        if math.degrees(angle) > 90:
            angle = angle % (math.pi / 2)


        angle = angle + math.radians(10)

        objects[objects.index(obj)]["position"]["yaw"] = angle

        # draw object main axes and display angle
        cv2.line(frame, v1_2D, v2_2D, (0,255,0))
        cv2.line(frame, v3_2D, v2_2D, (0,0,255))
        cv2.line(frame, (v2_2D[0],v2_2D[1] - height), (v2_2D[0],v2_2D[1]), (255,0,0))

        cv2.line(frame, v1_2D, v1_2D, (0,0,255),3)
        cv2.line(frame, v2_2D, v2_2D, (0,0,255),3)
        cv2.line(frame, v3_2D, v3_2D, (0,0,255),3)
        cv2.line(frame, (v2_2D[0],v2_2D[1] - height), (v2_2D[0],v2_2D[1] - height), (0,0,255),3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(round(angle,5))
        textSize = cv2.getTextSize(text,font, 1, 2)[0]
        textPos = (int((v1_2D[0] + v3_2D[0]) / 2 - textSize[0] / 2),v2_2D[1] + 40)

        top_left = (int(textPos[0] - 1),int(textPos[1] - textSize[1] - 2))
        bottom_right = (int(textPos[0] + textSize[0] + 1),int(textPos[1] + textSize[1]/2 - 8))

        frame = cv2.rectangle(frame, top_left, bottom_right, (0,0,255), -1)
        frame = cv2.putText(frame, text, textPos, font, 1, (255,255,255), 2, cv2.LINE_AA)

        frame = frame[zed_params.WINDOW_RECT["min_y"] : zed_params.WINDOW_RECT["max_y"] + 1, zed_params.WINDOW_RECT["min_x"]: zed_params.WINDOW_RECT["max_x"] + 1]
        
        #cv2.imshow(f"Debug",frame)
        #cv2.waitKey(0)

    return objects, frame

def process_image(img: np.ndarray, render: bool = False) -> np.ndarray:
    """
    Process the image captured by the zed camera

    Args:
        img (np.ndarray): Captured image.
        render (bool, optional): Specifies if the image with the bounding box of the detected objects has to be rendered. Defaults to False.

    Returns:
        dict: Updated objects.
    """

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

    # print detected objects
    #debug.print_objects(objects = objects)

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
    Receives, elaborates zed images and display them.

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
    frame = frame[zed_params.WINDOW_RECT["min_y"] : zed_params.WINDOW_RECT["max_y"] + 1, zed_params.WINDOW_RECT["min_x"]: zed_params.WINDOW_RECT["max_x"] + 1]

    cv2.imshow("Live detection",frame)

    # q for exit
    if cv2.waitKey(1) == ord('q'):
        exit(0)

if __name__ == '__main__':

    ros.init_node("vision", anonymous=True)

    #ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = background_detection, queue_size=1)
    ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = live_detection, queue_size=1)

    loop_rate = ros.Rate(1.)

    while True:
        loop_rate.sleep()
        pass