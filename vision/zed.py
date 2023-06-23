#!/usr/bin/env python

# Standard library modules
import os
from os.path import join, isfile
import pickle
import json
import inspect
import math
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
import torch
import tf

import cv2
from cv_bridge import CvBridge
from collections import Counter

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

def detect_objects(img: np.ndarray, threshold: float = 0.4, render: bool = False, yolo_path: str = "yolov5", model_path: str = "best.pt") -> tuple:
    """
    Detects objects inside a given image.

    Args:
        img (np.ndarray): Image to be processed.
        threshold (float, optional): Specifies the minimum score that a detected object must have in order to be considered "valid". Defaults to 0.4 (40%).
        render (bool, optional): Specifies if the function has to return the rendered image with the detected objects. Defaults to False.
        yolo_path (str, optional): Yolo folder path. Defaults to "yolov5".
        model_path (str, optional): Yolo .pt file path. Defaults to "best_v2.40.pt".

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

    frame = img.copy()
    image = image_utils.extract_objects(img=frame)

    for obj in objects:
        
        # process object box
        box = obj["box"]

        adj_box = 5

        min_x = min(box, key=lambda x: x[0])[0] - adj_box
        max_x = max(box, key=lambda x: x[0])[0] + adj_box

        min_y = min(box, key=lambda x: x[1])[1] - adj_box
        max_y = max(box, key=lambda x: x[1])[1] + adj_box

        points_2D = []

        # extract colored pixels except for the black ones
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                
                if image[y,x].tolist() == [0,0,0]:
                    continue
                
                points_2D.append([x,y])
        
        if not len(points_2D):
            continue
        
        # convert the extracted pixels (2d points) in 3d points
        points_3D = convert_to_gazebo_world_frame(points=points_2D)
        
        # final points array
        points = [[points_2D[i],points_3D[i]] for i in range(len(points_2D))]

        # object characterization
        base_line = []
        left_line = []
        right_line = []

        img_center = int((min_x + max_x) / 2)

        # scan each row
        for y in range(min_y, max_y+1):

            # current row
            pts = [point for point in points if point[0][1] == y]

            if not len(pts):
                continue
            
            # discard current row if not all the points have the same x (necessary for base line)
            if all(pts[i][1][0] == pts[i+1][1][0] for i in range(len(pts)-1)):
                base_line.append(pts)
            
            # extract all the points in the current row that are on the left side of the image
            left_points = [point for point in pts if point[0][0] < img_center]

            # extract all the points in the current row that are on the right side of the image
            right_points = [point for point in pts if point[0][0] >= img_center]

            # extract the point with the highest x on the left side of the image
            if len(left_points):
                left_line.append(max(left_points, key=lambda x: x[1][0]))
            
            # extract the point with the larger x on the rigth side of the image
            if len(right_points):
                right_line.append(max(right_points, key=lambda x: x[1][0]))

        # extract the row with the smaller x
        base_line = min(base_line, key=lambda x: x[0][1][0])
        
        # calculate the left and right points
        base_left = min(base_line, key=lambda x: x[0][0])
        base_right = max(base_line, key=lambda x: x[0][0])

        # remove all the points that have an x greater than the x of base_left        
        left_line = [point for point in left_line if point[1][2] == base_left[1][2] and point[0][0] <= base_left[0][0]]
        # extract the point with the highest x
        left_line_best = max(left_line, key=lambda x: x[1][0])
        # check if there are other pixels with the same x
        left_line = [point for point in left_line if point[0][0] == left_line_best[0][0]]
        # get the pixel with the smaller distance from base_left
        left_point = min(left_line, key=lambda x: math.dist(x[0], base_left[0]))

        # remove all the points that have an x smaller than the x of base_right  
        right_line = [point for point in right_line if point[1][2] == base_left[1][2] and point[0][0] >= base_right[0][0]]
        # extract the point with the highest x
        right_line_best = max(right_line, key=lambda x: x[1][0])
        # check if there are other pixels with the same x
        right_line = [point for point in right_line if point[0][0] == right_line_best[0][0]]
        # get the pixel with the smaller distance from base_right
        right_point = min(right_line, key=lambda x: math.dist(x[0], base_left[0]))

        # calculate the center as the middle point between the left and right points
        center_m, center_q = geometric_utils.calculate_line(left_point[1][:2],right_point[1][:2])

        center_x = (right_point[1][0] - left_point[1][0])/2 + left_point[1][0]
        center_y = (right_point[1][1] + left_point[1][1]) / 2 if center_m == 1 and center_q == 0 else center_x * center_m + center_q 

        center = (round(center_x,3), round(center_y,3), round(base_left[1][2],3))

        # calculate the angle (z)

        if left_point[1][0] != base_left[1][0]:
            angle_rad = math.atan((left_point[1][1] - base_left[1][1]) / (left_point[1][0] - base_left[1][0]))
        else:
            angle_rad = 0

        angle_deg = np.rad2deg(angle_rad)

        angle_rad = round(angle_rad,4)
        angle_deg = round(angle_deg,4)

        # save object
        objects[objects.index(obj)]["position"] = {
            "x": center[0],
            "y": center[1],
            "z": center[2],
            "roll": 0,
            "pitch": 0,
            "yaw": angle_rad
        }

        # draw the results on the frame (optional)
        cv2.line(frame, base_left[0], base_right[0], (0,0,255),2)
        cv2.line(frame, left_point[0], left_point[0], (0,0,255),2)
        cv2.line(frame, right_point[0], right_point[0], (0,0,255),2)

        print(f"========== OBJECT {obj['label_name']} ==========")
        print(f"Center: {center}")
        print(f"Rad: {angle_rad}\nDeg: {angle_deg}\n")

    if len(objects):
        cv2.imshow(f"Debug",frame)
        #cv2.imshow(f"Decolored",image_utils.extract_objects(img=frame))

    return objects, frame

def process_image(img: np.ndarray, render: bool = False, threshold: float = 3.5) -> np.ndarray:
    """
    Process the image captured by the zed camera

    Args:
        img (np.ndarray): Captured image.
        render (bool, optional): Specifies if the image with the bounding box of the detected objects has to be rendered. Defaults to False.
        threshold (float, optional): If the difference between the previous frame and the current one is greater or equal than the threshold, the frame will be processed. Otherwise, the old frame will be returned. Defaults to 3.5
    Returns:
        dict: Updated objects.
    """

    image = img.copy()

    # extract table from image
    image = image_utils.extract_table(img = image)

    function = eval(inspect.stack()[0][3])

    process = True

    try:
        
        # calculate the difference % between the previous frame and the current one
        diff = cv2.absdiff(function.img, image)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_gray = cv2.convertScaleAbs(diff_gray, alpha=5, beta=0)
        intensity = cv2.mean(diff_gray)[0]

        diff = abs((function.intensity - intensity) / intensity * 100)
        diff = round(diff,2)
        process = diff >= threshold

        # the current intensity becomes the previous one
        function.intensity = intensity
    except:

        # if this is the first time the function is called, initialize the parameters
        function.intensity = 0
        function.image_detected_objects = None

    # the current frame becomes the previous one
    function.img = image.copy()

    # if the current frame is different from the previous or if there is no previous processed image
    if process or function.image_detected_objects is None:
        #print("Updating...")

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

    else:
        # if the two frames are similar, then return the last processed image
        image_detected_objects = function.image_detected_objects

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

    ros.init_node("vision", anonymous=True)

    #ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = background_detection, queue_size=1)
    ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = live_detection, queue_size=1)
    
    loop_rate = ros.Rate(1.)

    while True:
        loop_rate.sleep()
        pass
