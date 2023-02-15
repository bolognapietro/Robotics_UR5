#!/usr/bin/env python

import inspect
import os
import subprocess

import numpy as np
import torch

import cv2
from cv_bridge import CvBridge

import rospy as ros
from copy import deepcopy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2

from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

import json

import hashlib


def send_objects(objects) -> None:
    # pub  = ros.Publisher('prova', String, queue_size=5)
    # msg = "Ciao"
    # pub.publish(msg)
    
    pub = ros.Publisher('/prova', Float32MultiArray, queue_size=5)

    if len(objects) > 0:  
        for i in range(len(objects)):
            msg = Float32MultiArray()
            data = objects[i]["robot_center"]
            oggetti = []
            for el in data:
                oggetti.append(el)
            
            msg.data = oggetti
        
            pub.publish(msg)

def detect_objects(img: np.ndarray, threshold: float = 0.8, render: bool = False, yolo_path: str = "./yolov5", model_path: str = "best_v6.pt") -> tuple:
    """
    Detects objects inside a given image.

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``threshold`` (optional, ``default = 0.8 -> 80%``) specify the minimum score that a detected object must have in order to be considered "valid"
    - ``render`` (optional, ``default = False``) specifies if the function has to return the image with the detected objects
    - ``yolo_path`` (optional, ``default = "./yolov5"``) yolo folder path
    - ``model_path`` (optional, ``default = "best_v6.pt") model pt file path

    Output parameters (``tuple``):
    - ``result_map`` (``dict``):
        - ``box`` the bounding box of the detected object
        - ``center_image`` the center of the bounding box of the detected object inside the image
        - ``center_robot`` to be filled by the zed
        - ``score`` the score of the detected object
        - ``label_index`` the predicted class in a numeric format (e.g. 0,1...)
        - ``label_name`` the predicted class in a human-readable format
    - ``frame`` (``None | np.ndarray``):
        - if ``render = True`` it contains the frame with the bounding boxes of the detected objects, otherwise it will be ``None``
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

    # process input image
    result = function.model([img], size = 640)

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

        point = convert_to_robot_world_frame(center)

        objects.append({
            "box": box,
            "score": round(item[4]*100,2),
            "label_index": item[5],
            "label_name": result.names[item[5]],
            "image_center": center,
            "robot_center": point,
        })

    # if specified, render the result image
    frame = None

    if render:
        result.render()
        frame = result.ims[0]

    return objects, frame

def extract_table(img: np.ndarray, resolution: tuple = (1920,1080), table: list = [[563,274],[452,602],[1026,601],[798,274]]) -> np.ndarray:
    """
    Extracts the table from a given image. In other words, it makes all the image black except for the table

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``resolution`` resolution of the zed camera
    - ``table`` (optional, ``default = [[563,274],[798,274],[452,602],[1026,601]]``) is a list of the 4 vertices (x,y) of the table

    Output parameter (``np.ndarray``):
    - ``img`` (``np.ndarray``):
        - the processed image
    """

    _table = deepcopy(table)

    for coordinate in _table:
        coordinate[0] = int(coordinate[0] * resolution[0] / 1280)
        coordinate[1] = int(coordinate[1] * resolution[1] / 720)

    # create the mask
    mask = np.array(_table, dtype = np.int32)

    # create a black background so, all the image except the table will be black
    background = np.zeros((img.shape[0], img.shape[1]), np.int8)

    # fill the mask with the black background
    cv2.fillPoly(background, [mask],255)

    mask_background = cv2.inRange(background, 1, 255)

    # apply the mask
    img = cv2.bitwise_and(img, img, mask = mask_background)

    return img

def convert_to_robot_world_frame(point: tuple) -> tuple:
    """
    Calculate 2D coordinates into robot world frame coordinates

    Input parameters:
    - ``point`` 2D point

    Output parameters:
    ``It returns the tuple of the 3D point``
    """

    point_cloud2_msg = ros.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    # retrieve the center of the detected object
    zed_coordinates = list(point)
    zed_coordinates = [int(coordinate) for coordinate in zed_coordinates]

    # convert the coordinates of the center into the reference system of the zed
    #zed_coordinates[0] = int(zed_coordinates[0] - width/2)
    #zed_coordinates[1] = int(zed_coordinates[1] - height/2) 
    #zed_coordinates = tuple(zed_coordinates)

    # get the 3d point (x,y,z)
    points = point_cloud2.read_points(point_cloud2_msg, field_names=['x','y','z'], skip_nans=False, uvs=[zed_coordinates])
    
    for point in points:
        zed_point = point[:3]

    # transform the 3d point coordinates into coordinates of the real world frame
    w_R_c = np.array([[ 0.     , -0.49948,  0.86632],[-1.     ,  0.     ,  0.     ],[-0.     , -0.86632, -0.49948]])
    x_c = np.array([-0.9 ,  0.24, -0.35])
    transl_offset = np.array([0.01, 0.00, 0.1])
    zed_point = w_R_c.dot(zed_point) + x_c + transl_offset

    return (round(zed_point[0],2), round(zed_point[1],2), round(zed_point[2],2))

def receive_image(msg: Image) -> None:
    """
    Receives and elaborates zed images in background (no display)

    Example:
        >>> import rospy as ros
        >>> from sensor_msgs.msg import Image
        >>> if __name__ == "__main__":
        >>>     ros.init_node('mynode', anonymous=True)
        >>>     ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = receive_image, queue_size=1)
        >>>     loop_rate = ros.Rate(1.)
        >>>     while True:
        >>>         loop_rate.sleep()
    """

    # convert received image (bgr8 format) to a cv2 image
    img = CvBridge().imgmsg_to_cv2(msg, "bgr8")

    # extract table from image
    img = extract_table(img = img)

    # detect objects
    objects, frame = detect_objects(img = img)

    # save detected objects
    send_objects(objects)

    # print detected objects
    #print_objects()

def live_detection(msg: Image) -> None:
    """
    Displays detected objects

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

    # get current function as object
    function = eval(inspect.stack()[0][3])

    # init function variables
    try:
        function.prev_hash
    except:
        function.prev_hash = None
    
    current_hash = str(hashlib.sha256(msg.data).hexdigest())

    # check if the previous frame is equal to the new one
    # in this case don't process the new frame
    if current_hash == function.prev_hash:
        return
    
    # save the hash of the new frame
    function.prev_hash = current_hash

    # convert received image (bgr8 format) to a cv2 image
    img = CvBridge().imgmsg_to_cv2(msg, "bgr8")

    # extract table from image
    img = extract_table(img = img)

    # detect objects
    objects, frame = detect_objects(img = img, render = True)

    # save detected objects
    send_objects(objects)
    
    # print detected objects
    #print_objects()

    # display processed image
    cv2.imshow("Live detection",frame)

    # q for exit
    if cv2.waitKey(1) == ord('q'):
        exit(0)

if __name__ == '__main__':

    ros.init_node("vision", anonymous=True)

    #ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = receive_image, queue_size=1)
    ros.Subscriber("/ur5/zed_node/left_raw/image_raw_color", Image, callback = live_detection, queue_size=1)

    loop_rate = ros.Rate(1.)

    while True:
        loop_rate.sleep()
        pass