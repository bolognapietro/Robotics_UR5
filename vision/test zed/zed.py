#!/usr/bin/env python
import os
import subprocess
import json
import inspect
import math

import numpy as np
import torch

import cv2
from cv_bridge import CvBridge

import rospy as ros
from copy import deepcopy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from std_msgs.msg import String

import logging
logging.getLogger("utils.general").setLevel(logging.WARNING)

import image_utils
import geometric_utils

TABLE_DEFAULT = [[563,274],[452,602],[1026,601],[798,274]]
TABLE_PERCENTAGE = 0.5

WINDOW_RECT = None

def send_objects(objects: dict, name: str = "detected_objects") -> None:
    """
    Sends detected objects. 
    Once the message is received, you can convert it to a json object by using json.loads(...)

    Input parameters:
    - ``objects`` dict of the detected objects
    - ``name`` (optional, ``default = detected_objects``) specify where the message can be retrieved
    """

    pub = ros.Publisher(name, String, queue_size=1)

    msg = String()
    msg.data = json.dumps(objects)

    pub.publish(msg)

def print_objects(objects: dict) -> None:
    """
    Prints detected objects. 
    This feature is intended for debugging purposes and will only work if the script is executed as the main program.

    Input parameters:
    - ``objects`` dict of the detected objects
    """

    if __name__ != '__main__':
        return
    
    subprocess.run("cls" if os.name == "nt" else "clear", shell=True)

    for obj in objects:
        print(f"{obj['label_name']} {obj['score']}%")

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
        - A dict of the detected objects with their characteristics
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

        robot_world_frame_center = convert_to_robot_world_frame(center)
        gazebo_world_frame_center = convert_to_gazebo_world_frame(center)

        objects.append({
            "label_name": result.names[item[5]],
            "label_index": item[5],
            "score": round(item[4]*100,2),
            "box": box,
            "image_center": center,
            "robot_world_frame_center": robot_world_frame_center,
            "gazebo_world_frame_center": gazebo_world_frame_center,
            "box_robot_world_frame": box_robot_world_frame(box),
            "box_gazebo_world_frame": box_gazebo_world_frame(box)
        })

    # if specified, render the result image
    frame = None

    if render:
        result.render()
        frame = result.ims[0]

    return objects, frame

def extract_table(img: np.ndarray, resolution: tuple = (1920,1080), table: list = TABLE_DEFAULT) -> np.ndarray:
    """
    Extracts the table from a given image. In other words, it makes all the image black except for the table

    Input parameters:
    - ``img`` the array of the image to be processed
    - ``resolution`` (optional, ``default = 1920x1080``) resolution of the zed camera
    - ``table`` (optional, ``default = [[563,274],[798,274],[452,602],[1026,601]]``) is a list of the 4 vertices (x,y) of the table

    Output parameter (``np.ndarray``):
    - ``img`` (``np.ndarray``):
        - the processed image
    """

    global TABLE_DEFAULT, WINDOW_RECT

    _table = deepcopy(table)

    if table == TABLE_DEFAULT:

        for coordinate in _table:
            coordinate[0] = int(coordinate[0] * resolution[0] / 1280)
            coordinate[1] = int(coordinate[1] * resolution[1] / 720)

    offset = 25

    min_x = min([coordinate[0] for coordinate in _table]) - offset
    min_x = min_x if min_x >= 0 else 0

    max_x = max([coordinate[0] for coordinate in _table]) + offset
    max_x = max_x if max_x < img.shape[1] else img.shape[1] - 1

    min_y = min([coordinate[1] for coordinate in _table]) - offset
    min_y = min_y if min_y >= 0 else 0
    
    max_y = max([coordinate[1] for coordinate in _table]) + offset
    max_y = max_y if max_y < img.shape[0] else img.shape[0] - 1

    WINDOW_RECT = {
        "max_x": max_x,
        "min_x": min_x,
        "max_y": max_y,
        "min_y": min_y
    }

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

def convert_to_robot_world_frame(point: tuple, precision: int = 2) -> tuple:
    """
    Converts 2D coordinates into robot world frame coordinates

    Input parameters:
    - ``point`` 2D point
    - ``precision`` (optional, ``default = 2``) number of decimal digits for the returned coordinates

    Output parameters:
    ``It returns the tuple of the 3D point``
    """

    point_cloud2_msg = ros.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    # retrieve the center of the detected object
    zed_coordinates = list(point)
    zed_coordinates = [int(coordinate) for coordinate in zed_coordinates]

    # get the 3d point (x,y,z)
    points = point_cloud2.read_points(point_cloud2_msg, field_names=['x','y','z'], skip_nans=False, uvs=[zed_coordinates])
    
    for point in points:
        zed_point = point[:3]

    # transform the 3d point coordinates
    w_R_c = np.array([[ 0.     , -0.49948,  0.86632],[-1.     ,  0.     ,  0.     ],[-0.     , -0.86632, -0.49948]])
    x_c = np.array([-0.9 ,  0.24, -0.35])
    transl_offset = np.array([0.01, 0.00, 0.1])
    zed_point = w_R_c.dot(zed_point) + x_c + transl_offset

    return (round(zed_point[0],precision), round(zed_point[1],precision), round(zed_point[2],precision))

def box_robot_world_frame(box: list, precision: int = 2) -> dict:
    """
    Converts 2D coordinates of the object bounding box to robot world frame coordinates 

    Input parameters:
    - ``box`` list of list [[...],[...],] which represents the bounding box of the detected object
    - ``precision`` (optional, ``default = 2``) number of decimal digits for the returned coordinates

    Output parameters:
    ``It returns a dict where each 2D coordinate has been mapped with its 3D coordinate``
    """

    point_cloud2_msg = ros.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    img_min_x = min(box, key=lambda x:x[0])[0]
    img_max_x = max(box, key=lambda x:x[0])[0]

    img_min_y = min(box, key=lambda x:x[1])[1]
    img_max_y = max(box, key=lambda x:x[1])[1]

    zed_points = []

    for y in range(img_min_y, img_max_y+1):
        for x in range(img_min_x, img_max_x+1):

            points = point_cloud2.read_points(point_cloud2_msg, field_names=['x','y','z'], skip_nans=False, uvs=[[x,y]])
    
            for point in points:
                zed_point = point[:3]

            #point = list(zed_point)
            #point = [p if not math.isnan(p) else -100 for p in point]

            w_R_c = np.array([[ 0.     , -0.49948,  0.86632],[-1.     ,  0.     ,  0.     ],[-0.     , -0.86632, -0.49948]])
            x_c = np.array([-0.9 ,  0.24, -0.35])
            transl_offset = np.array([0.01, 0.00, 0.1])
            zed_point = w_R_c.dot(zed_point) + x_c + transl_offset

            zed_points.append({
                "2D": (x - img_min_x, y - img_min_y),
                "3D": (round(zed_point[0],precision), round(zed_point[1],precision), round(zed_point[2],precision))
            })

    return zed_points

def convert_to_gazebo_world_frame(point: tuple, precision: int = 2) -> tuple:
    """
    Converts 2D coordinates into gazebo world frame coordinates

    Input parameters:
    - ``point`` 2D point
    - ``precision`` (optional, ``default = 2``) number of decimal digits for the returned coordinates

    Output parameters:
    ``It returns the tuple of the 3D point``
    """

    point_cloud2_msg = ros.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)

    # retrieve the center of the detected object
    zed_coordinates = list(point)
    zed_coordinates = [int(coordinate) for coordinate in zed_coordinates]

    # get the 3d point (x,y,z)
    points = point_cloud2.read_points(point_cloud2_msg, field_names=['x','y','z'], skip_nans=False, uvs=[zed_coordinates])
    
    for point in points:
        zed_point = point[:3]

    Ry = np.matrix([[ 0.     , -0.49948,  0.86632],[-1.     ,  0.     ,  0.     ],[-0.     , -0.86632, -0.49948]])
    pos_zed = np.array([-0.9 ,  0.24, -0.35])
    pos_base_link = np.array([0.5,0.35,1.75])
    
    data_world = Ry.dot(zed_point) + pos_zed + pos_base_link
    data_world = np.array(data_world)
    
    data_world = data_world.tolist()[0]
    #data_world[0] = data_world[0] - 0.5
    #data_world[1] = data_world[1] - 0.35
    #data_world[2] = data_world[2] - 1.64

    data_world = tuple(data_world)

    return (round(data_world[0],precision), round(data_world[1],precision), round(data_world[2],precision))

def box_gazebo_world_frame(box: list, precision: int = 2) -> dict:
    """
    Converts 2D coordinates of the object bounding box to gazebo world frame coordinates 

    Input parameters:
    - ``box`` list of list [[...],[...],] which represents the bounding box of the detected object
    - ``precision`` (optional, ``default = 2``) number of decimal digits for the returned coordinates

    Output parameters:
    ``It returns a dict where each 2D coordinate has been mapped with its 3D coordinate``
    """

    point_cloud2_msg = ros.wait_for_message("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2)
    
    img_min_x = min(box, key=lambda x:x[0])[0]
    img_max_x = max(box, key=lambda x:x[0])[0]

    img_min_y = min(box, key=lambda x:x[1])[1]
    img_max_y = max(box, key=lambda x:x[1])[1]

    zed_points = []

    for y in range(img_min_y, img_max_y+1):
        for x in range(img_min_x, img_max_x+1):

            points = point_cloud2.read_points(point_cloud2_msg, field_names=['x','y','z'], skip_nans=False, uvs=[[x,y]])
    
            for point in points:
                zed_point = point[:3]

            point = list(zed_point)
            point = [p if not math.isnan(p) else -100 for p in point]

            Ry = np.matrix([[ 0.     , -0.49948,  0.86632],[-1.     ,  0.     ,  0.     ],[-0.     , -0.86632, -0.49948]])
            pos_zed = np.array([-0.9 ,  0.24, -0.35])
            pos_base_link = np.array([0.5,0.35,1.75])
            
            data_world = Ry.dot(zed_point) + pos_zed + pos_base_link
            data_world = np.array(data_world)

            data_world = data_world.tolist()[0]
            #data_world[0] = data_world[0] - 0.5
            #data_world[1] = data_world[1] - 0.35
            #data_world[2] = data_world[2] - 1.64

            data_world = tuple(data_world)

            data_world = (round(data_world[0],precision), round(data_world[1],precision), round(data_world[2],precision))

            zed_points.append({
                "2D": (x - img_min_x, y - img_min_y),
                "3D": data_world
            })

    return zed_points

def process_objects(frame: np.ndarray, objects: dict) -> tuple:
    """
    Process the detected objects

    Input parameters:
    - ``frame`` the captured frame
    - ``objects`` dict of detected objects

    Output parameter (``tuple``):
    - ``img`` (``np.ndarray``):
        - the anized frame
    - ``objects`` (``dict``):
        - updated objects
    """

    for obj in objects:
        
        box = obj["box"]

        image = image_utils.extract_obj(img = frame, box = box)

        left_points = image_utils.left_side(img = image)
        right_points = image_utils.right_side(img = image)

        # buf fix
        if len(left_points) > 1:
            if type(left_points[0]) == np.ndarray:
                left_points = left_points[1]
        
        if len(right_points) > 1:
            if type(right_points[0]) == np.ndarray:
                right_points = right_points[1]

        # vertices
        v1, v2, v3 = left_points[1], [int((left_points[0][0] + right_points[0][0])/2),min(left_points[0][1],right_points[0][1])], right_points[1]

        # calculate the height of the object (in pixel)
        column = [image[y][v2[0]].tolist() for y in range(image.shape[0])]
        column.reverse()

        height = 0
        tolerance = 30
        main_color = image[min([v1[1],v2[1]])][v2[0]]

        # iterate over the column with v2 as end point
        for pixel in column:

            distance = geometric_utils.point_distance(main_color, pixel)

            if distance > tolerance and height > 1:
                break
            
            height = height + 1

        # convert points coordinates to frame coordinates
        min_x = min(box, key=lambda x: x[0])[0]
        max_x = max(box, key=lambda x: x[0])[0]

        min_y = min(box, key=lambda x: x[1])[1]
        max_y = max(box, key=lambda x: x[1])[1]

        v1[0], v1[1] = v1[0] + min_x, v1[1] + min_y
        v2[0], v2[1] = v2[0] + min_x, v2[1] + min_y
        v3[0], v3[1] = v3[0] + min_x, v3[1] + min_y

        v1, v2, v3 = tuple(v1), tuple(v2), tuple(v3)

        # draw object main axes
        cv2.line(frame, v1, v2, (0,255,0))
        cv2.line(frame, v3, v2, (0,0,255))
        cv2.line(frame, (v2[0],v2[1] - height), (v2[0],v2[1]), (255,0,0))

        cv2.line(frame, v1, v1, (0,0,255),3)
        cv2.line(frame, v2, v2, (0,0,255),3)
        cv2.line(frame, v3, v3, (0,0,255),3)
        cv2.line(frame, (v2[0],v2[1] - height), (v2[0],v2[1] - height), (0,0,255),3)

        # convert points
        v1 = convert_to_gazebo_world_frame(v1,2)
        v2 = convert_to_gazebo_world_frame(v2,2)
        v3 = convert_to_gazebo_world_frame(v3,2)

        # calculate angle
        if v2[0] - v1[0] != 0:
            angle = math.atan((v1[1]-v2[1]) / (v1[0]-v2[0]))
        else:
            angle = 0
        
        d12 = geometric_utils.point_distance(v1,v2)
        d23 = geometric_utils.point_distance(v2,v3)

        if(d12 > d23):
            angle = angle + math.pi/2
        
        if math.degrees(angle) > 90:
            angle = angle % (math.pi / 2)

        obj["rotations"] = {}
        obj["rotations"]["z"] = angle

        #cv2.imshow("debug",frame)

    return objects, image

def process_image(img: np.ndarray, render: bool = False) -> np.ndarray:
    """
    Converts 2D coordinates of the object bounding box to gazebo world frame coordinates 

    Input parameters:
    - ``img`` image retrieved by the zed camera

    Output parameters:
    ``processed image``
    """

    global TABLE_PERCENTAGE

    # extract table from image
    img = extract_table(img = img)
    clean_image = img.copy()

    # half table
    for y in range(int(img.shape[0]*TABLE_PERCENTAGE)):
        img[y] = 0

    # detect objects
    objects, frame = detect_objects(img = img, render = render)
    
    # process objects
    objects, image = process_objects(frame = clean_image, objects = objects)

    # send detected objects
    send_objects(objects = objects)

    # print detected objects
    print_objects(objects = objects)

    return frame

def background_detection(msg: Image) -> None:
    """
    Receives and elaborates zed images in background (no display)

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

    global WINDOW_RECT

    # convert received image (bgr8 format) to a cv2 image
    img = CvBridge().imgmsg_to_cv2(msg, "bgr8")

    # process image
    frame = process_image(img = img, render = True)

    # display processed image
    frame = frame[WINDOW_RECT["min_y"] : WINDOW_RECT["max_y"] + 1, WINDOW_RECT["min_x"]: WINDOW_RECT["max_x"] + 1]

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
