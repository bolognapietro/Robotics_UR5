import os
from os import walk, getcwd
from os.path import join, basename

import subprocess

from typing import Union
from termcolor import colored

import random
from time import sleep
import uuid

import numpy as np
import math

import rospy as ros
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, GetModelState, GetModelStateRequest, GetWorldProperties, DeleteModel
from std_msgs.msg import Bool
from tf.transformations import quaternion_from_euler, euler_from_quaternion

clear = lambda: subprocess.run("cls" if os.name == "nt" else "clear", shell=True)

BLENDER_MODELS_PATH = join(getcwd(),"vision","models")

SLEEP_TIME = 3

MARGIN = 0.07

X_MIN = 0 + MARGIN
X_MAX = 0.3 - MARGIN

Y_MIN = 0.25 + MARGIN
Y_MAX = 0.65 - MARGIN

MODELS_NUMBER = 4
MAX_MODELS = math.floor((X_MAX - X_MIN) * (Y_MAX - Y_MIN) / (math.pi * (MARGIN/2)**2))

assert MAX_MODELS >= MODELS_NUMBER, f"Too many models. Maximum number of models: {MAX_MODELS}"

def execute_request(service_name: str, service_class, request: Union[str, None] = None, wait: float = 0.2):
    """
    Execute ROS request

    Args:
        service_name (str): Name of service to call.
        service_class: Auto-generated service class.
        request (str, optional): Model's id (name). Defaults to None.
        wait (float, optional): Ros sleep time after executing the request. Defaults to 0.2.

    Returns:
        Any: response of the executed request
    """

    spawn_srv = ros.ServiceProxy(service_name, service_class)
    spawn_srv.wait_for_service()

    if request != None:
        response = spawn_srv.call(request)
    else:
        response = spawn_srv.call()

    ros.sleep(wait)

    return response

def create_spawnmodel_request(name: str, model: str, position: tuple, orientation: tuple) -> SpawnModelRequest:
    """
    Create SpawnModel request

    Args:
        name (str): Model's name.
        model (str): .ptl model file.
        position (tuple): Model's position (x,y,z).
        orientation (tuple): Model's orientation (roll, pitch, yaw).

    Returns:
        SpawnModelRequest: created request.
    """

    schema = """<?xml version="1.0"?>
    <sdf version="1.4"> 
    <gravity>0.0 0.0 -9.81</gravity>
    <model name="MODELNAME">
        <static>false</static>
        <link name="link">
        <inertial>
            <mass>1.0</mass>
            <inertia>
                <ixx>20.0</ixx>
                <iyy>20.0</iyy>
                <izz>20.0</izz>
                <ixy>0.0</ixy>
                <ixz>0.0</ixz>
                <iyz>0.0</iyz>
            </inertia>
        </inertial>
        <pose>0 0 0 0 0 0</pose>

        <visual name="visual">
            <geometry>
            <mesh>
                <uri>URL_MODEL</uri>
            </mesh>
            </geometry>
            <material>
            <script>
                <uri>file://media/materials/scripts/gazebo.material</uri>
                <name>COLOR</name>
            </script>
            </material>
        </visual>
        
        <collision name="collision">
            <geometry>
            <mesh>
                <uri>URL_MODEL</uri>
            </mesh>
            </geometry>
        </collision>
        </link>
    </model>
    </sdf>
    """

    schema = schema.replace('URL_MODEL', model)
    
    color_list = ["Purple", "Red", "Orange", "Blue"]
    schema = schema.replace('COLOR', "Gazebo/" + random.choice(color_list))

    request = SpawnModelRequest()
    request.model_name = name
    request.model_xml = schema
    request.initial_pose.position.x = position[0]
    request.initial_pose.position.y = position[1]
    request.initial_pose.position.z = position[2]

    request.initial_pose.orientation.x = orientation[0]
    request.initial_pose.orientation.y = orientation[1]
    request.initial_pose.orientation.z = orientation[2]
    request.initial_pose.orientation.w = orientation[3]

    return request

def get_world_models() -> list:
    """
    Retrieve world models

    Returns:
        list: world models list.
    """

    response = execute_request(service_name = "/gazebo/get_world_properties", service_class = GetWorldProperties)

    return [model for model in response.model_names if len(model) == 36]

def get_model_state(name : str) -> tuple:
    """
    Retrieve model's position and orientation

    Returns:
        tuple: position and orientation.
    """

    request = GetModelStateRequest()
    request.model_name = name

    response = execute_request(request = request, service_name = "/gazebo/get_model_state", service_class = GetModelState)

    position = response.pose.position
    position = (position.x, position.y, position.z)
    position = list(position)
    position = [round(pos,5) for pos in position]

    orientation = response.pose.orientation
    orientation = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
    orientation = list(orientation)
    orientation = [round(ori,5) for ori in orientation]

    return tuple(position), tuple(orientation)

def delete_model(name: str) -> bool:
    """
    Delete a specific model

    Args:
        name (str): Model's name.

    Returns:
        bool: result of the request. True success, False, fail.
    """

    response = execute_request(request = name, service_name = "/gazebo/delete_model", service_class = DeleteModel)
    
    return response.success

def delete_all_models() -> None:
    """
    Delete all the worlds models
    """

    for model in get_world_models():

        print(colored(f"Deleting {model}...","yellow"),end="")
        delete_model(model)
        print(colored(f"OK","yellow"))

    assert not len(get_world_models()), "Unable to delete all the models."

def random_model() -> str:
    """
    Pick a random .stl model file

    Returns:
        str: .stl file.
    """

    global BLENDER_MODELS_PATH

    models = next(walk(BLENDER_MODELS_PATH), (None, None, []))[2]
    return join(BLENDER_MODELS_PATH,random.choice(models))

def random_position(precision: int = 3) -> tuple:
    """
    Generate random position

    Args:
        precision (int, optional): Number of digits of the coordinates. Defaults to 3.

    Returns:
        tuple: Random position.
    """

    global MARGIN, X_MAX, X_MIN, Y_MAX, Y_MIN

    precision = 10**precision

    x_min = X_MIN * precision
    x_max = X_MAX * precision

    y_min = Y_MIN * precision
    y_max = Y_MAX * precision

    z = 0.93

    while True:

        x = random.randrange(int(x_min),int(x_max)) / precision
        y = random.randrange(int(y_min),int(y_max)) / precision

        collision = False

        for model in get_world_models():
            position, orientation = get_model_state(name = model)

            if math.dist((x,y), position[:2]) < MARGIN:
                collision = True
                break
        
        if not collision:
            break

    return (x,y,z)

def random_orientation(index: int = None) -> tuple:
    """
    Generate pseudo-random orientation

    Args:
        index (float, optional): Select a specific orientation from the list.

    Returns:
        tuple: NDArray[float64] random orientation and it's index.
    """

    orientations = [
        quaternion_from_euler(0,0,0), #?            0
        quaternion_from_euler(1.571,0,0), #?        1
        quaternion_from_euler(-1.571,0,0), #?       2
        #quaternion_from_euler(1.571,0,1.571), #?    3
        quaternion_from_euler(1.571,0,-1.571), #?   4
        #quaternion_from_euler(0,1.571,0), #?        5
        quaternion_from_euler(0,-1.571,0), #?       6
        quaternion_from_euler(1.571,1.571,0), #?    7
        quaternion_from_euler(-1.571,1.571,0), #?   8
        quaternion_from_euler(3.141,0,0), #?        9
        quaternion_from_euler(3.141,0,1.571) #?     10
    ]
    
    if index != None and index >= 0 and index < len(orientations):
        return orientations[index], index
    
    orientation_index = random.randint(0, len(orientations) - 1)
    orientation = orientations[orientation_index]
    
    return orientation, orientation_index

def random_name() -> str:
    """
    Generate random name

    Returns:
        str: Random name.
    """

    return str(uuid.uuid4())

def print_model(name: str, model: str, position: tuple, orientation: tuple, orientation_index: int) -> None:
    """
    Print model

    Args:
        str: Model's name.
        str: Model's .stl file.
        tuple: Model's position.
        tuple: Model's orientation.
        int: Model's orientation index.
    """

    print(f"Name: {name}")
    print(f"Model: {basename(model).split('.')[0]}")
    print(f"Position (gazebo): {tuple(round(pos, 2) for pos in position)}")
    print(f"Position (robot): {tuple([round(position[0] - 0.5,2), round(position[1] - 0.35,2), -0.74])}")
    print(f"Orientation: {tuple(round(np.rad2deg(angle) / 90)*90 for angle in euler_from_quaternion(orientation))}")
    print(f"Orientation index: {orientation_index}\n")

def assignment1():
    """
    There is only one object in the initial stand, which is positioned with its base “naturally” in contact with the ground. 
    The object can be of any of the classes specified by the project. 
    Each class has an assigned position on the final stand, which is marked by a coloured shape representing the silhouette of the object.
    KPI 1-1 time to detect the position of the object
    KPI 1-2 time to move the object between its initial and its final positions, counting from the instant in which both of them have been identified.
    """

    print(f"Deleting models...")
    delete_all_models()

    clear()
    
    name = random_name()
    print(colored(f"Generating {name}...","yellow"),end="\r")

    model = random_model()
    position = random_position()
    orientation, orientation_index = random_orientation(index=0)

    request = create_spawnmodel_request(name = name, model = model, position = position, orientation = orientation)
    execute_request(request = request, service_name = "/gazebo/spawn_sdf_model", service_class = SpawnModel)

    print("                                                  ",end="\r")
    print_model(name = name, model = model, position = position, orientation = orientation, orientation_index = orientation_index)
    
def assignment2():
    """
    There are multiple objects on the initial stand, one for each class. 
    There is no specific order in the initial configuration, except that the base of the object is “naturally” in contact with the ground. 
    Each object has to be picked up and stored in the position prescribed for its class and marked by the object’s silhouette. 
    KPI 2-1: Total time to move all the objects from their initial to their final positions.
    """

    global MODELS_NUMBER

    delete_all_models()
    
    clear()

    for _ in range(MODELS_NUMBER):

        name = random_name()
        print(colored(f"Generating {name}...","yellow"),end="\r")

        model = random_model()
        position = random_position()
        orientation, orientation_index = random_orientation(index=0)

        request = create_spawnmodel_request(name = name, model = model, position = position, orientation = orientation)
        execute_request(request = request, service_name = "/gazebo/spawn_sdf_model", service_class = SpawnModel)

        print("                                                  ",end="\r")
        print_model(name = name, model = model, position = position, orientation = orientation, orientation_index = orientation_index)

def assignment3():
    """
    There are multiple objects on the initial stand, and there can be more than one object for each class. 
    The objects are positioned randomly on the stand but would not stand or lean on each other. 
    An object could be lying on one of its lateral sides or on its top. 
    Each object has to be stored in the position prescribed by its class. 
    Objects of the same class have to be stacked up to form a tower.
    KPI 3-1: Total time to move all the objects from their initial to their final positions.
    """

    global MODELS_NUMBER

    delete_all_models()

    clear()

    for _ in range(MODELS_NUMBER):

        name = random_name()
        print(colored(f"Generating {name}...","yellow"),end="\r")

        model = random_model()
        position = random_position()
        orientation, orientation_index = random_orientation()

        request = create_spawnmodel_request(name = name, model = model, position = position, orientation = orientation)
        execute_request(request = request, service_name = "/gazebo/spawn_sdf_model", service_class = SpawnModel)

        print("                                                  ",end="\r")
        print_model(name = name, model = model, position = position, orientation = orientation, orientation_index = orientation_index)

def assignment4():
    """
    The objects on the initial stand are those needed to create a composite object with a known design (e.g., a castle). 
    The objects are positioned randomly on the stand. 
    An object could be lying on one of its lateral sides or on its top. 
    The objects could also stand or lean on each other. 
    The manipulator has to pick them up in sequence and create the desired composite object on the final stand.
    """
    pass

def start_zed():
    """
    Once executed, the zed camera is allowed to process the objects
    """

    pub = ros.Publisher("/start_zed", Bool, queue_size=10)

    msg = Bool()
    msg.data = True
    
    pub.publish(msg)

if __name__ == "__main__":

    ros.init_node("assignments", anonymous=True)

    try:
        
        while True:

            clear()

            print("1) Assignment 1")
            print("2) Assignment 2")
            print("3) Assignment 3")
            print("4) Assignment 4")
            option = int(input("\n> "))

            if option < 1 or option > 4:
                continue
            
            clear()

            assignment = eval(f"assignment{option}")
            assignment()

            for i in range(SLEEP_TIME):
                print(colored(f"Waiting {SLEEP_TIME - i}", "yellow"),end="\r")
                sleep(1)

            start_zed()
            ros.sleep(1)
            start_zed()

            print(f"                                                                ",end="\r")
            input("Done!")

    except KeyboardInterrupt:
        pass
