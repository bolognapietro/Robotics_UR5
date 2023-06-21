import os
import subprocess
import random
import uuid

import rospy as ros

from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SetModelState, SetModelStateRequest, GetModelState, GetModelStateRequest, GetWorldProperties, DeleteModel
from os import walk, getcwd
from os.path import join
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from termcolor import colored

BLENDER_MODELS_PATH = join(getcwd(),"models")
clear = lambda: subprocess.run("cls" if os.name == "nt" else "clear", shell=True)


##########################################################################


def execute_request(service_name: str, service_class, request = None, wait: float = 0.2):

    spawn_srv = ros.ServiceProxy(service_name, service_class)
    spawn_srv.wait_for_service()

    if request != None:
        response = spawn_srv.call(request)
    else:
        response = spawn_srv.call()

    ros.sleep(wait)

    return response


##########################################################################


def create_spawnmodel_request(name: str, model: str, position: tuple):
    
    schema = """<?xml version="1.0"?>
    <sdf version="1.4"> 
    <gravity>0.0 0.0 -9.81</gravity>
    <model name="MODELNAME">
        <static>false</static>
        <link name="link">
        <inertial>
            <mass>1.0</mass>
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

    req = SpawnModelRequest()
    req.model_name = name
    req.model_xml = schema
    req.initial_pose.position.x = position[0]
    req.initial_pose.position.y = position[1]
    req.initial_pose.position.z = position[2]

    q = quaternion_from_euler(0,0,0) #(random_angle(),random_angle(),random_angle())
    req.initial_pose.orientation.x = q[0]
    req.initial_pose.orientation.y = q[1]
    req.initial_pose.orientation.z = q[2]
    req.initial_pose.orientation.w = q[3]

    return req

def random_model(path: str) -> str:

    models = next(walk(path), (None, None, []))[2]
    return join(path,random.choice(models))

def random_position(x_min: float = 0.2, x_max: float = 0.5, y_min: float = 0.2, y_max: float = 0.65, z: float = 0.87, precision: int = 3) -> tuple:

    precision = 10**precision

    x_min = x_min * precision
    x_max = x_max * precision

    y_min = y_min * precision
    y_max = y_max * precision

    x = random.randrange(int(x_min),int(x_max)) / precision
    y = random.randrange(int(y_min),int(y_max)) / precision

    return (x,y,z)

def random_angle(precision: int = 3) -> float:
    return random.randint(0,314000) / 100000

def random_name() -> str:
    return str(uuid.uuid4())

def spawn_model(path: str):

    model = random_model(path = path)
    position = random_position()
    name = random_name()

    request = create_spawnmodel_request(name = name, model = model, position = position)
    execute_request(request = request, service_name = "/gazebo/spawn_sdf_model", service_class = SpawnModel)

    return name


##########################################################################


def create_getmodelstate_request(name: str):
    
    request = GetModelStateRequest()
    request.model_name = name

    return request

def get_model_state(name : str):

    request = create_getmodelstate_request(name = name)
    response = execute_request(request = request, service_name = "/gazebo/get_model_state", service_class = GetModelState)

    position = response.pose.position
    position = (position.x, position.y, position.z)
    position = list(position)
    position = [round(pos,5) for pos in position]

    orientation = response.pose.orientation
    orientation = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
    orientation = list(orientation)
    orientation = [round(ori,5) for ori in orientation]

    return list(position), list(orientation)


##########################################################################


def create_setmodelstate_request(name: str, position: list, orientation: list):

    request = SetModelStateRequest()
    request.model_state.model_name = name

    request.model_state.pose.position.x = position[0]
    request.model_state.pose.position.y = position[1]
    request.model_state.pose.position.z = position[2]

    angle = quaternion_from_euler(orientation[0], orientation[1], orientation[2])

    request.model_state.pose.orientation.x = angle[0]
    request.model_state.pose.orientation.y = angle[1]
    request.model_state.pose.orientation.z = angle[2]
    request.model_state.pose.orientation.w = angle[3]

    return request

def set_model_state(name: str, position: list, orientation: list):

    request = create_setmodelstate_request(name = name, position = position, orientation = orientation)
    response = execute_request(request = request, service_name = "/gazebo/set_model_state", service_class = SetModelState)

    return response.success


##########################################################################


def get_world_properties():

    response = execute_request(service_name = "/gazebo/get_world_properties", service_class = GetWorldProperties)

    return [model for model in response.model_names if len(model) == 36]


##########################################################################


def delete_model(name: str):

    response = execute_request(request = name, service_name = "/gazebo/delete_model", service_class = DeleteModel)
    
    return response.success


##########################################################################


if __name__ == '__main__':

    try:
        while True:
            
            clear()

            print("1) Spawn model")
            print("2) View model")
            print("3) Modify model")
            print("4) Delete model")
            print("5) Models list")
            option = input("\n> ")
            
            if not option.isdigit():
                continue

            option = int(option)

            clear()

            if option == 1:
                print(colored("Spawning new model...","yellow"))
                spawn_model(BLENDER_MODELS_PATH)

            elif option in [2,3,4]:

                clear()
                print(colored(f"Retrieving models...","yellow"))

                models = get_world_properties()

                clear()

                print("Choose a model: \n")

                if not len(models):
                    input(colored("No model found"))
                    continue
                
                if option == 4:
                    print(f"0) Delete all models")

                for i, model in enumerate(models):
                    print(f"{i+1}) {model}")
                
                op = input("\n> ")

                if not op.isdigit():
                    continue

                op = int(op) - 1

                if op >= len(models) or (op < 0 and (op < -1 and option == 4)):
                    continue

                if option == 4:

                    if op == -1:

                        print("\n")

                        for name in models:
                            print(colored(f"Deleting {name}...","yellow"))
                            delete_model(name = name)
                        
                    else:
                        print(colored(f"\nDeleting {name}...","yellow"))
                        delete_model(name = models[op])

                    continue
                
                name = models[op]

                print(colored(f"\nRetrieving {name}...","yellow"))
                position, orientation = get_model_state(name = name)

                clear()

                print(f"Model: {name}")
                print(f"Position (x,y,z): {tuple(position)}")
                print(f"Orientation (x,y,z): {tuple(orientation)}")
                
                if option == 2:
                    input("\n")
                    continue

                new_pos_x = input(f"\nNew position x: ")
                new_pos_y = input(f"New position y: ")
                new_pos_z = input(f"New position z: ")

                new_pos_x.isdecimal
                if not len(new_pos_x):
                    new_pos_x = position[0]
                elif not new_pos_x.replace(".", "", 1).replace("-","",1).isdigit():
                    continue
                else:
                    position[0] = float(new_pos_x)

                if not len(new_pos_y):
                    new_pos_y = position[1]
                elif not new_pos_y.replace(".", "", 1).replace("-","",1).isdigit():
                    continue
                else:
                    position[1] = float(new_pos_y)

                if not len(new_pos_z):
                    new_pos_z = position[2]
                elif not new_pos_y.replace(".", "", 1).replace("-","",1).replace("-","",1).isdigit():
                    continue
                else:
                    position[2] = float(new_pos_z)
                
                new_or_x = input(f"\nNew orientation x: ")
                new_or_y = input(f"New orientation y: ")
                new_or_z = input(f"New orientation z: ")

                if not len(new_or_x):
                    new_or_x = orientation[0]
                elif not new_or_x.replace(".", "", 1).replace("-","",1).isdigit():
                    continue
                else:
                    orientation[0] = float(new_or_x)
                
                if not len(new_or_y):
                    new_or_y = orientation[1]
                elif not new_or_y.replace(".", "", 1).replace("-","",1).isdigit():
                    continue
                else:
                    orientation[1] = float(new_or_y)

                if not len(new_or_z):
                    new_or_z = orientation[2]
                elif not new_or_z.replace(".", "", 1).replace("-","",1).isdigit():
                    continue
                else:
                    orientation[2] = float(new_or_z)
                
                print(colored(f"\nModifying {name}...","yellow"))
                set_model_state(name = name, position = position, orientation = orientation)
                
                clear()

                print(colored(f"\nRetrieving {name}...","yellow"))
                position, orientation = get_model_state(name = name)

                clear()

                print(f"Model: {name}")
                print(f"Position (x,y,z): {tuple(position)}")
                print(f"Orientation (x,y,z): {tuple(orientation)}")

                input("\n")


            elif option == 5:
                print(colored(f"Retrieving models...","yellow"))

                models = get_world_properties()

                clear()

                if not len(models):
                    input(colored("No model found"))
                    continue

                print("Models list: \n")

                for model in models:
                    print(model)
                
                input("\n")

    except KeyboardInterrupt:
        pass
