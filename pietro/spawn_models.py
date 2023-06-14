#!/usr/bin/env python

import rospy
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse
from copy import deepcopy
from tf.transformations import quaternion_from_euler
import random
from datetime import datetime
import numpy as np
from math import degrees

PATH = '/home/pietro/ros_ws/src/locosim/robot_control/lab_exercises/lab_palopoli/models/'

#        <pose>0.000061e-02 0.000456e-02 0.000295e-02 0 0 0</pose> 
#        <ixx>50.790573e-10</ixx>
#        <ixy>0.005056e-10</ixy>
#        <ixz>-0.000368e-10</ixz>
#        <iyy>23.509121e-10</iyy>
#        <iyz>-0.005099e-10</iyz>
#        <izz>50.08608e-10</izz>


sdf_cube = """<?xml version="1.0"?>
<sdf version="1.4"> 
  <gravity>0.0 0.0 -9.81</gravity>
  <model name="MODELNAME">
    <static>false</static>
    <link name="link">
    <inertial>
        <mass>0.03</mass>
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

def create_cube_request(modelname, px, py, pz, rr, rp, ry, type):
    """Create a SpawnModelRequest with the parameters of the cube given.
    modelname: name of the model for gazebo
    px py pz: position of the cube (and it's collision cube)
    rr rp ry: rotation (roll, pitch, yaw) of the model
    sx sy sz: size of the cube"""
    cube = deepcopy(sdf_cube)
    # Replace size of model
    # size_str = str(round(sx, 3)) + " " + \
    #     str(round(sy, 3)) + " " + str(round(sz, 3))
    #cube = cube.replace('SIZEXYZ', size_str)
    # Replace URL
    cube = cube.replace('URL_MODEL', PATH + str(type) + ".stl")
    
    #Replace Color
    color_list = ["Purple", "Red", "Orange", "Blue", ]
    cube = cube.replace('COLOR', "Gazebo/" + random.choice(color_list))

    req = SpawnModelRequest()
    req.model_name = modelname
    req.model_xml = cube
    req.initial_pose.position.x = px
    req.initial_pose.position.y = py
    req.initial_pose.position.z = pz

    q = quaternion_from_euler(rr, rp, ry)
    req.initial_pose.orientation.x = q[0]
    req.initial_pose.orientation.y = q[1]
    req.initial_pose.orientation.z = q[2]
    req.initial_pose.orientation.w = q[3]

    return req

def randNum(min, max):
    num = round(random.uniform(min, max), 2)
    return num


if __name__ == '__main__':
    #rospy.init_node('spawn_models')
    spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    #rospy.loginfo("Waiting for /gazebo/spawn_sdf_model service...")
    spawn_srv.wait_for_service()
    #rospy.loginfo("Connected to service!")

    
    #rospy.loginfo("Spawning cube")

    print("ModelName:\n  1:(X1-Y1-Z2)\n  2:(X1-Y2-Z1)\n  3:(X1-Y2-Z2-CHAMFER)\n  4:(X1-Y2-Z2-TWINFILLET)\n  5:(X1-Y2-Z2)\n  6:(X1-Y3-Z2)\n  7:(X1-Y4-Z1)\n  8:(X1-Y4-Z2)\n")

    n = input("Number: ")
    type = ["X1-Y1-Z2", "X1-Y2-Z1", "X1-Y2-Z2-CHAMFER", "X1-Y2-Z2-TWINFILLET", "X1-Y2-Z2", "X1-Y3-Z2", "X1-Y4-Z1", "X1-Y4-Z2", "X2-Y2-Z2-FILLET", "X2-Y2-Z2"] 
    
    if(n == '1'):
      sel = type[0]
    elif(n == '2'):
      sel = type[1] 
    elif(n == '3'):
      sel = type[2]
    elif(n == '4'):
      sel = type[3]
    elif(n == '5'):
      sel = type[4]
    elif(n == '6'):
      sel = type[5]
    elif(n == '7'):
      sel = type[6]
    elif(n == '8'):
      sel = type[7]
    else:
      print("ERR: Invalid number!")
      quit()

    # Spawn object 1
    #x_min = 0.01, x_max = 0.98
    #y_min = 0.2, y_max = 0.79
    #z = 0.9
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    x = randNum(0.0, 0.42)#0.25  #randNum(0.25, 0.26)   #0.42 + 0.5      -- (0.00, 0.42) --              LEFT: 0.89-0.90  RIGHT: 0.25-0.26
    y = randNum(0.54, 0.75)#0.75  #randNum(0.54, 0.79)   #0.2 + 0.35     -- (0.54, 0.79) -- 
    z = 0.93

    x = round(x,2)
    y = round(y,2)

    rot_y = randNum(0, np.pi)

    name = sel + " - " + time
    req1 = create_cube_request(name,
                              x, y, z,  # position
                              0,0,rot_y,         # For RIGTH: [np.pi/2, np.pi, -np.pi] [np.pi/2, 0, 0] 
                              sel)  # size
    
    # 2a: [np.pi/2, 0, 0]         2c: [np.pi/2, 0, np.pi/2]
    # 2b: [np.pi/2, 0, np.pi]     2d: [np.pi/2, 0, -np.pi/2]
    # 3a: [0, np.pi/2, 0]         3c: [0, -np.pi/2, np.pi/2]
    # 3b: [0, -np.pi/2, 0]        3d: [0, -np.pi/2, -np.pi/2]
    x = x - 0.5
    y = y - 0.35      #0.37    
    z = z - 1.64

    rot_y = degrees(rot_y)
    print("x: " + str(round(x, 2)), " y: " + str(round(y, 2)), " z: " + str(round(z, 2)), "rotation: ", str(rot_y))

    spawn_srv.call(req1)
    rospy.sleep(0.2)