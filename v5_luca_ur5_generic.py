# -*- coding: utf-8 -*-
"""
Created on 3 May  2022

@author: mfocchi
"""

from __future__ import print_function

import os
import rospy as ros
import sys
# messages for topic subscribers
from geometry_msgs.msg import WrenchStamped
from std_srvs.srv import Trigger, TriggerRequest

# ros utils
import roslaunch
import rosnode
import rosgraph
import rospkg

#other utils
from base_controllers.utils.math_tools import *
import pinocchio as pin
np.set_printoptions(threshold=np.inf, precision = 5, linewidth = 1000, suppress = True)
from termcolor import colored
from base_controllers.utils.common_functions import plotJoint, plotEndeff
import  params as conf
robotName = "ur5"

# controller manager management
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
from std_msgs.msg import Float64MultiArray
from base_controllers.base_controller_fixed import BaseControllerFixed
import tf
from rospy import Time
import time
from base_controllers.components.controller_manager import ControllerManager
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
import numpy as np

# from kinematics import inverse_kinematics,direct_kinematics,check_point, check_collision, check_angles
# import motionPlan as mp
from v5_kinematics_dev import inverse_kinematics,direct_kinematics,check_point, check_collision, check_angles
import v5_motionPlan_dev as mp
import spawn_models as spm

import zed.zed2 as zed
from std_msgs.msg import Float32MultiArray

class Ur5Generic(BaseControllerFixed):
    
    def __init__(self, robot_name="ur5"):
        super().__init__(robot_name=robot_name)
        self.real_robot = conf.robot_params[self.robot_name]['real_robot']
        #self.homing_flag = True
        self.obj = False
        self.gripper_on = 1
        if (conf.robot_params[self.robot_name]['control_type'] == "torque"):
            self.use_torque_control = 1
        else:
            self.use_torque_control = 0

        if self.use_torque_control and self.real_robot:
            print(colored(
                "ERRORS: unfortunately...you cannot use ur5 in torque control mode, talk with your course coordinator to buy a better robot...:))",
                'red'))
            sys.exit()

        if conf.robot_params[self.robot_name]['gripper_sim']:
            self.gripper = True
        else:
            self.gripper = False

        self.controller_manager = ControllerManager(conf.robot_params[self.robot_name])

        self.world_name = None # only the workbench
        #self.world_name = 'empty.world'
        #self.world_name = 'palopoli.world'

        print("Initialized ur5 generic  controller---------------------------------------------------------------")

    def startRealRobot(self):
        os.system("killall rviz gzserver gzclient")
        print(colored('------------------------------------------------ROBOT IS REAL!', 'blue'))

        # uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        # launch_file = rospkg.RosPack().get_path('ur_robot_driver') + '/launch/ur5e_bringup.launch'
        # cli_args = [launch_file,
        #             'headless_mode:=true',
        #             'robot_ip:=192.168.0.100',
        #             'kinematics_config:=/home/laboratorio/my_robot_calibration_1.yaml']

        # roslaunch_args = cli_args[1:]
        # roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        # parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)

        if (not rosgraph.is_master_online()) or (
                "/" + self.robot_name + "/ur_hardware_interface" not in rosnode.get_node_names()):
            print(colored('No ur driver found!', 'blue'))
            sys.exit()
            #print(colored('Launching the ur driver!', 'blue'))
            #parent.start()

        # run rviz
        package = 'rviz'
        executable = 'rviz'
        args = '-d ' + rospkg.RosPack().get_path('ros_impedance_controller') + '/config/operator.rviz'
        node = roslaunch.core.Node(package, executable, args=args)
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        process = launch.launch(node)

    def loadModelAndPublishers(self, xacro_path):
        super().loadModelAndPublishers(xacro_path)

        self.sub_ftsensor = ros.Subscriber("/" + self.robot_name + "/wrench", WrenchStamped,
                                           callback=self._receive_ftsensor, queue_size=1, tcp_nodelay=True)
        self.switch_controller_srv = ros.ServiceProxy(
            "/" + self.robot_name + "/controller_manager/switch_controller", SwitchController)
        self.load_controller_srv = ros.ServiceProxy("/" + self.robot_name + "/controller_manager/load_controller",
                                                    LoadController)
        # specific publisher for joint_group_pos_controller that publishes only position
        self.pub_reduced_des_jstate = ros.Publisher("/" + self.robot_name + "/joint_group_pos_controller/command",
                                                    Float64MultiArray, queue_size=10)

        self.zero_sensor = ros.ServiceProxy("/" + self.robot_name + "/ur_hardware_interface/zero_ftsensor", Trigger)
        self.controller_manager.initPublishers(self.robot_name)
        #  different controllers are available from the real robot and in simulation
        if self.real_robot:
            # specific publisher for joint_group_pos_controller that publishes only position
            self.pub_reduced_des_jstate = ros.Publisher("/" + self.robot_name + "/joint_group_pos_controller/command",
                                                        Float64MultiArray, queue_size=10)
            self.available_controllers = [
                "joint_group_pos_controller",
                "scaled_pos_joint_traj_controller" ]
        else:
            self.available_controllers = ["joint_group_pos_controller",
                                          "pos_joint_traj_controller" ]
        self.active_controller = self.available_controllers[0]

        self.broadcaster = tf.TransformBroadcaster()
        # store in the param server to be used from other planners
        self.utils = Utils()
        self.utils.putIntoGlobalParamServer("gripper_sim", self.gripper)

        self.sub_pointcloud = ros.Subscriber("/ur5/zed_node/point_cloud/cloud_registered", PointCloud2,   callback=self.receive_pointcloud, queue_size=1)
        # ros.Subscriber("/zedObjects", Float32MultiArray, callback = test, queue_size=5)

    def _receive_ftsensor(self, msg):
        contactForceTool0 = np.zeros(3)
        contactMomentTool0 = np.zeros(3)
        contactForceTool0[0] = msg.wrench.force.x
        contactForceTool0[1] = msg.wrench.force.y
        contactForceTool0[2] = msg.wrench.force.z
        contactMomentTool0[0] = msg.wrench.torque.x
        contactMomentTool0[1] = msg.wrench.torque.y
        contactMomentTool0[2] = msg.wrench.torque.z
        self.contactForceW = self.w_R_tool0.dot(contactForceTool0)
        self.contactMomentW = self.w_R_tool0.dot(contactMomentTool0)

    def deregister_node(self):
        print( "deregistering nodes"     )
        self.ros_pub.deregister_node()
        if not self.real_robot:
            os.system(" rosnode kill /"+self.robot_name+"/ros_impedance_controller")
            os.system(" rosnode kill /gzserver /gzclient")
                                                                                                                                     
    def updateKinematicsDynamics(self):
        # q is continuously updated
        # to compute in the base frame  you should put neutral base
        self.robot.computeAllTerms(self.q, self.qd)
        # joint space inertia matrix
        self.M = self.robot.mass(self.q)
        # bias terms
        self.h = self.robot.nle(self.q, self.qd)
        #gravity terms
        self.g = self.robot.gravity(self.q)
        #compute ee position  in the world frame
        frame_name = conf.robot_params[self.robot_name]['ee_frame']
        # this is expressed in the base frame
        self.x_ee = self.robot.framePlacement(self.q, self.robot.model.getFrameId(frame_name)).translation
        self.w_R_tool0 = self.robot.framePlacement(self.q, self.robot.model.getFrameId(frame_name)).rotation
        # camera frame
        self.x_c= self.robot.framePlacement(self.q, self.robot.model.getFrameId("zed2_left_camera_optical_frame")).translation
        self.w_R_c = self.robot.framePlacement(self.q, self.robot.model.getFrameId("zed2_left_camera_optical_frame")).rotation

        # compute jacobian of the end effector in the base or world frame (they are aligned so in terms of velocity they are the same)
        self.J6 = self.robot.frameJacobian(self.q, self.robot.model.getFrameId(frame_name), False, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)                    
        # take first 3 rows of J6 cause we have a point contact            
        self.J = self.J6[:3,:] 
        # broadcast base world TF
        self.broadcaster.sendTransform(self.base_offset, (0.0, 0.0, 0.0, 1.0), Time.now(), '/base_link', '/world')


    def startupProcedure(self):
        if (self.use_torque_control):
            #set joint pdi gains
            self.pid.setPDjoints( conf.robot_params[self.robot_name]['kp'], conf.robot_params[self.robot_name]['kd'], np.zeros(self.robot.na))
        if (self.real_robot):
            self.zero_sensor()
        self.u.putIntoGlobalParamServer("real_robot",  self.real_robot)
        print(colored("finished startup -- starting controller", "red"))

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        print('Available controllers: ',self.available_controllers)
        print('Controller manager: loading ', target_controller)

        other_controllers = (self.available_controllers)
        other_controllers.remove(target_controller)
        print('Controller manager:Switching off  :  ',other_controllers)

        srv = LoadControllerRequest()
        srv.name = target_controller

        self.load_controller_srv(srv)  
        
        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers 
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_controller_srv(srv)
        self.active_controller = target_controller

    def deregister_node(self):
        super().deregister_node()
        if not self.real_robot:
            os.system(" rosnode kill /"+self.robot_name+"/ros_impedance_controller")
            os.system(" rosnode kill /gzserver /gzclient")

    def plotStuff(self):
        plotJoint('position', 0, self.time_log, self.q_log, self.q_des_log)
    
    def receive_pointcloud(self, msg):
        #in the zed2_left_camera_optical_frame
        points_list = []
        for data in point_cloud2.read_points(msg, field_names=['x','y','z'], skip_nans=False, uvs=[(640, 360)]):
            points_list.append([data[0], data[1], data[2]])
        #print("Data Optical frame: ", points_list)
        pointW = self.w_R_c.dot(points_list[0]) + self.x_c + self.base_offset
        #print("Data World frame: ", pointW)

    def closeGripper(self):
        t = 1.0
        self.controller_manager.gm.move_gripper(10)
        while t < self.controller_manager.gm.gripping_duration + 1.0:
            self.controller_manager.sendReference(self.q)
            t = np.round(t + np.array([conf.robot_params[self.robot_name]['dt']]),  3)  # to avoid issues of dt 0.0009999

    def openGripper(self):
        t = 1.0
        self.controller_manager.gm.move_gripper(100)
        while t < self.controller_manager.gm.gripping_duration + 1.0:
            self.controller_manager.sendReference(self.q)
            t = np.round(t + np.array([conf.robot_params[self.robot_name]['dt']]),  3)  # to avoid issues of dt 0.0009999

    def move(self, Th, rate):
        t = 0
        while t < len(Th):
            self.controller_manager.sendReference(Th[t])
            rate.sleep()
            t += 1
    
    def home(self, rate):
        print(colored("home position", "green"))
        Th = mp.threep2p(self.q, [0.0, 0.3, -0.5], [0,0,0], 0, 3)
        self.move(Th, rate)

    def moveTrajectory(self, xef, phief, tMin, tMax, rate, rot, dir = False):
        q0, q1, q2, q3 ,q4, q5 = self.q
        xe0, phie0 = direct_kinematics(q0, q1, q2, q3 ,q4, q5)
        
        if not rot:     #SPOSTAMENTO
            try:  
                print(colored("  1) raggiungo oggetto...", "yellow"))
                Th = mp.threep2p(self.q, xef, phief, tMin, tMax)
                self.move(Th, rate)
                self.closeGripper()        
                time.sleep(2.)

                print(colored("  2) porto l'oggetto nella sua posizione finale", "yellow"))
                Th = mp.threep2p(self.q, [0.4, 0.2, -0.71], [0,0,0], 0, 3)
                self.move(Th, rate)
                self.openGripper()
            except:
                print("Errore nella traiettoria")
                return
        else :          #ROTAZIONE
            try:
                print(colored("  1) raggiungo oggetto...", "yellow"))
                Th = mp.ruota(self.q, xef, phief, tMin, tMax, dir)
                self.move(Th[0:int(len(Th)/3)] , rate)            #0 -> 199
                self.closeGripper()
                time.sleep(2.)
                print(colored("  2) ruoto l'oggetto...", "yellow"))
                self.move(Th[int(len(Th)/3): int(len(Th)*2/3)] , rate)   #200 -> 399
                self.openGripper()
                time.sleep(1.)
                print(colored("  3) riposiziono l'oggetto ruotato...", "yellow"))
                self.move(Th[int(len(Th)*2/3): int(len(Th))] , rate)   #400 -> 599
                self.closeGripper()
                time.sleep(2.)
                print(colored("  4) porto l'oggetto nella sua posizione finale", "yellow"))
                Th = mp.threep2p(self.q, [0.4, 0.2, -0.71], [0,0,0], 0, 3)
                self.move(Th, rate)
                self.openGripper()
            except:
                print("Errore nella rotazione")
                return            

        self.home(rate)
        self.openGripper()

        #self.ros_pub.publishVisual()
        #self.controller_manager.sendReference(Th[t])
        #self.closeGripper()
        #self.time = 1.0

        # if self.gripper_on == 1:
        #     self.gripper_on = 0
        # else:
        #     self.gripper_on = 1

        #self.controller_manager.gm.move_gripper(10)

        # if not self.obj:
        #     self.closeGripper()
        #     self.obj = True
        # else:
        #     self.obj = False

def startGazebo(p):
    if p.real_robot:
        p.startRealRobot()
    else:
        additional_args = ['gripper:=' + str(p.gripper), 'soft_gripper:='+ str(conf.robot_params[p.robot_name]['soft_gripper'])]#, 'gui:=false']
        p.startSimulator(world_name=p.world_name, use_torque_control=p.use_torque_control, additional_args =additional_args)

    # specify xacro location
    xacro_path = rospkg.RosPack().get_path('ur_description') + '/urdf/' + p.robot_name + '.urdf.xacro'
    p.loadModelAndPublishers(xacro_path)
    p.initVars()
    p.startupProcedure()

    # sleep to avoid that the real robot crashes on the table
    time.sleep(3.)

def talker(p):  

    # loop frequency
    rate = ros.Rate(1 / conf.robot_params[p.robot_name]['dt'])

    time.sleep(3.)
    cicciogamer89 = ros.wait_for_message("/zedObjects", Float32MultiArray)
    pos = np.array(cicciogamer89.data)

    coordinates = ["x", "y", "z"]

    # print ('Insert position of x,y,z: ')
    # for i in range(3):
    #     ok = False
    #     while not ok:
    #         n = input("value " + coordinates[i] + ": ")
    #         try:
    #             pos.append(float(n))
    #             ok = True
    #         except:
    #             print(colored("Input not float", "red"))

    if pos != []:
        pos[2] = -0.73
        print ('Position: ', pos)
        pos_final = np.array(pos)
    
        frameInizialeZ = int(input('Frame iniziale [z]: '))
        # frameInizialeZ = 180
        if frameInizialeZ == 0:
            frameInizialeZ = 0
        elif frameInizialeZ == 90:
            frameInizialeZ = np.pi/2
        elif frameInizialeZ == 180:
            frameInizialeZ = np.pi
        elif frameInizialeZ == -90:
            frameInizialeZ = -np.pi/2
    
        rot = int(input('Rotation [0:NO/1:SI]? '))
        #rot = 0
        if rot:
            dir = np.array([np.pi/2, 0, -np.pi/2])
        else:
            dir = np.array([0,0,0])

        #[2.3705259724671266, 0.8350330025362989, 0.25114659099651027, -2.656975920327706, 1.5707963267948966, -0.79972964567223]
        #get the angles to apply if it returns False it means that it can not reach that position 
        if(check_point(pos_final, p.q) != False):
            print(colored("Posizione raggiungibile dal robot!! ","green"))
            dt = 0.01
            p.moveTrajectory(pos_final, [frameInizialeZ, 0, 0], 0, 3, rate, rot, dir)
        else:
            print(colored("Posizione impossibile da raggiungere", 'red'))
    
    p.updateKinematicsDynamics()

    ## set joints here
    # p.q_des = p.q_des_q0  + 0.1 * np.sin(2*np.pi*0.5*p.time)
    # p.qd_des = 0.1 * 2 * np.pi * 0.5* np.cos(2 * np.pi * 0.5 * p.time)*np.ones(p.robot.na)

    #need to uncomment this to be able to send joints references (leave it commented if you have an external node setting them)
    #p.controller_manager.sendReference(p.q_des, p.qd_des, p.h)

    if p.real_robot:
        p.ros_pub.add_arrow(p.x_ee + p.base_offset, p.contactForceW / (6 * p.robot.robot_mass), "green")

    # log variables
    p.logData()
    # plot end-effector
    p.ros_pub.add_marker(p.x_ee + p.base_offset)
    p.ros_pub.publishVisual()

    #wait for synconization of the control loop
    rate.sleep()
    #p.time = np.round(p.time + np.array([conf.robot_params[p.robot_name]['dt']]),  3)  # to avoid issues of dt 0.0009999
        

if __name__ == '__main__':

    p = Ur5Generic(robotName)
    p.start()
    startGazebo(p)
    
    rate = ros.Rate(1 / conf.robot_params[p.robot_name]['dt'])
    p.home(rate)
    p.openGripper()
    
    try:
        while(1):
            talker(p)
    except (ros.ROSInterruptException, ros.service.ServiceException):
        ros.signal_shutdown("killed")
        p.deregister_node()
        if   conf.plotting:
            p.plotStuff()

    
        