#include "Eigen/Dense"
#include <iostream>
#include "kinematics.h"
#include "motionPlan.h"
#include "cmath"

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/Float64MultiArray.h"
#include "sensor_msgs/JointState.h"
#include "boost/shared_ptr.hpp"
#include "geometry_msgs/Pose.h"
#include "vision_pkg/legoMessage.h"

using namespace std;

typedef Eigen::Matrix<double,Eigen::Dynamic,6> Matrix6d;
typedef Eigen::Matrix<double,Eigen::Dynamic,8> Matrix8d;
typedef Matrix<double, 1, 6> RowVector6d;
typedef Matrix<double, 1, 8> RowVector8d;

bool real_robot = false;

RowVector8d getJointState();
RowVector6d getJointArm();
Eigen::RowVector2d getJointGripper();
void publish(Matrix8d& Th);
RowVector3d findPosDrop(int type_block);

/**
 * @brief manages the motion plan node
 */

 int main(int argc, char **argv) {
    ros::init(argc, argv, "motion_plan");
    ros::NodeHandle n;

    RowVector3d posHome {{- 0.2, 0.2, -0.50}};
    RowVector3d posDrop {{0.4, 0.2, -0.72}};
    RowVector3d phiZero {{0, 0, 0}};

    int start = 0;
    Eigen::RowVector3d pos;
    Eigen::RowVector3d phiEf{{0, 0, 0}};
    int maxT = 3;       // movement time ( 3 -> fast, 6 -> slow)

    // START POSITION
    {
        Matrix8d Th;
        bool cond[3] = {true, true, true};
        RowVector6d jointArm = getJointArm();

        cout << "Moving to START position " << endl;
        RowVector3d posInit {{0.4, 0.2, -0.5}};
        
        cond[0] = p2pMotionPlan(jointArm, posInit, phiEf, 0, maxT, Th);
        closeGripper(Th);
        jointArm = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
        cond[1] = p2pMotionPlan(jointArm, posHome, phiEf, 0, maxT, Th);
        openGripper(Th);
        start = 1;

        if (cond[0] and cond[1]) {  // MOVEMENT OK
            //PUBLISH
            publish(Th);
        }
    }


    while(ros::ok()) {

        // Position waited from zed

        ros::NodeHandle node_1;
        vision_pkg::legoMessage::ConstPtr sharedMsg = ros::topic::waitForMessage<vision_pkg::legoMessage>("/objects_info",node_1 );


        cout << "Waiting for messages " << endl;
        if (sharedMsg != NULL) {
            pos(0) = sharedMsg->pose.position.x - 0.5;
            pos(1) = sharedMsg->pose.position.y - 0.35;
            pos(2) = sharedMsg->pose.position.z - 1.61 -0.04;
        }

        pos(2) = -0.73;
        ROS_INFO("[%f, %f, %f]\n", pos(0), pos(1), pos(2));


        // To manually insert the coordinates

/*
        cout << "pos[x]: ";
        cin >> pos(0);
        cout << "pos[y]: ";
        cin >> pos(1);
        cout << "pos[z]: ";
        cin >> pos(2);

        bool rot;
        double frameInizialeZ = 0;
        int type_rot;
        int type_block;

        cout << "Pos drop [0-8]: ";
        cin >> type_block;
*/      
        bool rot = false;
        int type_rot = 0;

        // ZED

        double frameInizialeZ;
        frameInizialeZ = sharedMsg->pose.orientation.w;
        cout << "Frame iniziale [z]: " << frameInizialeZ << endl;

        // POS DROP
        string type_block_str = sharedMsg->model;
        int type_block = stoi(type_block_str);

        posDrop = findPosDrop(type_block);

        // cout << "Rotation? [0 = no, 1 = si] ";
        // cin >> rot;
        
        // if (rot){
        //     cout << "Tipologia rotazione: ";
        //     cin >> type_rot;  // 0-10 , 0 -> no_rot , 1-4 -> elementar , 5-10 -> composite
        // } else {
        //     frameInizialeZ = 0;
        // }

        phiEf(0) = frameInizialeZ;

        RowVector6d jointArm = getJointArm();

        Matrix8d Th;
        bool cond[3] = {true, true, true};
        if (check_point(pos, jointArm)) {
            ROS_INFO("Posizione raggiungibile dal Arm!\n");

            if (!rot) {
                cond[0] = threep2p(jointArm, pos, phiEf, 0, maxT, Th);
                // CLOSE GRIPPER
                closeGripper(Th);
                jointArm = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[1] = threep2p(jointArm, posDrop, phiZero, 0, maxT, Th);
                // OPEN GRIPPER
                openGripper(Th);
                // RETURN HOME
                jointArm = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[2] = threep2p(jointArm, posHome, phiZero, 0, maxT, Th);
            } else {
                cout << phiEf << endl;
                cond[0] = rotate(jointArm, pos, type_rot, 0, maxT, Th);
                // ROTATIONS
                jointArm = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[1] = threep2p(jointArm, posDrop, phiZero, 0, maxT, Th);
                // OPEN GRIPPER
                openGripper(Th);
                // RETURN HOME
                jointArm = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[2] = threep2p(jointArm, posHome, phiZero, 0, maxT, Th);
            }

            //PUBLISH
            if (cond[0] and cond[1] and cond[2]) {  //MOVE OK
                publish(Th);
            } else {
                ROS_INFO("Trajectory error\n");
            }

        } else {
            ROS_INFO("Position IMPOSSIBLE to reach!\n");
        }
    }

    return 0;
}

/**
 * @brief get the joint state published by the ur5 generic node on the topic
 * 
 * @return RowVector8d
 */

RowVector8d getJointState(){
    boost::shared_ptr<sensor_msgs::JointState const> sharedMsg;
    sharedMsg = ros::topic::waitForMessage<sensor_msgs::JointState>("/ur5/joint_states");

    RowVector8d joints;
    if (sharedMsg != NULL){
        sensor_msgs::JointState msg = *sharedMsg;
        for (int i=0;i<8;i++){
            joints(i) = msg.position[i];
        }
    }

    return joints;
}

/**
 * @brief get the joint state of the arm
 * 
 * @return RowVector6d
 */

RowVector6d getJointArm(){
    RowVector8d joints = getJointState();
    RowVector6d jointArm {{joints(4), joints(3), joints(0), joints(5), joints(6), joints(7)}};
    return jointArm;
}

/**
 * @brief get the joint state of the gripper
 * 
 * @return RowVector2d
 */

Eigen::RowVector2d getJointGripper(){
    RowVector8d joints = getJointState();
    RowVector2d jointGripper {{joints(1), joints(2)}};
    return jointGripper;
}

/**
 * @brief publishes the computed joint configurations on the topic
 * 
 */

void publish(Matrix8d& Th){
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<std_msgs::Float64MultiArray>("/ur5/joint_group_pos_controller/command", 10);

    ros::Rate loop_rate(125);

    Matrix6d realTh;

    if (real_robot){
        for (int i=0;i < realTh.rows(); i++){
            std_msgs::Float64MultiArray msg;
            msg.data.clear();
            RowVector8d th = Th.row(i);
            for(int j=0; j<6; j++){
                msg.data.push_back(th(j));
            }
            pub.publish(msg);
            loop_rate.sleep();
        }
    } else {
        for (int i = 0; i < Th.rows(); i++) {
            std_msgs::Float64MultiArray msg;
            msg.data.clear();
            RowVector8d th = Th.row(i);
            for(int j=0; j<8; j++){
                msg.data.push_back(th(j));
            }
            pub.publish(msg);
            loop_rate.sleep();
        }
    }
}

/**
 * @brief finds the drop position from the index of the block
 * 
 * @return RowVector3d
 */

RowVector3d findPosDrop(int type_block){
    RowVector3d posDrop;

    switch(type_block){
        case 0:
            posDrop(0) = 0.3;
            posDrop(1) = 0.35;
            break;
        case 1:
            posDrop(0) = 0.37;
            posDrop(1) = 0.35;
            break;
        case 2:
            posDrop(0) = 0.45;
            posDrop(1) = 0.35;
            break;
        case 3:
            posDrop(0) = 0.3;
            posDrop(1) = 0.15;
            break;
        case 4:
            posDrop(0) = 0.37;
            posDrop(1) = 0.15;
            break;
        case 5:
            posDrop(0) = 0.45;
            posDrop(1) = 0.15;
            break;
        case 6:
            posDrop(0) = 0.3;
            posDrop(1) = -0.05;
            break;
        case 7:
            posDrop(0) = 0.37;
            posDrop(1) = -0.05;
            break;
        case 8:
            posDrop(0) = 0.45;
            posDrop(1) = -0.05;
            break;

        default:
            posDrop(0) = 0.3;
            posDrop(1) = 0.35;
    }
    posDrop(2) = -0.72;
    return posDrop;
}
