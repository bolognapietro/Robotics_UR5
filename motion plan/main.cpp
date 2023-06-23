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

using namespace std;

typedef Eigen::Matrix<double,Eigen::Dynamic,6> Matrix6d;
typedef Eigen::Matrix<double,Eigen::Dynamic,8> Matrix8d;
typedef Matrix<double, 1, 6> RowVector6d;
typedef Matrix<double, 1, 8> RowVector8d;

bool real_robot = false;

RowVector8d getJointState();
RowVector6d getJointBraccio();
Eigen::RowVector2d getJointGripper();
void publish(Matrix8d& Th);

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

    /*
    while(1){
        RowVector8d test = getJointState();
        for(int i=0; i<8; i++){
            ROS_INFO("%f ", test(i));
        }
        cout << endl;
    }
    */

    // START POSITION
    {
        Matrix8d Th;
        bool cond[3] = {true, true, true};
        RowVector6d jointBraccio = getJointBraccio();

        cout << "Moving to START position " << endl;
        RowVector3d posInit {{0.4, 0.2, -0.5}};
        
        cond[0] = p2pMotionPlan(jointBraccio, posInit, phiEf, 0, maxT, Th);
        closeGripper(Th);
        jointBraccio = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
        cond[1] = p2pMotionPlan(jointBraccio, posHome, phiEf, 0, maxT, Th);
        openGripper(Th);
        start = 1;

        if (cond[0] and cond[1] and cond[2]) {  // MOVEMENT OK
            //PUBLISH
            publish(Th);
        }
    }


    while(ros::ok()) {
        boost::shared_ptr<geometry_msgs::Pose const> sharedMsg;

        // Position waited from zed
/*
        cout << "Waiting for messages " << endl;
        sharedMsg = ros::topic::waitForMessage<geometry_msgs::Pose>("/objects_info");
        if (sharedMsg != NULL) {
            pos(0) = sharedMsg->position.x - 0.5;
            pos(1) = sharedMsg->position.y - 0.35;
            //pos(2) = sharedMsg->position.z;
        }

        pos(2) = -0.73;
        ROS_INFO("[%f, %f, %f]\n", pos(0), pos(1), pos(2));
*/

        // To manually insert the coordinates

        cout << "pos[x]: ";
        cin >> pos(0);
        cout << "pos[y]: ";
        cin >> pos(1);
        cout << "pos[z]: ";
        cin >> pos(2);

        // offset manual 
        /*
        pos(0) = pos(0) - 0.5;
        pos(1) = pos(1) -0.35;
        */

        bool rot;
        double frameInizialeZ = 0;
        int type_rot;

        //double frameInizialeZ;
        //frameInizialeZ = sharedMsg->orientation.w;
        //cout << "Frame iniziale [z]: " << frameInizialeZ << endl;

        cout << "Rotazione? [0 = no, 1 = si] ";
        cin >> rot;
        
        if (rot){
            cout << "Tipologia rotazione: ";
            cin >> type_rot;  // 0-10 , 0 -> no_rot , 1-4 -> elementar , 5-10 -> composte
        } else {
            frameInizialeZ = 0;
        }

        phiEf(0) = frameInizialeZ;

        RowVector6d jointBraccio = getJointBraccio();

        Matrix8d Th;
        bool cond[3] = {true, true, true};
        if (check_point(pos, jointBraccio)) {
            ROS_INFO("Posizione raggiungibile dal braccio!\n");

            if (!rot) {
                cond[0] = threep2p(jointBraccio, pos, phiEf, 0, maxT, Th);
                // CLOSE GRIPPER
                closeGripper(Th);
                jointBraccio = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[1] = threep2p(jointBraccio, posDrop, phiZero, 0, maxT, Th);
                // OPEN GRIPPER
                openGripper(Th);
                // TORNO ALLA HOME
                jointBraccio = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[2] = threep2p(jointBraccio, posHome, phiZero, 0, maxT, Th);
            } else {
                cout << phiEf << endl;
                cond[0] = ruota(jointBraccio, pos, type_rot, 0, maxT, Th);
                // ROTAZIONI NELLA RUOTA
                jointBraccio = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[1] = threep2p(jointBraccio, posDrop, phiZero, 0, maxT, Th);
                // OPEN GRIPPER
                openGripper(Th);
                // TORNO ALLA HOME
                jointBraccio = Th.row(Th.rows() - 1).block<1, 6>(0, 0);
                cond[2] = threep2p(jointBraccio, posHome, phiZero, 0, maxT, Th);
            }

            //PUBLISH
            if (cond[0] and cond[1] and cond[2]) {  //MOVIMENTO OK
                //PUBLISH
                publish(Th);
            } else {
                ROS_INFO("Errore nella traiettoria\n");
                //cout << "Errore nella traiettoria" << endl;
            }

        } else {
            ROS_INFO("Posizione IMPOSSIBILE da raggiungere!\n");
            //cout << "Posizione IMPOSSIBILE da raggiungere!" << endl;
        }
    }

    return 0;
}


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

RowVector6d getJointBraccio(){
    RowVector8d joints = getJointState();
    RowVector6d jointBraccio {{joints(4), joints(3), joints(0), joints(5), joints(6), joints(7)}};
    return jointBraccio;
}

Eigen::RowVector2d getJointGripper(){
    RowVector8d joints = getJointState();
    RowVector2d jointGripper {{joints(1), joints(2)}};
    return jointGripper;
}

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
