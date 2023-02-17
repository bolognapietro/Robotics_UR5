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


using namespace std;

typedef Eigen::Matrix<double,Eigen::Dynamic,6> Matrix6d;
typedef Eigen::Matrix<double,Eigen::Dynamic,8> Matrix8d;
typedef Matrix<double, 1, 6> RowVector6d;
typedef Matrix<double, 1, 8> RowVector8d;

bool real_robot = true;

RowVector8d getJointState();
RowVector6d getJointBraccio();
Eigen::RowVector2d getJointGripper();
void publish(Matrix8d& Th);

int main(int argc, char **argv) {
    ros::init(argc, argv, "motion_plan");
    ros::NodeHandle n;

    RowVector3d posStandard {{0.4, 0.2, -0.6}};
    RowVector3d phiZero {{0, 0, 0}};

    /*
    while(1){
        RowVector8d test = getJointState();
        for(int i=0; i<8; i++){
            ROS_INFO("%f ", test(i));
        }
        cout << endl;
    }
    */


    while(ros::ok()) {

        Eigen::RowVector3d pos;
        cout << "input pos_final: x= ";
        cin >> pos(0);

        if(pos(0) ==  99) {

            boost::shared_ptr<std_msgs::Float32MultiArray const> sharedMsg;
            cout << "Waiting for messages ";
            sharedMsg = ros::topic::waitForMessage<std_msgs::Float32MultiArray>("/prova");

            //Eigen::RowVector3d pos;
            if (sharedMsg != NULL) {
                std_msgs::Float32MultiArray msg = *sharedMsg;
                for (int i = 0; i < 3; i++) {
                    pos(i) = msg.data[i];
                }
            }
            pos(2) = -0.6;
            ROS_INFO("[%f, %f, %f]\n", pos(0), pos(1), pos(2));
        }
        else
        {
            cout << " y= ";
            cin >> pos(1);
            cout << " z= ";
            cin >> pos(2);
        }

        //cout << pos << endl;

        int fZ;
        double frameInizialeZ;
        cout << "Frame iniziale [z]: ";
        cin >> fZ;  // 90 - 180 - -90
        switch (fZ) {
            case 90:
                frameInizialeZ = M_PI_2;
                break;
            case 180:
                frameInizialeZ = M_PI;
                break;
            case -90:
                frameInizialeZ = -M_PI_2;
                break;
            default:
                frameInizialeZ = 0;
                break;
        }
        bool rot;
        cout << "Rotazione? [0 = no, 1 = si]";
        cin >> rot;
        Eigen::RowVector3d phiEf{{frameInizialeZ, 0, 0}};

        RowVector6d jointBraccio = getJointBraccio();

        Matrix8d Th;
        bool cond[2] = {true, true};
        if (check_point(pos, jointBraccio)) {
            ROS_INFO("Posizione raggiungibile dal braccio!\n");
            //cout << "Posizione raggiungibile dal braccio!" << endl;
            // moveTrajectory(pos, [frameIniziale, 0, 0], 0, 3, rate, rot, dir)
            if (!rot) {
                cond[0] = threep2p(jointBraccio, pos, phiEf, 0, 3, Th);
                // CLOSE GRIPPER
                closeGripper(Th);
                jointBraccio = Th.row(Th.rows()-1).block<1, 6>(0, 0);
                cond[1] = threep2p(jointBraccio, posStandard, phiZero, 0, 3, Th);
                //OPEN GRIPPER
                openGripper(Th);
            } else {
                cond[0] = ruota(jointBraccio, pos, phiEf, 0, 3, Th);
                // ROTAZIONI NELLA RUOTA
                jointBraccio = Th.row(Th.rows()-1).block<1, 6>(0, 0);
                cond[1] = threep2p(jointBraccio, posStandard, phiZero, 0, 3, Th);
                // OPEN GRIPPER
                //openGripper(Th);
            }

            //PUBLISH
            if (cond[0] and cond[1]) {  //MOVIMENTO OK
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

    ros::Rate loop_rate(50);

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