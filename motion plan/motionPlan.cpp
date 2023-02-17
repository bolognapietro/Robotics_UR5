//
// Created by utente on 19/12/2022.
//
#include "Eigen/Dense"
#include "kinematics.h"
//#include "motionPlan.h"
#include <iostream>
#include <cmath>
#include <cstring>

using namespace Eigen;

typedef Matrix<double,Dynamic,6> Matrix6d;
typedef Matrix<double, 1, 6> RowVector6d;
typedef Matrix<double, 1, 8> RowVector8d;
typedef Matrix<double, Dynamic, 8> Matrix8d;

Matrix3d eul2rotm(Vector3d angles){
    double phi = angles(0);
    double theta = angles(1);
    double gamma = angles(2);

    Matrix3d R {
            {cos(phi)*cos(theta), cos(phi)*sin(theta)*sin(gamma) - sin(phi)*cos(gamma), cos(phi)*sin(theta)*cos(gamma) + sin(phi)*sin(gamma)},
            {sin(phi)*cos(theta), sin(phi)*sin(theta)*sin(gamma) + cos(phi)*cos(gamma), sin(phi)*sin(theta)*cos(gamma) - cos(phi)*sin(gamma)},
            {-sin(theta), cos(theta)*sin(gamma), cos(theta)*cos(gamma)}
    };

    return R;
}

Vector3d rotm2eul(Matrix3d R){
    double x = atan2(R(1,0),R(0,0));
    double y = atan2(-R(2,0), sqrt(pow(R(2,1), 2) + pow(R(2,2), 2)));
    double z = atan2(R(2,1), R(2,2));
    Vector3d vet {{x, y, z}};
    return vet;
}

void closeGripper(Matrix8d& Th){
    RowVector8d th = Th.row(Th.rows()-1);
    double start = -0.3;
    for (int i=0;i<60;i++){
        th(6) = start+0.01;
        th(7) = start+0.01;
        Th.conservativeResize(Th.rows()+1, Eigen::NoChange );
        Th.row(Th.rows()-1) = th;
    }
}

void openGripper(Matrix8d& Th){
    RowVector8d th = Th.row(Th.rows()-1);
    double start = 0.3;
    for (int i=0;i<60;i++){
        th(6) = start-0.01;
        th(7) = start-0.01;
        Th.conservativeResize(Th.rows()+1, Eigen::NoChange );
        Th.row(Th.rows()-1) = th;
    }
}

bool p2pMotionPlan(RowVector6d qEs, Vector3d xEf, Vector3d phiEf, double minT, double maxT, Matrix8d& Th, bool slow=false){
    //double grip = Th.row(Th.rows()-1)(6);


    double dt = 0.01; //come prendere da params??
    if (slow){
        dt = dt/2;
    }
    Matrix6d qEf;
    inverse_kinematics(xEf, qEf, eul2rotm(phiEf));
    bool res = bestInverse(qEs, qEf);
    if (not res)
        return false;

    for (int j=0;j<qEf.rows();j++){
        bool error = false;
        Matrix<double,Dynamic,4> A;
        for (int i=0;i<qEs.size();i++){
            Matrix4d M {
                    {1, minT, pow(minT,2), pow(minT,3)},
                    {0,1, 2*minT, 3*pow(minT,2)},
                    {1, maxT, pow(maxT,2), pow(maxT,3)},
                    {0,1, 2*maxT, 3*pow(maxT,2)}
            };
            Vector4d b {{qEs(i), 0, qEf(j,i), 0}};
            Vector4d a = M.inverse()*b;

            A.conservativeResize(A.rows()+1, Eigen::NoChange );
            A.row(A.rows()-1) = a.transpose();
        }
        double t = minT;
        while (t+0.000001 < maxT){
            RowVector6d th;
            for (int i=0;i < qEs.size(); i++){
                double q = A(i,0) + A(i,1)*t + A(i,2)*t*t + A(i,3)*t*t*t;
                th(i) = q;
            }

            if (not check_angles(th)){
                std::cout << "ERRORE TRAIETTORIA " << t << std::endl;
                error = true;
                break;
            }
            RowVector8d th8;
            th8 << th, 0, 0;   //FORSE NON VA

            Th.conservativeResize(Th.rows()+1, Eigen::NoChange );
            Th.row(Th.rows()-1) = th8;

            t += dt;
        }
        if (not error){
            return true;
        }
    }
    return false;
}

bool threep2p(RowVector6d qEs, Vector3d xEf, Vector3d phiEf, double minT, double maxT, Matrix8d& Th){

    int moveT = (int)((maxT-minT)/3);
    RowVector3d t0 {{minT, minT+moveT, minT+2*moveT}};
    RowVector3d tf {{maxT-2*moveT, maxT-moveT, maxT}};

    //FIRST MOVE
    Vector3d xE1;
    Matrix3d phi;
    direct_kinematics(qEs, phi, xE1);
    xE1(1) = -xE1(1);
    xE1(2) = -0.5;
    RowVector3d phiE1 {{0,0,0}};

    bool res = p2pMotionPlan(qEs, xE1, phiE1, t0(0), tf(0), Th);
    if (not res){
        return false;
    }


    //std::cout << Th << std::endl;

    //SECOND MOVE
    qEs = Th.row(Th.rows()-1).block<1,6>(0,0);
    RowVector3d xE2 = xEf;
    xE2(2) = -0.5;
    double sogliaX = 0.3;
    bool slow = false;
    if (abs(xE1(0) - xE2(0)) > sogliaX)
        slow = true;

    res = p2pMotionPlan(qEs, xE2, phiEf, t0(1), tf(1), Th, slow);
    if (not res){
        return false;
    }

    //THIRD MOVE
    qEs = Th.row(Th.rows()-1).block<1,6>(0,0);
    res = p2pMotionPlan(qEs, xEf, phiEf, t0(2), tf(2), Th);
    if (not res){
        return false;
    }

    return true;
}

bool ruota(RowVector6d qEs, Vector3d xEf, Vector3d phiEf, double minT, double maxT, Matrix8d& Th){
    Vector3d xe1 = xEf;
    Eigen::RowVector3d dir {{M_PI_2, 0, -M_PI_2}};

    bool res = threep2p(qEs, xEf, phiEf, minT, maxT, Th); //da dove sono all'oggetto
    if (not res)
        return false;
    // CLOSE GRIPPER
    //closeGripper(Th);

    qEs = Th.row(Th.rows()-1).block<1,6>(0,0);

    xe1(1) = xe1(1) + 0.123;
    xe1(2) = xe1(2) - 0.1;

    res = threep2p(qEs, xe1, dir, minT, maxT, Th);  //gira oggetto e lo rimette lì
    if (not res)
        return false;
    // OPEN GRIPPER
    //openGripper(Th);

    qEs = Th.row(Th.rows()-1).block<1,6>(0,0);
    phiEf << 0,0,0;

    res = threep2p(qEs, xEf, phiEf, minT, maxT, Th);  //prende l'oggetto da sopra
    if (not res)
        return false;
    // CLOSE GRIPPER
    //closeGripper(Th);

    return true;
}