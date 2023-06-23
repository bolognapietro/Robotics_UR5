//
// Created by utente on 19/12/2022.
//

#ifndef PROJECTROBOTICA_MOTIONPLAN_H
#define PROJECTROBOTICA_MOTIONPLAN_H

#include "Eigen/Dense"
#include <iostream>

using namespace Eigen;

typedef Matrix<double,Dynamic,6> Matrix6d;
typedef Matrix<double, 1, 6> RowVector6d;
typedef Matrix<double, Dynamic, 8> Matrix8d;

Matrix3d eul2rotm(Vector3d angles);
Vector3d rotm2eul(Matrix3d R);
bool p2pMotionPlan(RowVector6d qEs, Vector3d xEf, Vector3d phiEf, double minT, double maxT, Matrix8d& Th, bool slow=false);
bool threep2p(RowVector6d qEs, Vector3d xEf, Vector3d phiEf, double minT, double maxT, Matrix8d& Th);
bool ruota(RowVector6d qEs, Vector3d xEf, int type_rot, double minT, double maxT, Matrix8d& Th, bool ricorsiva = false);
void openGripper(Matrix8d& Th);
void closeGripper(Matrix8d& Th);

#endif //PROJECTROBOTICA_MOTIONPLAN_H