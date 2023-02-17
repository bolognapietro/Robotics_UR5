//
// Created by utente on 19/12/2022.
//

#ifndef PROJECTROBOTICA_KINEMATICS_H
#define PROJECTROBOTICA_KINEMATICS_H


#include "Eigen/Dense"
#include "Eigen/LU"
#include <iostream>
#include <cmath>
#include <complex>

using namespace Eigen;

typedef Matrix<double,Dynamic,6> Matrix6d;
typedef Matrix<double, 1, 6> RowVector6d;

Matrix4d T10f (double th1);
Matrix4d T21f (double th2);
Matrix4d T32f (double th3);
Matrix4d T43f (double th4);
Matrix4d T54f (double th5);
Matrix4d T65f (double th6);
void direct_kinematics(RowVector6d Th, Matrix3d& R06, Vector3d& ef);
double hypot(double x1, double x2);
double arccos(std::complex<double> theta);
double arcsin(std::complex<double> theta);
void inverse_kinematics(Vector3d p60, Matrix6d& Th, Matrix3d R60 = Matrix3d {{1,0,0},{0,1,0},{0,0,1}});
bool check_collision(RowVector6d Th);
bool check_angles(RowVector6d Th);
double norma(RowVector6d Th0, RowVector6d th);
void removeRow(Matrix6d& matrix, unsigned int rowToRemove);
bool bestInverse(RowVector6d Th0, Matrix6d& all);
bool check_point(Vector3d pos, RowVector6d q0);


#endif //PROJECTROBOTICA_KINEMATICS_H