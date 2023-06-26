#include "Eigen/Dense"
#include "Eigen/LU"
#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>

#include "ros/ros.h"

using namespace Eigen;

typedef Matrix<double,Dynamic,6> Matrix6d;
typedef Matrix<double, 1, 6> RowVector6d;

RowVectorXd A{{0.0, -0.425, -0.3922, 0.0, 0.0, 0.0}};        
RowVectorXd D{{0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996}};   

/**
 * @brief Give th1, returns the transformation matrix from frame 0 to frame 1
 * 
 * @param th1 
 * @return ** Matrix4d
 */

Matrix4d T10f (double th1){
    Matrix4d T10f {
            {cos(th1) , -sin(th1) , 0 , 0},
            {sin(th1) , cos(th1) , 0 , 0},
            {0 , 0 , 1 , D[0]},
            {0 , 0 , 0 , 1}
    };
    return T10f;
}

/**
 * @brief Give th1, returns the transformation matrix from frame 1 to frame 2
 * 
 * @param th1 
 * @return ** Matrix4d
 */

Matrix4d T21f (double th2){
    Matrix4d T21f {
            {cos(th2) , -sin(th2) , 0 , 0},
            {0 , 0 , -1 , 0},
            {sin(th2) , cos(th2) , 0 , 0},
            {0 , 0 , 0 , 1}
    };
    return T21f;
}

/**
 * @brief Give th2, returns the transformation matrix from frame 2 to frame 3
 * 
 * @param th2
 * @return ** Matrix4d
 */

Matrix4d T32f (double th3){
    Matrix4d T32f {
            {cos(th3) , -sin(th3) , 0 , A[1]},
            {sin(th3) , cos(th3) , 0 , 0},
            {0 , 0 , 1 , D[2]},
            {0 , 0 , 0 , 1}
    };
    return T32f;
}

/**
 * @brief Give th3, returns the transformation matrix from frame 3 to frame 4
 * 
 * @param th3 
 * @return ** Matrix4d
 */

Matrix4d T43f (double th4){
    Matrix4d T43f {
            {cos(th4) , -sin(th4) , 0 , A[2]},
            {sin(th4) , cos(th4) , 0 , 0},
            {0 , 0 , 1 , D[3]},
            {0 , 0 , 0 , 1}
    };
    return T43f;
}

/**
 * @brief Give th4, returns the transformation matrix from frame 4 to frame 5
 * 
 * @param th4
 * @return ** Matrix4d
 */

Matrix4d T54f (double th5){
    Matrix4d T54f {
            {cos(th5) , -sin(th5) , 0 , 0},
            {0 , 0 , -1 , -D[4]},
            {sin(th5) , cos(th5) , 0 , 0},
            {0 , 0 , 0 , 1}
    };
    return T54f;
}

/**
 * @brief Give th5, returns the transformation matrix from frame 5 to frame 6
 * 
 * @param th5
 * @return ** Matrix4d
 */

Matrix4d T65f (double th6){
    Matrix4d T65f {
            {cos(th6) , -sin(th6) , 0 , 0},
            {0 , 0 , 1 , D[5]},
            {-sin(th6) , -cos(th6) , 0 , 0},
            {0 , 0 , 0 , 1}
    };
    return T65f;
}

/**
 * @brief Transform from the joint angles space to the operational space, which is specified in terms of position and orientation of the end-effector
 * 
 * @param[in] Th
 * @param[in] R06
 * @param[in] ef
 */

void direct_kinematics(RowVector6d Th, Matrix3d& R06, Vector3d& ef){

    double th1 = Th[0];
    double th2 = Th[1];
    double th3 = Th[2];
    double th4 = Th[3];
    double th5 = Th[4];
    double th6 = Th[5];

    Matrix4d T06 = T10f(th1)*T21f(th2)*T32f(th3)*T43f(th4)*T54f(th5)*T65f(th6);

    ef = T06.block<3,1>(0,3);
    R06 = T06.block<3,3>(0,0);
}


/**
 * @brief return the hypotenuse of two sides
 * 
 * @param[in] x1
 * @param[in] x2 
 * @return double
 */

double hypot(double x1, double x2){
    double hy = sqrt(x1*x1 + x2*x2);
    return hy;
}

/**
 * @brief return the real part of the asin of an angle
 * 
 * @param[in] theta
 * @return double
 */

double arccos(std::complex<double> theta){
    return acos(theta).real();
}

/**
 * @brief return the real part of the sin of an angle
 * 
 * @param[in] theta
 * @return double
 */

double arcsin(std::complex<double> theta){
    return asin(theta).real();
}

/**
 * @brief Transform from the operational space to the joint angles space
 * 
 * @param[in] p60 position you want to bring in the joint space
 * @param[in] Th Matrix6d passed as reference to save the computed joints
 * @param[in] R60 Matrix3d passed as reference to save the orientation matrix
 */

void inverse_kinematics(Vector3d p60, Matrix6d& Th, Matrix3d R60 = Matrix3d {{1,0,0},{0,1,0},{0,0,1}}){


    p60[1] = -p60[1];
    p60[2] = -p60[2];

    Matrix4d T60 {
            {R60(0,0) , R60(0,1) , R60(0,2) , p60[0]},
            {R60(1,0) , R60(1,1) , R60(1,2) , p60[1]},
            {R60(2,0) , R60(2,1) , R60(2,2) , p60[2]},
            {0 , 0 , 0 , 1}
    };

    //Finding th1
    Vector4d a {{0,0,-D[5],1}};
    Vector4d p50 = T60*a;
    float th1_1 = (atan2(p50[1], p50[0]) + arccos(D[3]/hypot(p50[1], p50[0]))) + M_PI_2;
    float th1_2 = (atan2(p50[1], p50[0]) - arccos(D[3]/hypot(p50[1], p50[0]))) + M_PI_2;

    //Finding th5
    float th5_1 = +(arccos((p60[0]*sin(th1_1) - p60[1]*cos(th1_1) - D[3]) / D[5]));
    float th5_2 = -(arccos((p60[0]*sin(th1_1) - p60[1]*cos(th1_1) - D[3]) / D[5]));
    float th5_3 = +(arccos((p60[0]*sin(th1_2) - p60[1]*cos(th1_2) - D[3]) / D[5]));
    float th5_4 = -(arccos((p60[0]*sin(th1_2) - p60[1]*cos(th1_2) - D[3]) / D[5]));

    //Related to th11 a th51
    Matrix4d T06 = T60.inverse();
    Vector3d Xhat = T06.block<3,1>(0,0);  
    Vector3d Yhat = T06.block<3,1>(0,1);   //[:,1][0:3]

    float th6_1 = (atan2(((-Xhat[1]*sin(th1_1) + Yhat[1]*cos(th1_1))) / sin(th5_1), ((Xhat[0]*sin(th1_1) - Yhat[0]*cos(th1_1)))/sin(th5_1)));
    //related to th11 a th52
    float th6_2 = (atan2(((-Xhat[1]*sin(th1_1) + Yhat[1]*cos(th1_1)) / sin(th5_2)), ((Xhat[0]*sin(th1_1) - Yhat[0]*cos(th1_1))/sin(th5_2))));
    //related to th12 a th53
    float th6_3 = (atan2(((-Xhat[1]*sin(th1_2) + Yhat[1]*cos(th1_2)) / sin(th5_3)), ((Xhat[0]*sin(th1_2) - Yhat[0]*cos(th1_2))/sin(th5_3))));
    //related to th12 a th54
    float th6_4 = (atan2(((-Xhat[1]*sin(th1_2) + Yhat[1]*cos(th1_2)) / sin(th5_4)), ((Xhat[0]*sin(th1_2) - Yhat[0]*cos(th1_2))/sin(th5_4))));

    //One
    Matrix4d T41m = T10f(th1_1).inverse() * T60 * T65f(th6_1).inverse() * T54f(th5_1).inverse();
    Vector3d p41_1 = T41m.block<3,1>(0,3);
    double p41xz_1 = hypot(p41_1[0], p41_1[2]);

    //Two
    T41m = T10f(th1_1).inverse()*T60*T65f(th6_2).inverse()*T54f(th5_2).inverse();
    Vector3d p41_2 = T41m.block<3,1>(0,3);
    double p41xz_2 = hypot(p41_2[0], p41_2[2]);

    //Three
    T41m = T10f(th1_2).inverse()*T60*T65f(th6_3).inverse()*T54f(th5_3).inverse();
    Vector3d p41_3 = T41m.block<3,1>(0,3);
    double p41xz_3 = hypot(p41_3[0], p41_3[2]);

    //Four
    T41m = T10f(th1_2).inverse()*T60*T65f(th6_4).inverse()*T54f(th5_4).inverse();
    Vector3d p41_4 = T41m.block<3,1>(0,3);
    double p41xz_4 = hypot(p41_4[0], p41_4[2]);

    //Computation of the 8 possible values for th3
    double th3_1 = arccos((p41xz_1*p41xz_1-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2]));
    double th3_2 = arccos((p41xz_2*p41xz_2-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2]));
    double th3_3 = arccos((p41xz_3*p41xz_3-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2]));
    double th3_4 = arccos((p41xz_4*p41xz_4-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2]));

    double th3_5 = -th3_1;
    double th3_6 = -th3_2;
    double th3_7 = -th3_3;
    double th3_8 = -th3_4;

    //Computation of eight possible value for th2
    double th2_1 = atan2(-p41_1[2], -p41_1[0])-arcsin((-A[2]*sin(th3_1))/p41xz_1);
    double th2_2 = atan2(-p41_2[2], -p41_2[0])-arcsin((-A[2]*sin(th3_2))/p41xz_2);
    double th2_3 = atan2(-p41_3[2], -p41_3[0])-arcsin((-A[2]*sin(th3_3))/p41xz_3);
    double th2_4 = atan2(-p41_4[2], -p41_4[0])-arcsin((-A[2]*sin(th3_4))/p41xz_4);

    double th2_5 = atan2(-p41_1[2], -p41_1[0])-arcsin((A[2]*sin(th3_1))/p41xz_1);
    double th2_6 = atan2(-p41_2[2], -p41_2[0])-arcsin((A[2]*sin(th3_2))/p41xz_2);
    double th2_7 = atan2(-p41_3[2], -p41_3[0])-arcsin((A[2]*sin(th3_3))/p41xz_3);
    double th2_8 = atan2(-p41_4[2], -p41_4[0])-arcsin((A[2]*sin(th3_4))/p41xz_4);

    //Five
    Matrix4d T43m = T32f(th3_1).inverse()*T21f(th2_1).inverse()*T10f(th1_1).inverse()*T60*T65f(th6_1).inverse()*T54f(th5_1).inverse();
    Vector3d Xhat43 = T43m.block<3,1>(0,0);
    double th4_1 = atan2(Xhat43[1], Xhat43[0]);

    //Six
    T43m = T32f(th3_2).inverse()*T21f(th2_2).inverse()*T10f(th1_1).inverse()*T60*T65f(th6_2).inverse()*T54f(th5_2).inverse();
    Xhat43 = T43m.block<3,1>(0,0);
    double th4_2 = atan2(Xhat43[1], Xhat43[0]);

    //Seven
    T43m = T32f(th3_3).inverse()*T21f(th2_3).inverse()*T10f(th1_2).inverse()*T60*T65f(th6_3).inverse()*T54f(th5_3).inverse();
    Xhat43 = T43m.block<3,1>(0,0);
    double th4_3 = atan2(Xhat43[1], Xhat43[0]);

    //Eight
    T43m = T32f(th3_4).inverse()*T21f(th2_4).inverse()*T10f(th1_2).inverse()*T60*T65f(th6_4).inverse()*T54f(th5_4).inverse();
    Xhat43 = T43m.block<3,1>(0,0);
    double th4_4 = atan2(Xhat43[1], Xhat43[0]);

    //Nive
    T43m = T32f(th3_5).inverse()*T21f(th2_5).inverse()*T10f(th1_1).inverse()*T60*T65f(th6_1).inverse()*T54f(th5_1).inverse();
    Xhat43 = T43m.block<3,1>(0,0);
    double th4_5 = atan2(Xhat43[1], Xhat43[0]);

    //Ten
    T43m = T32f(th3_6).inverse()*T21f(th2_6).inverse()*T10f(th1_1).inverse()*T60*T65f(th6_2).inverse()*T54f(th5_2).inverse();
    Xhat43 = T43m.block<3,1>(0,0);
    double th4_6 = atan2(Xhat43[1], Xhat43[0]);

    //Eleven
    T43m = T32f(th3_7).inverse()*T21f(th2_7).inverse()*T10f(th1_2).inverse()*T60*T65f(th6_3).inverse()*T54f(th5_3).inverse();
    Xhat43 = T43m.block<3,1>(0,0);
    double th4_7 = float(atan2(Xhat43[1], Xhat43[0]));

    //Twelve
    T43m = T32f(th3_8).inverse()*T21f(th2_8).inverse()*T10f(th1_2).inverse()*T60*T65f(th6_4).inverse()*T54f(th5_4).inverse();
    Xhat43 = T43m.block<3,1>(0,0);
    double th4_8 = atan2(Xhat43[1], Xhat43[0]);

    Eigen::Matrix<double, 8, 6> T {
            {th1_1,th2_1,th3_1,th4_1,th5_1,th6_1},
            {th1_1,th2_2,th3_2,th4_2,th5_2,th6_2},
            {th1_2,th2_3,th3_3,th4_3,th5_3,th6_3},
            {th1_2,th2_4,th3_4,th4_4,th5_4,th6_4},
            {th1_1,th2_5,th3_5,th4_5,th5_1,th6_1},
            {th1_1,th2_6,th3_6,th4_6,th5_2,th6_2},
            {th1_2,th2_7,th3_7,th4_7,th5_3,th6_3},
            {th1_2,th2_8,th3_8,th4_8,th5_4,th6_4}
    };

    Th = T;
}

/**
 * @brief Evaluate if two decimal numbers are almost equal
 * 
 * @param[in] n1
 * @param[in] Tn2
 * @param[in] precision
 * @return bool
 */

bool areEqual(double n1, double n2, double precision = 0.001){
    double diff = std::abs(n1 - n2);
    return diff < precision;
}

/**
 * @brief checks if the final pose of every single link of the robot is over some space limits, such as the table height and the backward wall.
 * 
 * @param[in] Th
 * @return bool
 */

bool check_collision(RowVector6d Th){

    double th1 = Th[0];
    double th2 = Th[1];
    double th3 = Th[2];
    double th4 = Th[3];
    double th5 = Th[4];
    double th6 = Th[5];

    Matrix4d Tn;
    Tn.setIdentity(4,4);

    bool cond = true;

    Matrix4d matrici[6] = {T10f(th1),T21f(th2),T32f(th3),T43f(th4),T54f(th5),T65f(th6)};

    double pos_z;
    double pos_y;

    for(int i=0; i<6; i++){
        Tn = Tn*matrici[i];
        pos_z = Tn(2,3);
        pos_y = Tn(1,3);

        double max_y[2] = {0.22, 0.12};
        if(pos_z < 0 or pos_z >0.835 or pos_y > max_y[int(i/3)]){
            cond = false;
            break;
        }
    }

    return cond;
}

/**
 * @brief checks if the computed joint are reachable by the ur5 robot.
 * 
 * @param[in] Th
 * @return bool
 */

bool check_angles(RowVector6d Th){

    int cont = 0;
    bool ret = true;

    RowVector2d J1{{-6.14, 6.14}};
    RowVector2d J2{{-3.14, 0.0}};
    RowVector2d J3{{-3.14, 3.14}};
    RowVector2d J4{{-6.28, 6.28}};
    RowVector2d J5{{-6.28, 6.28}};
    RowVector2d J6{{-6.28, 6.28}};


    RowVector2d max_angles_value[6] = {J1,J2,J3,J4,J5,J6};

    double theta;

    for(int i=0; i<6; i++){       // i<th.cols()
        theta = Th[i];
        RowVector2d mav = max_angles_value[cont];
        if(theta > mav[0] and theta < mav[1]){
            cont++;
            continue;
        }
        else{
            ret = false;
        }
    }
    if(not check_collision(Th=Th)){
        ret = false;
    }

    return ret;
}

/**
 * @brief compute the norm of two joint configurations
 * 
 * @param[in] Th0
 * @param[in] th
 * @return double
 */

double norm(RowVector6d Th0, RowVector6d th){
    return sqrt(pow(Th0[0]-th[0], 2) + pow(Th0[1]-th[1], 2) + pow(Th0[2]-th[2], 2) + pow(Th0[3]-th[3], 2) + pow(Th0[4]-th[4], 2) + pow(Th0[5]-th[5], 2));
}


/**
 * @brief removes a row from a matrix
 * 
 * @param[in] matrix
 * @param[in] rowToRemove
 */

void removeRow(Matrix6d& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

    matrix.conservativeResize(numRows,numCols);
}

/**
 * @brief sorts the result of the inverse kinematics and removes the not allowed rows
 * 
 * @param[in] Th0
 * @param[in] all
 * @return bool
 */

bool bestInverse(RowVector6d Th0, Matrix6d& all){
    Matrix<double, 1, 6> thi;
    Matrix<double, 1, 6> thj;
    Matrix<double, 1, 6> tmp;

    bool ret = true;

    // Avoid too much rotation 
    for (int i=0;i<all.rows();i++){
        if (all.row(i)(0) > 0.8*M_PI){
            all.row(i)(0) = all.row(i)(0) - 2 * M_PI;
        }
        if (all.row(i)(0) < -M_PI){
            all.row(i)(0) = all.row(i)(0) +2*M_PI;
        }
    }


    int i=0;
    while (i<all.rows()){
        thi = all.row(i);
        if (!check_angles(thi)){
            removeRow(all, i);
        } else {
            i++;
        }
    }

    for(int i=0; i<all.rows(); i++){
        thi = all.row(i);
        double normaMin = norm(Th0, thi);
        int iMin = i;

        for(int j=i; j<all.rows(); j++){
            thj = all.row(j);
            if(norm(Th0, thj) < normaMin){
                normaMin = norm(Th0, thj);
                iMin = j;
            }
        }

        tmp = all.row(iMin);
        all.row(iMin) = all.row(i);
        all.row(i) = tmp;

    }

    if(all.size() == 0){
        return false;
    }

    return ret;
}

/**
 * @brief checks if the inserted position is reachable by the robot, comparing the result of the inverse and the direct kinematics
 * 
 * @param[in] pos
 * @param[in] q0
 * @return bool
 */

bool check_point(Vector3d pos, RowVector6d q0){

    Matrix<double, Dynamic, 6> confs;
    Matrix<double, 1, 6> Th;


    inverse_kinematics(pos, confs);    
    bestInverse(q0, confs);

    if(confs.rows() == 0){
      return false;
    }

    std::cout << "ciao" << std::endl;

    Th = confs.row(0);
    Matrix3d R06;
    Vector3d ef;
    direct_kinematics(Th,R06,ef);
    ef(1) = -ef(1);
    ef(2) = -ef(2);

    for(int i=0; i<3; i++){
        if(areEqual(ef[i], pos[i])){
            continue;
        }
        else{
            return false;
        }
    }
    return true;

}


