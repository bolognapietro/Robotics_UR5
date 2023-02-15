import numpy as np
from numpy import linalg
from numpy import array as arr
import cmath
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from cmath import acos as arccos
from cmath import asin as arcsin
from math import sqrt as sqrt
from math import pi


def acos(tetha):
    return arccos(tetha).real

def asin(tetha):
    return arcsin(tetha).real

#Direct Kinematics of the UR5
#Th: six joint angles
#pe: cartesian position of the end effector
#Re: Rotation matrix of the end effecto



#function [pe,Re] = ur5Direct(Th)
#Vector of the a distance (expressed in metres)

A = [0, -0.425, -0.3922, 0, 0, 0]
#Vector of the D distance (expressed in metres)
D = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]

def direct_kinematics(th1,th2,th3,th4,th5,th6):

    global A,D

    alfa = [0, pi/2, 0, 0, pi/2, -pi/2]


    T10f = np.matrix([
        [cos(th1), -sin(th1), 0, 0],
        [sin(th1), cos(th1), 0, 0],
        [0, 0, 1, D[0]],
        [0, 0, 0, 1],
        ])
    T21f = np.matrix([
        [cos(th2), -sin(th2), 0, 0],
        [0, 0, -1, 0],
        [sin(th2), cos(th2), 0, 0],
        [0, 0, 0, 1]
        ])

    T32f =np.matrix([
        [cos(th3), -sin(th3), 0, A[1]],
        [sin(th3), cos(th3), 0, 0],
        [0, 0, 1, D[2]],
        [0, 0, 0, 1]
        ])

    T43f = np.matrix([
        [cos(th4), -sin(th4), 0, A[2]],
        [sin(th4), cos(th4), 0, 0],
        [0, 0, 1, D[3]],
        [0, 0, 0, 1]
        ])

    T54f = np.matrix([
        [cos(th5), -sin(th5), 0, 0],
        [0, 0, -1, -D[4]],
        [sin(th5), cos(th5), 0, 0],
        [0, 0, 0, 1]
        ])
    T65f = np.matrix([
        [cos(th6), -sin(th6), 0, 0],
        [0, 0, 1, D[5]],
        [-sin(th6), -cos(th6), 0, 0],
        [0, 0, 0, 1]
        ])

    T06 =T10f*T21f*T32f*T43f*T54f*T65f

    Oe = T06*np.array([[0],[0],[0],[1]])

    R06 = T06[0:3,0:3]
    R06 = np.array(R06)


    #su rviz gli assi hanno direzione linalg.inversa

    #Oe[1]=-Oe[1]
    #Oe[2]=-Oe[2]

    return Oe[0:3],R06
                    

def inverse_kinematics(p60,R60=np.identity(3)): #p60 sono x,y,z R60 è la matrice di rotazione
    #linalg.inverse Kineamtics of UR5

    #INVERTO LA Y e Z PERCHE' SU GAZEBO E' SEMPRE IL CONTRARIO (assi invertiti ? )

    p60[1] = -p60[1]
    p60[2] = -p60[2]

    
    #function [Th] = ur5linalg.inverse(p60, R60)

    #Vector of the a distance (expressed in metres)
    A = [0, -0.425, -0.3922, 0, 0, 0]
    #Vector of the D distance (expressed in metres)
    D = [0.1625, 0, 0, 0.1333, 0.0997, 0.0996]

    T60 = np.matrix([[R60[0][0],R60[0][1],R60[0][2],p60[0]], 
                    [R60[1][0],R60[1][1],R60[1][2],p60[1]],
                    [R60[2][0],R60[2][1],R60[2][2],p60[2]],
                    [0,0,0,1]
                    ])
    
    def T10f(th1):
        return np.matrix([
        [cos(th1), -sin(th1), 0, 0],
        [sin(th1), cos(th1), 0, 0],
        [0, 0, 1, D[0]],
        [0, 0, 0, 1],
        ])

    def T21f(th2) :
        return np.matrix([
        [cos(th2), -sin(th2), 0, 0],
        [0, 0, -1, 0],
        [sin(th2), cos(th2), 0, 0],
        [0, 0, 0, 1]
        ])

    def T32f(th3) :
        return np.matrix([
        [cos(th3), -sin(th3), 0, A[1]],
        [sin(th3), cos(th3), 0, 0],
        [0, 0, 1, D[2]],
        [0, 0, 0, 1]
        ])

    def T43f(th4) :
        return np.matrix([
        [cos(th4), -sin(th4), 0, A[2]],
        [sin(th4), cos(th4), 0, 0],
        [0, 0, 1, D[3]],
        [0, 0, 0, 1]
        ])

    def T54f(th5) :
        return np.matrix([
        [cos(th5), -sin(th5), 0, 0],
        [0, 0, -1, -D[4]],
        [sin(th5), cos(th5), 0, 0],
        [0, 0, 0, 1]
        ])
        
    def T65f(th6) :
        return np.matrix([
        [cos(th6), -sin(th6), 0, 0],
        [0, 0, 1, D[5]],
        [-sin(th6), -cos(th6), 0, 0],
        [0, 0, 0, 1]
        ])

    def hypot(x1,x2): #metteva abs
        return (sqrt(x1*x1 + x2*x2))

    #Finding th1
    p50 = T60*np.matrix([[0],[0],[-D[5]],[1]])
    th1_1 = float(atan2(p50[1], p50[0]) + acos(D[3]/hypot(p50[1], p50[0])))+pi/2
    th1_2 = float(atan2(p50[1], p50[0]) - acos(D[3]/hypot(p50[1], p50[0])))+pi/2
    
    #finding th5
    th5_1 = +float(acos((p60[0]*sin(th1_1) - p60[1]*cos(th1_1)-D[3]) / D[5]))
    th5_2 = -float(acos((p60[0]*sin(th1_1) - p60[1]*cos(th1_1)-D[3]) / D[5]))
    th5_3 = +float(acos((p60[0]*sin(th1_2) - p60[1]*cos(th1_2)-D[3]) / D[5]))
    th5_4 = -float(acos((p60[0]*sin(th1_2) - p60[1]*cos(th1_2)-D[3]) / D[5]))
    
    T60 = np.matrix(T60)
    T60 = T60.astype('float64')
    
    #related to th11 a th51
    T06 = linalg.inv(T60)
    Xhat = T06[:,0][0:3] #seleziono la prima colonna e ne estraggo solamente i primi 3 elementi 
    Yhat = T06[:,1][0:3]
    
    th6_1 = float(atan2(((-Xhat[1]*sin(th1_1)+Yhat[1]*cos(th1_1)))/sin(th5_1), ((Xhat[0]*sin(th1_1)-Yhat[0]*cos(th1_1)))/sin(th5_1)))
    #related to th11 a th52
    th6_2 = float(atan2(((-Xhat[1]*sin(th1_1)+Yhat[1]*cos(th1_1))/sin(th5_2)), ((Xhat[0]*sin(th1_1)-Yhat[0]*cos(th1_1))/sin(th5_2))))
    #related to th12 a th53
    th6_3 = float(atan2(((-Xhat[1]*sin(th1_2)+Yhat[1]*cos(th1_2))/sin(th5_3)), ((Xhat[0]*sin(th1_2)-Yhat[0]*cos(th1_2))/sin(th5_3))))
    #related to th12 a th54
    th6_4 = float(atan2(((-Xhat[1]*sin(th1_2)+Yhat[1]*cos(th1_2))/sin(th5_4)), ((Xhat[0]*sin(th1_2)-Yhat[0]*cos(th1_2))/sin(th5_4))))
    
    #One
    T41m = linalg.inv(T10f(th1_1))*T60*linalg.inv(T65f(th6_1))*linalg.inv(T54f(th5_1))
    p41_1 = T41m[:,3][0:3]
    p41xz_1 = hypot(p41_1[0], p41_1[2])
    
    #Two
    T41m = linalg.inv(T10f(th1_1))*T60*linalg.inv(T65f(th6_2))*linalg.inv(T54f(th5_2))
    p41_2 = T41m[:,3][0:3]
    p41xz_2 = hypot(p41_2[0], p41_2[2])
    
    #Three
    T41m = linalg.inv(T10f(th1_2))*T60*linalg.inv(T65f(th6_3))*linalg.inv(T54f(th5_3))
    p41_3 = T41m[:,3][0:3]
    p41xz_3 = hypot(p41_3[0], p41_3[2])
    
    #Four
    T41m = linalg.inv(T10f(th1_2))*T60*linalg.inv(T65f(th6_4))*linalg.inv(T54f(th5_4))
    p41_4 = T41m[:,3][0:3]
    p41xz_4 = hypot(p41_4[0], p41_4[2])
    
    #Computation of the 8 possible values for th3    
    th3_1 = float(acos((p41xz_1*p41xz_1-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2])))
    th3_2 = float(acos((p41xz_2*p41xz_2-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2])))
    th3_3 = float(acos((p41xz_3*p41xz_3-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2])))
    th3_4 = float(acos((p41xz_4*p41xz_4-A[1]*A[1]-A[2]*A[2])/(2*A[1]*A[2])))
    
    th3_5 = -th3_1
    th3_6 = -th3_2
    th3_7 = -th3_3
    th3_8 = -th3_4
    
    #Computation of eight possible value for th2
    th2_1 = float(atan2(-p41_1[2], -p41_1[0])-asin((-A[2]*sin(th3_1))/p41xz_1))
    th2_2 = float(atan2(-p41_2[2], -p41_2[0])-asin((-A[2]*sin(th3_2))/p41xz_2))
    th2_3 = float(atan2(-p41_3[2], -p41_3[0])-asin((-A[2]*sin(th3_3))/p41xz_3))
    th2_4 = float(atan2(-p41_4[2], -p41_4[0])-asin((-A[2]*sin(th3_4))/p41xz_4))
    
    th2_5 = float(atan2(-p41_1[2], -p41_1[0])-asin((A[2]*sin(th3_1))/p41xz_1))
    th2_6 = float(atan2(-p41_2[2], -p41_2[0])-asin((A[2]*sin(th3_2))/p41xz_2))
    th2_7 = float(atan2(-p41_3[2], -p41_3[0])-asin((A[2]*sin(th3_3))/p41xz_3))
    th2_8 = float(atan2(-p41_4[2], -p41_4[0])-asin((A[2]*sin(th3_4))/p41xz_4))
    
    #Five
    T43m = linalg.inv(T32f(th3_1))*linalg.inv(T21f(th2_1))*linalg.inv(T10f(th1_1))*T60*linalg.inv(T65f(th6_1))*linalg.inv(T54f(th5_1))
    Xhat43 = T43m[:,0][0:3]
    th4_1 = float(atan2(Xhat43[1], Xhat43[0]))
    
    #Six
    T43m = linalg.inv(T32f(th3_2))*linalg.inv(T21f(th2_2))*linalg.inv(T10f(th1_1))*T60*linalg.inv(T65f(th6_2))*linalg.inv(T54f(th5_2))
    Xhat43 = T43m[:,0][0:3]
    th4_2 = float(atan2(Xhat43[1], Xhat43[0]))
    
    #Seven
    T43m = linalg.inv(T32f(th3_3))*linalg.inv(T21f(th2_3))*linalg.inv(T10f(th1_2))*T60*linalg.inv(T65f(th6_3))*linalg.inv(T54f(th5_3))
    Xhat43 = T43m[:,0][0:3]
    th4_3 = float(atan2(Xhat43[1], Xhat43[0]))
    
    #Eight
    T43m = linalg.inv(T32f(th3_4))*linalg.inv(T21f(th2_4))*linalg.inv(T10f(th1_2))*T60*linalg.inv(T65f(th6_4))*linalg.inv(T54f(th5_4))
    Xhat43 = T43m[:,0][0:3]
    th4_4 = float(atan2(Xhat43[1], Xhat43[0]))
    
    #Nive
    T43m = linalg.inv(T32f(th3_5))*linalg.inv(T21f(th2_5))*linalg.inv(T10f(th1_1))*T60*linalg.inv(T65f(th6_1))*linalg.inv(T54f(th5_1))
    Xhat43 = T43m[:,0][0:3]
    th4_5 = float(atan2(Xhat43[1], Xhat43[0]))
    
    #Ten
    T43m = linalg.inv(T32f(th3_6))*linalg.inv(T21f(th2_6))*linalg.inv(T10f(th1_1))*T60*linalg.inv(T65f(th6_2))*linalg.inv(T54f(th5_2))
    Xhat43 = T43m[:,0][0:3]
    th4_6 = float(atan2(Xhat43[1], Xhat43[0]))
    
    #Eleven
    T43m = linalg.inv(T32f(th3_7))*linalg.inv(T21f(th2_7))*linalg.inv(T10f(th1_2))*T60*linalg.inv(T65f(th6_3))*linalg.inv(T54f(th5_3))
    Xhat43 = T43m[:,0][0:3]
    th4_7 = float(atan2(Xhat43[1], Xhat43[0]))
    
    #Twelve
    T43m = linalg.inv(T32f(th3_8))*linalg.inv(T21f(th2_8))*linalg.inv(T10f(th1_2))*T60*linalg.inv(T65f(th6_4))*linalg.inv(T54f(th5_4))
    Xhat43 = T43m[:,0][0:3]
    th4_8 = float(atan2(Xhat43[1], Xhat43[0])) 
    
    Th = [[th1_1,th2_1,th3_1,th4_1,th5_1,th6_1],
        [th1_1,th2_2,th3_2,th4_2,th5_2,th6_2],
        [th1_2,th2_3,th3_3,th4_3,th5_3,th6_3],
        [th1_2,th2_4,th3_4,th4_4,th5_4,th6_4],
        [th1_1,th2_5,th3_5,th4_5,th5_1,th6_1],
        [th1_1,th2_6,th3_6,th4_6,th5_2,th6_2],
        [th1_2,th2_7,th3_7,th4_7,th5_3,th6_3],
        [th1_2,th2_8,th3_8,th4_8,th5_4,th6_4]
        ]
    
    return Th
    # for configuration in Th :
    #     if(check_angles(configuration)):
    #         return configuration
    #     else:
    #       continue
    
    #return False # dovrebbe dire di stare fermo quindi di non applicare nessun angolo per ora ritorniamo falso

    #return [th1_1,th2_1,th3_1,th4_1,th5_1,th6_1]


def check_collision(Th):  # Th array di 6 angol

    global A,D

    th1,th2,th3,th4,th5,th6 = Th

    T10f = np.matrix([
        [cos(th1), -sin(th1), 0, 0],
        [sin(th1), cos(th1), 0, 0],
        [0, 0, 1, D[0]],
        [0, 0, 0, 1],
        ])
    T21f = np.matrix([
        [cos(th2), -sin(th2), 0, 0],
        [0, 0, -1, 0],
        [sin(th2), cos(th2), 0, 0],
        [0, 0, 0, 1]
        ])

    T32f =np.matrix([
        [cos(th3), -sin(th3), 0, A[1]],
        [sin(th3), cos(th3), 0, 0],
        [0, 0, 1, D[2]],
        [0, 0, 0, 1]
        ])

    T43f = np.matrix([
        [cos(th4), -sin(th4), 0, A[2]],
        [sin(th4), cos(th4), 0, 0],
        [0, 0, 1, D[3]],
        [0, 0, 0, 1]
        ])

    T54f = np.matrix([
        [cos(th5), -sin(th5), 0, 0],
        [0, 0, -1, -D[4]],
        [sin(th5), cos(th5), 0, 0],
        [0, 0, 0, 1]
        ])
    T65f = np.matrix([
        [cos(th6), -sin(th6), 0, 0],
        [0, 0, 1, D[5]],
        [-sin(th6), -cos(th6), 0, 0],
        [0, 0, 0, 1]
        ])

    matrici = [T10f,T21f,T32f,T43f,T54f,T65f]
    Tn=np.identity(4)

    cond = True
    #calcolo la posizione dell n-esimo end effector
    #Per selezionare il numero di end-effector modificare la variabile num_joint
    for i in range(6):
        Tn = Tn*matrici[i]
        pos_z = Tn[2,3]
        pos_y = Tn[1,3]

        # pos_y (not wrist joints) = -0.22
        # pos_y (wrist joints) = -0.11
        max_y = [0.22, 0.12]
        #or pos_y > max_y[int(i/3)]
        if(pos_z < 0 or pos_z > 0.835 ):  # non si sa perchè bisogna controllare che sia negativa e non positiva
            cond = False
            break

    return cond



def check_angles(Th):

    # check for each angles if the ur5 can apply it . Checking for each angle if is inside the associate range in max_angles_value

    cont = 0

    max_angles_value = [[-6.14,6.14],
                        [-3.14,0],
                        [-3.14,3.14],
                        [-6.28,6.28],
                        [-6.28,6.28],
                        [-6.28,6.28]]

    for tetha in Th:

        if(tetha > max_angles_value[cont][0] and tetha < max_angles_value[cont][1] ):
            cont += 1
            continue
        else:
            return False

    if(not check_collision(Th=Th)):
        return False

    return True

def norma(Th0, th):
    return sqrt(pow(Th0[0]-th[0], 2) + pow(Th0[1]-th[1], 2) + pow(Th0[2]-th[2], 2) + pow(Th0[3]-th[3], 2) + pow(Th0[4]-th[4], 2) + pow(Th0[5]-th[5], 2))

def bestInverse(Th0, best):
    
    for th in best:
        if check_angles(th) == False:
            best.remove(th)
            
    for i in range(len(best)):
        normaMin = norma(Th0, best[i])
        iMin = i
        for j in range(i, len(best)):
            if norma(Th0, best[j]) < normaMin:
                normaMin = norma(Th0, best[j])
                iMin = j

        temp = np.copy(best[iMin])
        best[iMin] = np.copy(best[i])
        best[i] = np.copy(temp)

    if best == []:
        return False
    return best   #best = array di tutte le configurazioni buone, ordinate per norma

def check_point(_pos, q0):

    #chiede gli angoli alla inverse kinematics che tra la lista di 8 array guarda che ci sia almeno un array di angoli che 
    #rispetta le condizioni, ovvero angoli la cui applicazione non comportano che nessun joint sbatta sul soffitto , ovvero abbia z > 0
    # e che gli angoli che bisogna applicare si possano veramente 
    position = np.copy(_pos)
    confs = inverse_kinematics(position)
    bestInverse(q0, confs)
     
    if(not confs):
        return False
    else:
        th1,th2,th3,th4,th5,th6 = confs[0]

    #controlla che la posizione inserita sia raggiungibile dal robot , per farlo controlla che il risultaro della inverse messo dentro la direct dia la medesima posizione 
    pos,r06 = direct_kinematics(th1,th2,th3,th4,th5,th6 )


    for item in range(0,3) :
        if round(float(pos[item]),4) == round(position[item],4):
            continue
        else:
            return False

    return True






#pos,R60 = direct_kinematics(4.789388287800727, 0.28031363309080415, 0.6501166699637116, -2.5012266298494126, 1.5707963267948974, 3.0645933461737562)
#pos = np.array([-0.19001646,0.72999316,-0.16997545])
#R60 = np.identity(3)
"""Th = [0,0,0,0,0,0]
check_z_collison(Th=Th)"""

"""R60=np.array([[ 9.99999999e-01,-3.67490534e-06 ,-4.36920760e-05],
 [ 3.67115102e-06 , 9.99999996e-01, -8.59264006e-05],
 [ 4.36923916e-05 , 8.59262401e-05 , 9.99999995e-01]])"""
#Th = inverse_kinematics([0.4,0.2,-0.73])
#bestInverse([0.245, 1.254, 0.145, -0.15, 1.011, 0.140], Th)
print(direct_kinematics(0,0,0,0,0,0))