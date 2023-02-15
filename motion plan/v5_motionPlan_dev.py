from v4_kinematics_dev import direct_kinematics, inverse_kinematics, check_point, bestInverse, check_angles, check_collision
import numpy as np
from math import cos
from math import sin
from math import pi
from math import pow
from math import atan2
from math import sqrt
from enum import Enum
import params


def eul2rotm(angles):
    phi = angles[0]
    theta = angles[1]
    gamma = angles[2]
    R = np.array([
        [cos(phi)*cos(theta), cos(phi)*sin(theta)*sin(gamma) - sin(phi)*cos(gamma), cos(phi)*sin(theta)*cos(gamma) + sin(phi)*sin(gamma)],
        [sin(phi)*cos(theta), sin(phi)*sin(theta)*sin(gamma) + cos(phi)*cos(gamma), sin(phi)*sin(theta)*cos(gamma) - cos(phi)*sin(gamma)],
        [-sin(theta), cos(theta)*sin(gamma), cos(theta)*cos(gamma)]
    ])
    return R

def rotm2eul(R):
    x = atan2(R[1,0],R[0, 0])
    y = atan2(-R[2,0],sqrt(pow(R[2,1],2)+pow(R[2,2],2)))
    z = atan2(R[2,1],R[2,2])
    return np.array([x,y,z])

def threep2p(qEs, xEf, phiEf, minT, maxT):
    Th = []

    moveT = int((maxT-minT)/3)   #Tempo di ogni movimento (scegliamo maxT-minT sempre divisibile per 3)
    t0 = [minT, minT+moveT, minT+2*moveT] #tempi iniziali dei tre movimenti
    tf = [maxT-2*moveT, maxT-moveT, maxT] #tempi finali dei tre movimenti
    
    # First move
    q0, q1, q2, q3 ,q4, q5 = qEs
    xE1 = direct_kinematics(q0, q1, q2, q3 ,q4, q5)[0]
    xE1[1] = -xE1[1]
    xE1[2] = -0.5
    phiE1 = np.array([0,0,0])        #0,0,0

    try:
        a, b, c = p2pMotionPlan(qEs, xE1, phiE1, t0[0], tf[0])
    except:
        return False

    Th = Th + a

    # Second move
    qEs = Th[len(Th)-1]
    xE2 = np.copy(xEf) 
    xE2[2] = -0.5
    sogliaX = 0.5
    if (abs(xE1[0] - xE2[0]) > sogliaX):
        slow = True
    else:
        slow = False

    try:
        a, b, c = p2pMotionPlan(qEs, xE2, phiEf, t0[1], tf[1], slow)
    except:
        return False
    
    #if not a:
    #    return False
    Th = Th + a
    #xE = xE + b
    #phiE = phiE + c

    # Third move
    #qEs = Th[int(((tf[1] - (t0[1]))/dt1) + (tf[0] - t0[0])/dt) -1]
    qEs = Th[len(Th)-1]
    '''
    phiEf[1] = angleRot
    '''

    try:
        a, b, c = p2pMotionPlan(qEs, xEf, phiEf, t0[2], tf[2])
    except:
       return False
    #if not a:
    #    return False

    Th = Th + a

    return Th


def p2pMotionPlan(qEs, xEf, phiEf, minT, maxT, slow=False):

    dt = np.array([params.robot_params["ur5"]['dt']])  #ur5 = robot_name
    if slow:
        dt = dt/2

    xE1 = np.copy(xEf)    
    #qEs = inverse_kinematics(xEs, phiEs)
    qEf = inverse_kinematics(xE1, eul2rotm(phiEf))  
    bestInverse(qEs, qEf)  #qEf array delle inverse OK ordinate per norma
    if not qEf:
        return False

    for j in range(len(qEf)):
        error = False

        A = ([])
        for i in range(len(qEs)):
            M = np.matrix([
                [1, minT, pow(minT, 2), pow(minT, 3)],
                [0, 1, 2*minT, 3*pow(minT, 2)],
                [1, maxT, pow(maxT, 2), pow(maxT, 3)],
                [0, 1, 2*maxT, 3*pow(maxT, 2)]
            ])
            b = np.matrix([
                [qEs[i]], [0], [qEf[j][i]], [0]
                ])
            a = np.linalg.inv(M)*b
            tem = np.array(a.T)
            A.append(tem)
        Th = ([])
        xE = ([])
        phiE = ([])
        t = minT
        while (t+0.000001 < maxT):
            th = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for i in range(len(qEs)):
                q = A[i][0][0] + A[i][0][1]*t + A[i][0][2]*t*t + A[i][0][3]*t*t*t
                th[i] = q
            if not check_angles(th):
                    print("ERRORE TRAIETTORIA " + str(t))
                    error = True
                    break   #esce dal while
            Th.append(th)
            mx, mr = direct_kinematics(th[0], th[1], th[2], th[3], th[4], th[5])
            xE.append([t, mx.T])
            phiE.append([t, rotm2eul(mr)])

            t += dt

        if not error:
            return Th, xE, phiE
    
    #Th:    set di tutte le configurazioni dei joint da P0 a PF
    #xE:    set di tutti i punti raggiunti da P0 a PF
    #phiE:  set di tutti frame dell'end effector durante la traiettoria
    return False
 
def ruota(qEs, xef, phief, tMin, tMax, direzione):
        
        Th = []
        xe1 = np.copy(xef)

        # if direzione == "r":    #"r"
        #     phief[0] = np.pi    

        a  = threep2p(qEs, xe1, phief, tMin, tMax)   #da dove sono al'oggetto

        if not a:
            return False

        Th = Th + a
        
        qEs = Th[len(Th)-1]
        '''
        # [z,y,x]
        if direzione == "l":
            xe1[0] = xe1[0]-0.123
            xe1[2] = xe1[2]-0.1
            phief = np.array([np.pi/2, 0, np.pi/2])       #np.pi, -np.pi/2, 0   orario positivo
        elif direzione == "f":
            xe1[1] = xe1[1]-0.123
            xe1[2] = xe1[2]-0.1
            phief = np.array([0, 0, np.pi/2])
        elif direzione == "r":
            xe1[0] = xe1[0]+0.123
            xe1[2] = xe1[2]-0.1
            phief = np.array([-np.pi/2, 0, np.pi/2])     #np.pi/2, 0, -np.pi/2
        '''
        
        xe1[0] = xe1[0]+0.123
        xe1[2] = xe1[2]-0.09
        
        a = threep2p(qEs, xe1, direzione, tMin, tMax)   #gira oggetto e lo rimette lì

        if not a:
            return False

        Th = Th + a

        qEs = Th[len(Th)-1]
        xe1 = np.copy(xef)
        phief = np.array([np.pi/2, 0, 0])
        a = threep2p(qEs, xe1, phief, tMin, tMax)   #prende l'oggetto da sopra

        if not a:
            return False

        Th = Th + a
        
        return Th

"""xe0 = np.array([0.1, -0.3, 0.1])
xef = np.array([0.5, 0.1, 0.5])
phie0 = np.array([0,0,0])
phief = np.array([pi/6, pi/3, pi/4])

#mat = eul2rotm(phie0)
#tem = inverse_kinematics(xe0, mat)

p2pMotionPlan(xe0, phie0, xef, phief, 0, 1, 0.01)"""

# threep2p([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.4,0.2,-0.6], [0,0,0], 0, 3)
# Th = threep2p([2.3705259724671266, -0.8350330025362989, 0.25114659099651027, -2.656975920327706, 1.5707963267948966, -0.79972964567223], [0.4, 0.2, -0.73], [0,0,0], 0, 3)
# th1, th2, th3, th4, th5, th6 = Th[299]
# print(direct_kinematics(th1, th2, th3, th4, th5, th6))
