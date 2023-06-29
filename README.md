<div align="center">
    <p>
        <img width="100%" src="assets/images/splash-logo.png"/>
    </p>
    <div>
        <img src="https://img.shields.io/badge/c++-%2300599C.svg?style=flat&logo=c%2B%2B&logoColor=white" alt="C++"/>
        <img src="https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54" alt="Python"/>
    </div>
</div>

# Table of contents

-   [Introduction](#introduction)
-   [Installation and Configuration](#installation-and-configuration)
-   [Getting started](#getting-started)
-   [Contributions](#contributions)

# Introduction

There are four different assignments with increasing difficulty. In each of them, the 3D camera has to recognize the blocks on the table, retrieve the position and the orientation of them, and finally the blocks have to be taken by the robot arm and moved in a certain position. (drop area). The assignments are the following:

* **Assignment 1**: there is only one object in the initial stand, which is positioned with its base “naturally” in contact with the ground. The object can be of any of the classes specified by the project. Each class has an assigned position on the final stand, which is marked by a coloured shape representing the silhouette of the object. KPI 1-1 time to detect the position of the object. KPI 1-2 time to move the object between its initial and its final positions, counting from the instant in which both of them have been identified.
* **Assignment 2**: there are multiple objects on the initial stand, one for each class. There is no specific order in the initial configuration, except that the base of the object is “naturally” in contact with the ground. Each object has to be picked up and stored in the position prescribed for its class and marked by the object’s silhouette. KPI 2-1: Total time to move all the objects from their initial to their final positions.
* **Assignment 3**: there are multiple objects on the initial stand, and there can be more than one object for each class. The objects are positioned randomly on the stand but would not stand or lean on each other. 
  An object could be lying on one of its lateral sides or on its top. Each object has to be stored in the position prescribed by its class. Objects of the same class have to be stacked up to form a tower. KPI 3-1: Total time to move all the objects from their initial to their final positions.
* **Assignment 4**: the objects on the initial stand are those needed to create a composite object with a known design (e.g., a castle). The objects are positioned randomly on the stand. An object could be lying on one of its lateral sides or on its top. The objects could also stand or lean on each other. The manipulator has to pick them up in sequence and create the desired composite object on the final stand.

# Installation and Configuration

## Locosim

Follow the instruction available at this [link](https://github.com/mfocchi/locosim/tree/659f15fe13895336c0cb11469ef34e747bd84c7f#native-installation-on-linux).

## Repository

Clone the repository:

```bash
git clone https://github.com/christiansassi/robotics-project.git
```

Init and update the yolov5 submodule:

```bash
git submodule init
git submodule update
```

Once done, copy the [motion plan](motion%20plan) folder inside the `ros_ws/src/motion_plan` folder (create the folder if it not exists via `mkdir motion_plan`) and inside the `ros_ws/src/motion_plan` extract the [Eigen](motion%20plan/Eigen.tar.gz) library:

```bash
tar -xzvf Eigen.tar.gz
rm Eigen.tar.gz
```

Inside `ros_ws` folder type:

```bash
catkin_make install
```

Finally, change the gripper type inside `ros_ws/src/locosim/robot_control/lab_exercises/lab_palopoli/params.py` script by setting [this line](https://github.com/mfocchi/robot_control/blob/a1babcb55217681b73229f3f9ec8ce93c477bc4d/lab_exercises/lab_palopoli/params.py#L45C8-L45C8) to `True`.

# Getting started

Open a new terminal and navigate inside `ros_ws/src/locosim/robot_control/lab_exercises/lab_palopoli` folder. Once done type:

```bash
python3 ur5_generic.py
```

After the homing procedure has finished, navigate inside `ros_ws` folder, open a new terminal and type:

```bash
rosrun motion_plan_pkg motion_plan
```

Once you see *"Waiting for messages"*, navigate inside [vision](vision) folder and type:

```bash
python3 zed.py
```

Finally, once you see the "Live detection" window opening, open a new terminal and type:

```bash
python3 assignments.py
```

Once done, you can choose which assignment to execute.

# Contributions

Luca Pedercini [218551] - [luca.pedercini@studenti.unitn.it](mailto:luca.pedercini@studenti.unitn.it)

Pietro Bologna [218186] - [pietro.bologna@studenti.unitn.it](mailto:pietro.bologna@studenti.unitn.it)

Christian Sassi [217640] - [christian.sassi@studenti.unitn.it](mailto:christian.sassi@studenti.unitn.it)

<a href="https://www.unitn.it/"><img src="assets/images/unitn-logo.png" width="300px"></a>
