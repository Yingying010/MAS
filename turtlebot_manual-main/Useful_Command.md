After pulling the repository, copy all packages to the *catkin_ws/src* directory and then run 
```shell
$ catkin_make
```

update the ROS IP settings
```shell
$ nano ~/.bashrc
$ source ~/.bashrc
```

build the packages and run
```shell
$ source ~/catkin_ws/devel/setup.bash
```

run 
```shell
$ ssh ubuntu@192.168.0.105
```

bring up basic packages to start TurtleBot3 applications by running:
```shell
$ roslaunch turtlebot3_bringup turtlebot3_robot.launch
```
 
 run the applications by using rosrun and roslaunch on local PC. 
```shell
$ export TURTLEBOT3_MODEL=${TB3_MODEL}
```

on local PC to launch the teleoperation of turtlebot.
```shell
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

run auto_aruco_maker_finder
```shell
$ roslaunch auto_aruco_marker_finder multiple_aruco_marker_finder.launch
```

open rqt
```shell
$ rosrun rqt_gui rqt_gui
```

run navigation
```shell
$ rosrun auto_navigation goal_pose.py
```

