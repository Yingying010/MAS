After pulling the repository, copy all packages to the *catkin_ws/src* directory and then run 
```shell
cd ~/catkin_ws
catkin_make
```

update the ROS IP settings
```shell
nano ~/.bashrc
source ~/.bashrc
```

build the packages and run
```shell
source ~/catkin_ws/devel/setup.bash
```

run 
```shell
ssh ubuntu@192.168.0.106
ssh ubuntu@192.168.0.217
```

bring up basic packages to start TurtleBot3 applications by running:
```shell
ROS_NAMESPACE=tb3_0 roslaunch turtlebot3_bringup turtlebot3_robot.launch
ROS_NAMESPACE=tb3_1 roslaunch turtlebot3_bringup turtlebot3_robot.launch
```
 
 run the applications by using rosrun and roslaunch on local PC. 
```shell
export TURTLEBOT3_MODEL=${TB3_MODEL}
```

on local PC to launch the teleoperation of turtlebot.
```shell
ROS_NAMESPACE=tb3_0 roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
ROS_NAMESPACE=tb3_1 roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```

check camera list
```shell
ls /dev/video*
```

check if another application is using the camera
```shell
lsof /dev/video0
```

check if user has the right of this Camera
```shell
ls -l /dev/video0
```

make sure the right is in the groups, if not and add and restart
```shell
groups
sudo usermod -aG video $USER
```

check the status of camera lists connecting ubuntu
```shell
ls /dev |grep video
```

check if your camera supports autofocus:
```shell
uvcdynctrl --device=/dev/video0 --clist
```

turn off the autofocus:
```shell
uvcdynctrl --device=/dev/video0 --set='Focus, Automatic Continuous' 0
```

check if the autofocus is off:
```shell
uvcdynctrl --device=/dev/video0 --get='Focus, Automatic Continuous'
```

change the focus
```shell
uvcdynctrl --device=/dev/video0 --set='Focus, Absolute' 20
```

run auto_aruco_maker_finder
```shell
roslaunch auto_aruco_marker_finder multiple_aruco_marker_finder.launch
```

Verify the package
```shell
rospack find aruco_ros
```

open rqt
```shell
rosrun rqt_gui rqt_gui
```

run navigation
```shell
rosrun auto_navigation goal_pose_old.py
rosrun auto_navigation goal_pose_SR_autoNav.py
rosrun auto_navigation goal_pose_MR_autoNav.py

```


goal_pose_old.py is one robot and one aruco
goal_pose_SR_autoNav.py is one robot and one goal but using the yaml file


