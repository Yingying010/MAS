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
ssh ubuntu@192.168.0.105
```

bring up basic packages to start TurtleBot3 applications by running:
```shell
roslaunch turtlebot3_bringup turtlebot3_robot.launch
```
 
 run the applications by using rosrun and roslaunch on local PC. 
```shell
export TURTLEBOT3_MODEL=${TB3_MODEL}
```

on local PC to launch the teleoperation of turtlebot.
```shell
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
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
uvcdynctrl --device=/dev/video0 --set='Focus, Auto' 0
```

check if the autofocus is off:
```shell
uvcdynctrl --device=/dev/video0 --get='Focus, Auto'
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
rosrun auto_navigation goal_pose.py
```

