---
Date: 2025-11-22
tags:
  - FinalChanllege
---
# CodeAnalysis
## main
针对**单个机器人**进行导航
```python
"""

Main function to initialize the ROS node and start the navigation process.

"""

# 初始化一个节点名称
# 在同一个ROS网络中，不能有两个相同的节点
rospy.init_node('robot_navigation')


# 这些标记的实际位置（真实世界坐标）已知，可以用于仿真与现实之间的坐标映射。
aruco_markers = ['id500', 'id501', 'id502', 'id503']

  
try:
	# 计算透视变换矩阵
	# Get the transformation matrix using the corner detection function
	matrix = get_transformation_matrix(aruco_markers)
except Exception as e:
	rospy.logerr(f"Failed to get transformation matrix: {e}")
	return


try:
	# 读取并转换路径点
	# Read and transform waypoints from the YAML file
	coordinates = read_and_transform_waypoints("./cbs_output.yaml", matrix)
except Exception as e:
	rospy.logerr(f"Failed to read and transform waypoints: {e}")
	return

  

# Start navigation with the first agent's waypoints
turtlebot_name = "turtle1" # Name of your TurtleBot
aruco_id = "id402" # ArUco marker ID for localization


# Begin the navigation process
navigation(turtlebot_name, aruco_id, coordinates)

```

## function

### navigation
该函数负责控制一个 TurtleBot 沿着指定路径点（waypoints）进行导航，使用 ArUco 标记提供机器人当前的位置。
- 发送速度指令
- 获取robot的初始位置
- 初始化Twist消息结构 （linear，angular）
- 导航主循环（是否到达目的地，是否ROS关闭）
	- 从路径点列表中取出当前目标点的坐标 `(x, y)`
	- 检查路径是否到达 
		- 未完成则更新next目标位置
		- 如果当前路径点列表都循环完了直接结束while
	- 计算机器人的当前方向角 `Orientation`（从四元数转换为欧拉角）
	- 计算机器人相对于目标路径点的方向角 `goal_direction`
	- 计算机器人相对于目标路径点的角度差 `theta`
		- `goal_direction - current_orientation`
	- 计算与下一个目标点的距离 `distance` 
	- 计算控制命令 （linear_velocity， angular_velocity 需要限制速度）
	- 发布Twist消息去控制小车
	
和sim的区别在于：
- 初始化阶段  未设置距离阈值
- 主循环条件 增加ROS关闭即退出
- 检查路径是否完成 并未设置小车的速度为0

```python
  

def navigation(turtlebot_name, aruco_id, goal_list):

"""
Navigates the TurtleBot through a list of waypoints.

Parameters:

- turtlebot_name (str): Name of the TurtleBot.

- aruco_id (str): ArUco marker ID used for localization.

- goal_list (List[Tuple[float, float]]): List of (X, Y) coordinates as waypoints.

"""
# 初始化
	# Index of the current waypoint
	current_position_idx = 0 
	
	# 发布 `Twist` 消息，用于控制机器人的线速度和角速度。
	# Publisher to send velocity commands to the robot
	cmd_pub = rospy.Publisher(f'/{turtlebot_name}/cmd_vel', Twist, queue_size=1)
	
	  
	# 从 ArUco 标记的 ROS 话题中获取机器人的初始位姿。
	# Wait for the initial pose message from the ArUco marker
	init_pose = rospy.wait_for_message(f'/{aruco_id}/aruco_single/pose', PoseStamped)
	
	  
	# Twist message for velocity commands
	twist = Twist()
	  
# 主循环
	# Loop until all waypoints are reached or ROS is shut down
	while current_position_idx < len(goal_list) and not rospy.is_shutdown():
	# 取出当前目标点的坐标
		# Get current goal coordinates
		goal_x, goal_y = goal_list[current_position_idx]

	# 检查路径是否到达 
		# Check if the goal has been reached
		if check_goal_reached(init_pose, goal_x, goal_y, tolerance=0.1):
		
			rospy.loginfo(f"Waypoint {current_position_idx + 1} reached: Moving to next waypoint.")
			current_position_idx += 1 # Move to the next waypoint
		
			# If all waypoints are reached, exit the loop
			if current_position_idx >= len(goal_list):
				rospy.loginfo("All waypoints have been reached.")
				break
	
	# 更新当前位置信息
		init_pose = rospy.wait_for_message(f'/{aruco_id}/aruco_single/pose', PoseStamped)

	# 计算机器人的当前方向角
		# Extract the current orientation in radians from quaternion
		orientation_q = init_pose.pose.orientation
		orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
		(roll, pitch, yaw) = euler_from_quaternion(orientation_list)
		current_orientation = yaw # Current heading of the robot
		
		  
	
	# 计算机器人相对于目标路径点的方向角
		dx = goal_x - init_pose.pose.position.x
		dy = goal_y - init_pose.pose.position.y
	
		goal_direction = math.atan2(dy, dx) # Angle to the goal
	
		# Normalize angles to range [0, 2π)
		current_orientation = (current_orientation + 2 * math.pi) % (2 * math.pi)
		goal_direction = (goal_direction + 2 * math.pi) % (2 * math.pi)

	# 计算机器人相对于目标路径点的角度差
		# Compute the smallest angle difference
		theta = goal_direction - current_orientation
			
		# Adjust theta to be within [-π, π]
		if theta > math.pi:
			theta -= 2 * math.pi
		elif theta < -math.pi:
			theta += 2 * math.pi
			
	# 计算与下一个目标点的距离
	  	distance = math.hypot(dx, dy) # Euclidean distance to the goal
	
		# Log debug information
		
		rospy.logdebug(f"Current Position: ({init_pose.pose.position.x:.2f}, {init_pose.pose.position.y:.2f})")
		
		rospy.logdebug(f"Goal Position: ({goal_x:.2f}, {goal_y:.2f})")
		
		rospy.logdebug(f"Current Orientation: {current_orientation:.2f} rad")
		
		rospy.logdebug(f"Goal Direction: {goal_direction:.2f} rad")
		
		rospy.logdebug(f"Theta (Angle to Goal): {theta:.2f} rad")
		
		rospy.logdebug(f"Distance to Goal: {distance:.2f} meters")
		
			  
	
	# 计算控制命令
		# Control parameters (adjust these as needed)
		k_linear = 0.5 # Linear speed gain
		k_angular = 2.0 # Angular speed gain
		
		# Compute control commands
		linear_velocity = k_linear * distance * math.cos(theta)
		angular_velocity = -k_angular * theta
		
		  
		# Limit maximum speeds if necessary
		max_linear_speed = 0.2 # meters per second
		max_angular_speed = 1.0 # radians per second
		
		linear_velocity = max(-max_linear_speed, min(max_linear_speed, linear_velocity))
		angular_velocity = max(-max_angular_speed, min(max_angular_speed, angular_velocity))
		
		  
	# 发送控制命令
		twist.linear.x = linear_velocity
		twist.angular.z = angular_velocity
		
		# Publish the velocity commands
		cmd_pub.publish(twist)
		
	
		# Sleep to maintain the loop rate
		rospy.sleep(0.1) # Adjust the sleep duration as needed
	
	```





## Output

goal_list = [(1, 1), (2, 2), (3, 3)]

在sim里面是
schedule = [ {"x": 1, "y": 1},  {"x": 2, "y": 2},  {"x": 3, "y": 3} ]



# Update MR Version
step1:
改成多线程
```python

# Start navigation with the first agent's waypoints
agents = ["turtle1", "turtle2"]
aruco_ids = ["id402", "id403"]

# Begin the navigation process
# navigation(turtlebot_name, aruco_id, coordinates)

run(agent_box_ids, box_id_to_goal, box_id_to_schedule)

```


