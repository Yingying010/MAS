---
Date: 2025-11-22
tags:
  - FinalChanllege
  - CodeAnalysis
  - cbs
  - simulation
---

# CodeAnalysis

## main
- 连接到 PyBullet GUI（图形用户界面），用于显示仿真窗口
- 设置 PyBullet 的资源搜索路径
- 禁用渲染，暂时关闭图形渲染
- 禁用 GUI 控件
- 禁用 TinyRenderer（基于 CPU 的渲染器）
- 加载pybullet内置的平面模型
- 标记环境是否已加载。
- 使用配置好的障碍物环境文件
- 定义机器人属性
- 重新使用渲染
- 启用实时仿真
- 设置重力加速度
- 调整摄像机视角
- 调用 `cbs.run` 执行路径规划，规划结果保存到 `cbs_output.yaml` 文件
- 使用 `read_cbs_output` 读取路径规划结果
- 执行仿真
- 仿真结束后等待 2 秒，确保所有机器人完成操作。

```python
  
# 连接到 PyBullet GUI（图形用户界面），用于显示仿真窗口。
# `--width=1920 --height=1080`：设置 GUI 窗口的分辨率为 1920x1080。
# `--mp4=Robot2_finalChanllege.mp4`：记录仿真过程
# `--mp4fps=30`：设置导出 MP4 视频的帧率为 30 FPS。
# 只连接GUI不录制视频 physics_client = p.connect(p.GUI)
physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=Robot2_finalChanllege.mp4 --mp4fps=30')


# 设置 PyBullet 的资源搜索路径，默认包含一些基础模型（如 `plane.urdf` 和 `r2d2.urdf`）。
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 禁用渲染，暂时关闭图形渲染以提高初始化效率。
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# 禁用 GUI 控件（如左侧滑块）。
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# 禁用 TinyRenderer（基于 CPU 的渲染器），仿真中不需要时可以提高效率。
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

  
# 加载pybullet内置的平面模型
plane_id = p.loadURDF("plane.urdf")

# 用于标记环境是否已加载。
global env_loaded
env_loaded = False

# 使用配置好的障碍物环境文件 (dimensions / obstacles)
env_params = create_env("final_challenge/env.yaml")

# 定义机器人属性（name / goal / start)
# 返回值：
# -`agent_box_ids`：所有机器人在 PyBullet 中的唯一 ID 列表。
# - `agent_name_to_box_id`：将机器人名称映射到对应的 PyBullet 对象 ID。
# - `box_id_to_goal`：将机器人 ID 映射到目标位置。
# - `agent_yaml_params`：从 YAML 文件中解析的机器人属性字典。
agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params = create_agents("final_challenge/actors.yaml")

  
# 重新使用渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

# 启用实时仿真，物理引擎根据真实时间步运行。
p.setRealTimeSimulation(1)
# 设置重力加速度，单位为 m/s²
p.setGravity(0, 0, -10)
# 调整摄像机视角
p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9,
cameraTargetPosition=[4.5, 4.5, 4])

# 调用 `cbs.run` 执行路径规划
# 规划结果保存到 `cbs_output.yaml` 文件
cbs.run(dimensions=env_params["map"]["dimensions"], obstacles=env_params["map"]["obstacles"], agents=agent_yaml_params["agents"], out_file="./final_challenge/cbs_output.yaml")
# 使用 `read_cbs_output` 读取路径规划结果
cbs_schedule = read_cbs_output("final_challenge/cbs_output.yaml")

# Replace agent name with box id in cbs_schedule
box_id_to_schedule = {}
for name, value in cbs_schedule.items():
	box_id_to_schedule[agent_name_to_box_id[name]] = value

# 执行仿真
run(agent_box_ids, box_id_to_goal, box_id_to_schedule)

# 仿真结束后等待 2 秒，确保所有机器人完成操作。
time.sleep(2)

```



## function
	def create_boundaries(length, width):
	def create_env(yaml_file):
	def create_agents(yaml_file):
	def read_cbs_output(file):
	def checkPosWithBias(Pos, goal, bias):
	def navigation(agent, goal, schedule):
	def run(agents, goals, schedule):


### create_env
根据指定的 YAML 配置文件加载仿真环境。
```python
def create_env(yaml_file):
"""

Creates and loads assets only related to the environment such as boundaries and obstacles.

Robots are not created in this function (check `create_turtlebot_actor`).

"""

	with open(yaml_file, 'r') as f:
	
		try:
	
			env_params = yaml.load(f, Loader=yaml.FullLoader)
	
		except yaml.YAMLError as e:
	
			print(e)
		
	# Create env boundaries
	
	dimensions = env_params["map"]["dimensions"]
	
	create_boundaries(dimensions[0], dimensions[1])
	
	  
	
	# Create env obstacles
	
	for obstacle in env_params["map"]["obstacles"]:
	
		p.loadURDF("./final_challenge/assets/cube.urdf", [obstacle[0], obstacle[1], 0.5])

	return env_params
```

### run
多线程运行navigation()
```python
def run(agents, goals, schedule):
	"""
	Set up loop to publish leftwheel and rightwheel velocity for each robot to reach goal position.
	
	Args:
	agents: array containing the boxID for each agent
	schedule: dictionary with boxID as key and path to the goal as list for each robot.
	goals: dictionary with boxID as the key and the corresponding goal positions as values
	"""
	
	threads = []
	
	for agent in agents:
		t = threading.Thread(target=navigation, args=(agent, goals[agent], schedule[agent]))
		threads.append(t)
		t.start()
	
	for t in threads:
		t.join()
```

### navigation
针对于单个机器人沿着预定义的路径（`schedule`）移动机器人（`agent`）到目标点（`goal`）。
- 初始化 当前位置信息 / 当前路径点索引 / 距离阈值
- 导航主循环（是否到达目标位置）
	- 获取当前位置
	- 获取下一时刻位置
	- 检查路径是否完成， 
		- 未完成则更新next目标位置
		- 完成则停止机器人（设置轮速为0）并退出导航循环
	- 计算当前运动方向
		- `Orientation`：机器人当前的方向（从四元数转换为欧拉角）。
		- `goal_direction`：目标路径点相对于机器人的方向角。
	- 计算运动差
		- `theta`：当前方向与目标方向之间的角度差
		- `goal_direction - current_orientation`
	- 计算与下一个目标点的距离
		- `distance` 机器人当前位置和目标路径点之间的欧几里得距离。
	- 计算速度控制
		- `k1` 和 `k2`：线速度和角速度的增益参数
		- `linear`：线速度，主要用于沿目标方向前进
		- `angular`：角速度，用于调整方向以对准目标
	- 设置轮子速度
		- `rightWheelVelocity` 和 `leftWheelVelocity`：分别计算左右轮的速度，基于差速驱动模型
	- 为机器人设置轮子速度控制模式
```python
  

def navigation(agent, goal, schedule):

"""
Set velocity for robots to follow the path in the schedule. 

Args:
	agents: array containing the IDs for each agent
	schedule: dictionary with agent IDs as keys and the list of waypoints to the goal as values
	index: index of the current position in the path.

Returns:
	Leftwheel and rightwheel velocity.

"""
# 初始化
	basePos = p.getBasePositionAndOrientation(agent)
	index = 0
	dis_th = 0.4

# 导航主循环
	while(not checkPosWithBias(basePos[0], goal, dis_th)):
		# 当前位置
		basePos = p.getBasePositionAndOrientation(agent)
		# 获取下一时刻位置
		next = [schedule[index]["x"], schedule[index]["y"]]

		if(checkPosWithBias(basePos[0], next, dis_th)):
	
		index = index + 1
	
		if(index == len(schedule)):
		
			p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
			
			p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=1)
			
			break
			
	# 计算运动方向
		x = basePos[0][0]
		y = basePos[0][1]
		Orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
		goal_direction = math.atan2((schedule[index]["y"] - y), (schedule[index]["x"] - x))
	
		# 确保角度的表示范围统一
		if(Orientation < 0):
			Orientation = Orientation + 2 * math.pi
		if(goal_direction < 0):
			goal_direction = goal_direction + 2 * math.pi
		
	
	# 计算角度差
		theta = goal_direction - Orientation
		if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
			theta = theta + 2 * math.pi
		elif theta > 0 and abs(theta - 2 * math.pi) < theta:
			theta = theta - 2 * math.pi
	
	  
	# 计算与下一个目标点的距离
		current = [x, y]
		distance = math.dist(current, next)
	
	# 计算线速度和角速度
		k1, k2, A = 20, 5, 20
		linear = k1 * math.cos(theta)
		angular = k2 * theta
	
	
	#计算左右轮子的速度控制
		rightWheelVelocity = linear + angular
		leftWheelVelocity = linear - angular
	
	  
	# 控制轮子
		p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=1)
		
		p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=1)
	
		# time.sleep(0.001)
	
		print(agent, "here")


```


## output

cbs_schedule = read_cbs_output("final_challenge/cbs_output.yaml")

{1: [{'t': 0, 'x': 9, 'y': 9}, {'t': 1, 'x': 8, 'y': 9}, {'t': 2, 'x': 7, 'y': 9}, {'t': 3, 'x': 6, 'y': 9}, {'t': 4, 'x': 5, 'y': 9}, {'t': 5, 'x': 4, 'y': 9}, {'t': 6, 'x': 4, 'y': 8}, {'t': 7, 'x': 4, 'y': 7}, {'t': 8, 'x': 3, 'y': 7}, {'t': 9, 'x': 2, 'y': 7}, {'t': 10, 'x': 2, 'y': 6}, {'t': 11, 'x': 2, 'y': 5}, {'t': 12, 'x': 3, 'y': 5}, {'t': 13, 'x': 4, 'y': 5}, {'t': 14, 'x': 5, 'y': 5}, {'t': 15, 'x': 5, 'y': 4}, {'t': 16, 'x': 6, 'y': 4}, {'t': 17, 'x': 6, 'y': 3}, {'t': 18, 'x': 7, 'y': 3}, {'t': 19, 'x': 7, 'y': 2}, {'t': 20, 'x': 7, 'y': 1}, {'t': 21, 'x': 6, 'y': 1}, {'t': 22, 'x': 6, 'y': 0}],
2: [{'t': 0, 'x': 0, 'y': 9}, {'t': 1, 'x': 1, 'y': 9}, {'t': 2, 'x': 2, 'y': 9}, {'t': 3, 'x': 3, 'y': 9}, {'t': 4, 'x': 3, 'y': 8}, {'t': 5, 'x': 3, 'y': 7}, {'t': 6, 'x': 2, 'y': 7}, {'t': 7, 'x': 2, 'y': 6}, {'t': 8, 'x': 2, 'y': 5}, {'t': 9, 'x': 3, 'y': 5}, {'t': 10, 'x': 4, 'y': 5}, {'t': 11, 'x': 5, 'y': 5}, {'t': 12, 'x': 5, 'y': 4}, {'t': 13, 'x': 5, 'y': 3}, {'t': 14, 'x': 4, 'y': 3}, {'t': 15, 'x': 3, 'y': 3}, {'t': 16, 'x': 2, 'y': 3}, {'t': 17, 'x': 1, 'y': 3}, {'t': 18, 'x': 1, 'y': 2}, {'t': 19, 'x': 1, 'y': 1}, {'t': 20, 'x': 1, 'y': 0}, {'t': 21, 'x': 2, 'y': 0}, {'t': 22, 'x': 3, 'y': 0}]}

agent_name_to_box_id:{1: 71, 2: 72}
agent_box_ids:[71, 72]
box_id_to_goal:{71: [6, 0], 72: [3, 0]}
box_id_to_schedule:{71: [{'t': 0, 'x': 9, 'y': 9}, {'t': 1, 'x': 8, 'y': 9}, {'t': 2, 'x': 7, 'y': 9}, {'t': 3, 'x': 6, 'y': 9}, {'t': 4, 'x': 5, 'y': 9}, {'t': 5, 'x': 4, 'y': 9}, {'t': 6, 'x': 4, 'y': 8}, {'t': 7, 'x': 4, 'y': 7}, {'t': 8, 'x': 3, 'y': 7}, {'t': 9, 'x': 2, 'y': 7}, {'t': 10, 'x': 2, 'y': 6}, {'t': 11, 'x': 2, 'y': 5}, {'t': 12, 'x': 3, 'y': 5}, {'t': 13, 'x': 4, 'y': 5}, {'t': 14, 'x': 5, 'y': 5}, {'t': 15, 'x': 5, 'y': 4}, {'t': 16, 'x': 6, 'y': 4}, {'t': 17, 'x': 6, 'y': 3}, {'t': 18, 'x': 7, 'y': 3}, {'t': 19, 'x': 7, 'y': 2}, {'t': 20, 'x': 7, 'y': 1}, {'t': 21, 'x': 6, 'y': 1}, {'t': 22, 'x': 6, 'y': 0}], 72: [{'t': 0, 'x': 0, 'y': 9}, {'t': 1, 'x': 1, 'y': 9}, {'t': 2, 'x': 2, 'y': 9}, {'t': 3, 'x': 3, 'y': 9}, {'t': 4, 'x': 3, 'y': 8}, {'t': 5, 'x': 3, 'y': 7}, {'t': 6, 'x': 2, 'y': 7}, {'t': 7, 'x': 2, 'y': 6}, {'t': 8, 'x': 2, 'y': 5}, {'t': 9, 'x': 3, 'y': 5}, {'t': 10, 'x': 4, 'y': 5}, {'t': 11, 'x': 5, 'y': 5}, {'t': 12, 'x': 5, 'y': 4}, {'t': 13, 'x': 5, 'y': 3}, {'t': 14, 'x': 4, 'y': 3}, {'t': 15, 'x': 3, 'y': 3}, {'t': 16, 'x': 2, 'y': 3}, {'t': 17, 'x': 1, 'y': 3}, {'t': 18, 'x': 1, 'y': 2}, {'t': 19, 'x': 1, 'y': 1}, {'t': 20, 'x': 1, 'y': 0}, {'t': 21, 'x': 2, 'y': 0}, {'t': 22, 'x': 3, 'y': 0}]}

71和72的id值是使用turtlebot.udrf生成的
所以在MR中，

{1: [{'t': 1, 'x': 8, 'y': 9}, {'t': 2, 'x': 7, 'y': 9}, {'t': 3, 'x': 6, 'y': 9}, {'t': 4, 'x': 5, 'y': 9}, {'t': 5, 'x': 4, 'y': 9}, {'t': 6, 'x': 3, 'y': 9}, {'t': 7, 'x': 3, 'y': 8}, {'t': 8, 'x': 3, 'y': 7}, {'t': 9, 'x': 2, 'y': 7}, {'t': 10, 'x': 2, 'y': 6}, {'t': 11, 'x': 2, 'y': 5}, {'t': 12, 'x': 3, 'y': 5}, {'t': 13, 'x': 4, 'y': 5}, {'t': 14, 'x': 5, 'y': 5}, {'t': 15, 'x': 5, 'y': 4}, {'t': 16, 'x': 5, 'y': 3}, {'t': 17, 'x': 6, 'y': 3}, {'t': 18, 'x': 7, 'y': 3}, {'t': 19, 'x': 7, 'y': 2}, {'t': 20, 'x': 7, 'y': 1}, {'t': 21, 'x': 6, 'y': 1}, {'t': 22, 'x': 6, 'y': 0}], 2: [{'t': 1, 'x': 1, 'y': 9}, {'t': 2, 'x': 2, 'y': 9}, {'t': 3, 'x': 3, 'y': 9}, {'t': 4, 'x': 3, 'y': 8}, {'t': 5, 'x': 3, 'y': 7}, {'t': 6, 'x': 2, 'y': 7}, {'t': 7, 'x': 2, 'y': 6}, {'t': 8, 'x': 2, 'y': 5}, {'t': 9, 'x': 3, 'y': 5}, {'t': 10, 'x': 4, 'y': 5}, {'t': 11, 'x': 5, 'y': 5}, {'t': 12, 'x': 5, 'y': 4}, {'t': 13, 'x': 5, 'y': 3}, {'t': 14, 'x': 4, 'y': 3}, {'t': 15, 'x': 3, 'y': 3}, {'t': 16, 'x': 2, 'y': 3}, {'t': 17, 'x': 1, 'y': 3}, {'t': 18, 'x': 1, 'y': 2}, {'t': 19, 'x': 1, 'y': 1}, {'t': 20, 'x': 1, 'y': 0}, {'t': 21, 'x': 2, 'y': 0}, {'t': 22, 'x': 3, 'y': 0}]}