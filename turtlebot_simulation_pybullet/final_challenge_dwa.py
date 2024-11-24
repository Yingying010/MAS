import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import yaml

# 小车参数
class RobotParams:
    def __init__(self):
        self.max_speed = 1.0  # 最大速度 [m/s]
        self.min_speed = -0.5  # 最小速度 [m/s]
        self.max_yaw_rate = 40.0 * np.pi / 180.0  # 最大角速度 [rad/s]
        self.max_accel = 0.2  # 最大加速度 [m/s^2]
        self.max_delta_yaw_rate = 40.0 * np.pi / 180.0  # 最大角速度变化 [rad/s^2]
        self.v_resolution = 0.01  # 速度分辨率 [m/s]
        self.omega_resolution = 0.1 * np.pi / 180.0  # 角速度分辨率 [rad/s]
        self.dt = 0.1  # 时间间隔 [s]
        self.predict_time = 2.0  # 轨迹预测时间 [s]
        self.robot_radius = 0.5  # 小车半径 [m]

# 状态更新
def motion(state, v, omega, dt):
    state[0] += v * np.cos(state[2]) * dt
    state[1] += v * np.sin(state[2]) * dt
    state[2] += omega * dt
    state[3] = v
    state[4] = omega
    return state

# 动态窗口生成
def calc_dynamic_window(state, params):
    # 基于速度限制生成窗口
    vs = [params.min_speed, params.max_speed]
    ws = [-params.max_yaw_rate, params.max_yaw_rate]

    # 基于加速度生成窗口
    vs[0] = max(vs[0], state[3] - params.max_accel * params.dt)
    vs[1] = min(vs[1], state[3] + params.max_accel * params.dt)
    ws[0] = max(ws[0], state[4] - params.max_delta_yaw_rate * params.dt)
    ws[1] = min(ws[1], state[4] + params.max_delta_yaw_rate * params.dt)

    return vs, ws

# 轨迹生成
def predict_trajectory(state, v, omega, params):
    trajectory = np.array(state)
    time = 0
    while time <= params.predict_time:
        state = motion(state.copy(), v, omega, params.dt)
        trajectory = np.vstack((trajectory, state))
        time += params.dt
    return trajectory

# 目标代价函数
def calc_to_goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    return np.sqrt(dx ** 2 + dy ** 2)

# 障碍物代价函数
def calc_obstacle_cost(trajectory, obstacles, params):
    min_dist = float("inf")
    for pos in trajectory[:, :2]:
        for obs in obstacles:
            dist = np.linalg.norm(pos - obs)
            if dist < params.robot_radius:
                return float("inf")  # 碰撞
            if dist < min_dist:
                min_dist = dist
    return 1.0 / min_dist  # 距离越远代价越小



# 动态窗口评估
def dwa_control(state, params, goal, obstacles):
    vs, ws = calc_dynamic_window(state, params)
    best_u = [0.0, 0.0]
    min_cost = float("inf")
    best_trajectory = None

    for v in np.arange(vs[0], vs[1], params.v_resolution):
        for omega in np.arange(ws[0], ws[1], params.omega_resolution):
            trajectory = predict_trajectory(state, v, omega, params)
            to_goal_cost = calc_to_goal_cost(trajectory, goal)
            obstacle_cost = calc_obstacle_cost(trajectory, obstacles, params)
            total_cost = to_goal_cost + 1.0 * obstacle_cost  # 加权求和

            if total_cost < min_cost:
                min_cost = total_cost
                best_u = [v, omega]
                best_trajectory = trajectory

    return best_u, best_trajectory



def create_boundaries(length, width):
    """
        create rectangular boundaries with length and width

        Args:

        length: integer

        width: integer
    """
    for i in range(length):
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, -1, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [i, width, 0.5])
    for i in range(width):
        p.loadURDF("./final_challenge/assets/cube.urdf", [-1, i, 0.5])
        p.loadURDF("./final_challenge/assets/cube.urdf", [length, i, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, -1, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [length, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, width, 0.5])
    p.loadURDF("./final_challenge/assets/cube.urdf", [-1, -1, 0.5])


def create_env(yaml_file):
    """
    Creates and loads assets only related to the environment such as boundaries and obstacles.
    Robots are not created in this function (check `create_turtlebot_actor`).
    """

    obstacles = []
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
        obstacles.append(obstacle[0], obstacle[1])
    return env_params, obstacles

def create_agents(yaml_file):
    """
    Creates and loads turtlebot agents.

    Returns list of agent IDs and dictionary of agent IDs mapped to each agent's goal.
    """
    agent_box_ids = []
    box_id_to_goal = {}
    agent_name_to_box_id = {}
    with open(yaml_file, 'r') as f:
        try:
            agent_yaml_params = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            print(e)
        
    start_orientation = p.getQuaternionFromEuler([0,0,0])
    for agent in agent_yaml_params["agents"]:
        start_position = (agent["start"][0], agent["start"][1], 0)
        box_id = p.loadURDF("data/turtlebot.urdf", start_position, start_orientation, globalScaling=1)
        agent_box_ids.append(box_id)
        box_id_to_goal[box_id] = agent["goal"]
        agent_name_to_box_id[agent["name"]] = box_id
    return agent_box_ids, agent_name_to_box_id, box_id_to_goal, agent_yaml_params


def save_trajectory_to_yaml(file_path, trajectory_data):
    """
    将轨迹数据保存到YAML文件
    :param file_path: 文件路径
    :param trajectory_data: 轨迹数据列表，每个元素为字典 {'t': 时间, 'x': x坐标, 'y': y坐标}
    """
    with open(file_path, 'w') as file:
        yaml.dump(trajectory_data, file, default_flow_style=False)


# 主函数
def main():

    # 使用配置好的障碍物环境文件 (dimensions / obstacles)
    env_params, obstacles = create_env("final_challenge/env.yaml") 
    dimensions = env_params["map"]["dimensions"] 
    obstacles = env_params["map"]["obstacles"]
    print(obstacles)

    
    # 配置虚拟环境
    physics_client = p.connect(p.GUI, options='--width=1920 --height=1080 --mp4=Robot2_finalChanllege.mp4 --mp4fps=30')
    plane_id = p.loadURDF("plane.urdf")
    global env_loaded
    env_loaded = False



    '''
    初始化机器人属性
    state = np.array([0.0, 0.0, np.pi / 8, 0.0, 0.0])  # [x, y, yaw, v, omega]
    goal = np.array([10.0, 10.0])  # 目标点
    obstacles = np.array([[2.0, 2.0], [3.0, 6.0], [7.0, 8.0]])  # 障碍物位置

    '''
    # Create turtlebots
    params = RobotParams()
    state = np.array([0.0, 0.0, np.pi / 8, 0.0, 0.0])  # [x, y, yaw, v, omega]
    goal = np.array([10.0, 10.0])  # 目标点
    obstacles = np.array([[2.0, 2.0], [3.0, 6.0], [7.0, 8.0]])  # 障碍物位置

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(1)
    p.setGravity(0, 0, -10)
    p.resetDebugVisualizerCamera(cameraDistance=5.7, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[4.5, 4.5, 4])

    # 调用dwa执行路径规划
    # 保存轨迹数据的列表
    trajectory_data = []
    time_elapsed = 0  # 记录时间

    for _ in range(1000):
        plt.cla()
        # DWA控制
        u, trajectory = dwa_control(state, params, goal, obstacles)
        state = motion(state, u[0], u[1], params.dt)
        time_elapsed += params.dt  # 时间增加

        # 保存当前时刻数据到轨迹列表
        if trajectory is not None:
            trajectory_data.append({
                't': round(time_elapsed, 2),
                'x': round(trajectory[-1, 0], 2),  # 记录轨迹最后一点的x
                'y': round(trajectory[-1, 1], 2)   # 记录轨迹最后一点的y
            })

        # 绘图
        plt.plot(goal[0], goal[1], "r*", label="Goal")
        plt.plot(obstacles[:, 0], obstacles[:, 1], "ko", label="Obstacles")
        plt.plot(state[0], state[1], "bo", label="Robot")
        if trajectory is not None:
            plt.plot(trajectory[:, 0], trajectory[:, 1], "-g")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.1)

        # 检查是否到达目标
        if np.linalg.norm(state[:2] - goal) < params.robot_radius:
            print("Goal reached!")
            break

    plt.ioff()
    plt.show()

    # 保存到YAML文件
    save_trajectory_to_yaml('trajectory.yaml', trajectory_data)
    print("轨迹数据已保存到 trajectory.yaml 文件中。")

if __name__ == "__main__":
    main()
