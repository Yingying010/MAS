import pybullet as p
import time
import numpy as np

# 初始化PyBullet模拟环境
p.connect(p.GUI)
p.setGravity(0, 0, -9.8)

# 加载地面和两个小车（使用PyBullet自带的URDF模型）

# 加载平面URDF模型
plane_id = p.loadURDF("plane.urdf")
robot1_id = p.loadURDF("r2d2.urdf", basePosition=[0, 0, 0.1])  # 机器人1
robot2_id = p.loadURDF("r2d2.urdf", basePosition=[1, 0, 0.1])  # 机器人2

# 设置初始速度
p.setJointMotorControlArray(robot1_id, [0, 1, 2, 3], p.VELOCITY_CONTROL, targetVelocities=[1, 1, 1, 1])
p.setJointMotorControlArray(robot2_id, [0, 1, 2, 3], p.VELOCITY_CONTROL, targetVelocities=[1, 1, 1, 1])

# 模拟主循环
for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)  # 控制模拟的步长

def calculate_path(start, end, steps=100):
    return np.linspace(start, end, steps)

# 定义两个机器人路径
robot1_start = np.array([0, 0])
robot1_end = np.array([5, 5])
robot2_start = np.array([1, 0])
robot2_end = np.array([6, 5])

robot1_path = calculate_path(robot1_start, robot1_end)
robot2_path = calculate_path(robot2_start, robot2_end)

def detect_conflict(path1, path2, threshold=0.5):
    conflicts = []
    for i, pos1 in enumerate(path1):
        for j, pos2 in enumerate(path2):
            if np.linalg.norm(pos1 - pos2) < threshold:
                conflicts.append((i, j))
    return conflicts

# 检测冲突
conflicts = detect_conflict(robot1_path, robot2_path)
print(f"Detected conflicts: {conflicts}")

def resolve_conflict(conflicts, robot1_path, robot2_path, robot1_id, robot2_id):
    # 模拟LLM的冲突解决：根据优先级调整路径
    for conflict in conflicts:
        step1, step2 = conflict
        print(f"Conflict detected at step {step1} and {step2}")
        
        # 假设机器人1的任务优先级高（ID较小的优先）
        # 将机器人2的路径偏移，以避开冲突
        robot2_path[step2:] = robot2_path[step2:] + np.array([0.5, 0])  # 偏移机器人2路径
        print(f"Resolving conflict by shifting Robot 2's path.")
        
        # 重新计算机器人2的位置
        p.resetBasePositionAndOrientation(robot2_id, robot2_path[0].tolist() + [0.1], [0, 0, 0, 1])
        p.setJointMotorControlArray(robot2_id, [0, 1, 2, 3], p.VELOCITY_CONTROL, targetVelocities=[1, 1, 1, 1])

# 解决冲突
if conflicts:
    resolve_conflict(conflicts, robot1_path, robot2_path, robot1_id, robot2_id)

def move_robot(robot_id, path):
    for pos in path:
        p.resetBasePositionAndOrientation(robot_id, pos.tolist() + [0.1], [0, 0, 0, 1])
        p.stepSimulation()
        time.sleep(1./240.)

# 执行路径运动
move_robot(robot1_id, robot1_path)
move_robot(robot2_id, robot2_path)
