import pybullet as p
import pybullet_data
import math
import matplotlib.pyplot as plt


def check_pos(Pos, goal, bias):
    """
        Check if pos is at goal with bias

        Args:

        Pos: Position to be checked, [x, y]

        goal: goal position, [x, y]

        bias: bias allowed

        Returns:

        True if pos is at goal, False otherwise
    """
    if goal[0] + bias > Pos[0] > goal[0] - bias and goal[1] + bias > Pos[1] > goal[1] - bias:
        return True
    else:
        return False


def goto(agent, goal_x, goal_y, actual_trajectory):
    dis_th = 0.1
    basePos = p.getBasePositionAndOrientation(agent)
    current_x = basePos[0][0]
    current_y = basePos[0][1]
    pos = [current_x, current_y]
    goal = [goal_x, goal_y]

    # PID controller parameters
    Kp = 20.0  # Proportional gain
    Ki = 0.0  # Integral gain
    Kd = 10.0  # Derivative gain

    # Variables for PID control
    integral = 0
    previous_error = 0

    while not check_pos(pos, goal, dis_th):
        basePos = p.getBasePositionAndOrientation(agent)
        current_x = basePos[0][0]
        current_y = basePos[0][1]
        pos = [current_x, current_y]
        actual_trajectory.append((current_x, current_y))


        # Check for keyboard input to stop the simulation
        events = p.getKeyboardEvents()
        if ord('q') in events and events[ord('q')] & p.KEY_WAS_TRIGGERED:
            print("Exit command received, stopping simulation.")
            break  # Exit the loop and stop the simulation



        current_orientation = list(p.getEulerFromQuaternion(basePos[1]))[2]
        goal_direction = math.atan2((goal_y - current_y), (goal_x - current_x))

        if current_orientation < 0:
            current_orientation = current_orientation + 2 * math.pi
        if goal_direction < 0:
            goal_direction = goal_direction + 2 * math.pi

        theta = goal_direction - current_orientation
        if theta < 0 and abs(theta) > abs(theta + 2 * math.pi):
            theta = theta + 2 * math.pi
        elif theta > 0 and abs(theta - 2 * math.pi) < theta:
            theta = theta - 2 * math.pi


        # PID calculations
        error = theta
        integral += error
        derivative = error - previous_error
        output = Kp * error + Ki * integral + Kd * derivative
        previous_error = error


        k_linear = 10
        linear = k_linear * math.cos(theta)
        angular = output  # Use PID output to adjust angular velocity

        rightWheelVelocity = linear + angular
        leftWheelVelocity = linear - angular

        p.setJointMotorControl2(agent, 0, p.VELOCITY_CONTROL, targetVelocity=leftWheelVelocity, force=10)
        p.setJointMotorControl2(agent, 1, p.VELOCITY_CONTROL, targetVelocity=rightWheelVelocity, force=10)

def load_waypoints(file_path):
    """从文件中加载路径点"""
    waypoints = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split(','))
            waypoints.append((x, y))
    return waypoints

def navigate_waypoints(agent, waypoints):
    """按照路径点依次导航"""
    actual_trajectory = []  # List to store actual trajectory
    for waypoint in waypoints:
        goal_x, goal_y = waypoint
        print(f"Navigating to waypoint: {goal_x}, {goal_y}")
        goto(agent, goal_x, goal_y, actual_trajectory)
    return actual_trajectory


def plot_trajectory(actual_trajectory, waypoints):
    # Extract x and y coordinates of actual trajectory and waypoints
    actual_x, actual_y = zip(*actual_trajectory)
    waypoint_x, waypoint_y = zip(*waypoints)

    plt.figure(figsize=(8, 6))
    plt.plot(actual_x, actual_y, label="Actual Trajectory", color='blue', marker='o')
    plt.plot(waypoint_x, waypoint_y, label="Target Waypoints", color='red', linestyle='--', marker='x')

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Actual vs Target Trajectory")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('trajectory_comparison.png', format='png')
    plt.show()


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
p.setRealTimeSimulation(1)
p.setGravity(0, 0, -10)

startPosition = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

boxId = p.loadURDF("data/turtlebot.urdf", startPosition, startOrientation, globalScaling=1)
p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-89.9,
                             cameraTargetPosition=[0, 0, 0])

# goto(boxId, 1, 1)

waypoints = load_waypoints("waypoints.txt")
actual_trajectory = navigate_waypoints(boxId, waypoints)

plot_trajectory(actual_trajectory, waypoints)


