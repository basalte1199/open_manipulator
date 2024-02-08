from concurrent.futures import ThreadPoolExecutor
from math import exp
import os
import select
import sys
import rclpy
import time
import copy

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD
from ultralytics import YOLO
from std_msgs.msg import String
from typing import List
from sensor_msgs.msg import Image
from ultralytics.engine.results import Results




from open_manipulator_msgs.msg import KinematicsPose, OpenManipulatorState
from open_manipulator_msgs.srv import SetJointPosition, SetKinematicsPose
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
# from rclpy.executors import Executor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from threading import Timer

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

present_joint_angle = [0.0, 0.0, 0.0, 0.0 ,0.0]
goal_joint_angle = [0.0, 0.0, 0.0, 0.0 ,0.0]
prev_goal_joint_angle = [0.0, 0.0, 0.0, 0.0 ,0.0]
present_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
prev_goal_kinematics_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

debug = True
task_position_delta = 0.002  # meter
joint_angle_delta = 0.06  # radian
path_time = 1.0 # second
send_time = 3.0

usage = """
Control Your OpenManipulator!
---------------------------
Task Space Control:
         (Forward, X+)
              W                   Q (Upward, Z+)
(Left, Y+) A     D (Right, Y-)    Z (Downward, Z-)
              X 
        (Backward, X-)

Joint Space Control:
- Joint1 : Increase (Y), Decrease (H)
- Joint2 : Increase (U), Decrease (J)
- Joint3 : Increase (I), Decrease (K)
- Joint4 : Increase (O), Decrease (L)
- Gripper: Open     (G),    Close (F)

INIT : (1)
HOME : (2)
pick : (3)
side pick : (4)
home2 : (5)
low pick : (6)

CTRL-C to quit
"""

e = """
Communications Failed
"""


min_depth, max_depth = 200, 500 # mm
obj = []


class TeleopRsSub(Node):

    qos = QoSProfile(depth=10)
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)

    def __init__(self):
        super().__init__('teleop_realsense')
        key_value = ''

        #RealSenseの画像をsubしてくるためのコード
        self.bridge = CvBridge()
        self.realsense_subscription = self.create_subscription(RGBD, '/camera/camera/rgbd', self.listener_callback, self.qos)
        self.model = YOLO('yolov8n.pt')
        self.depth_image = None
        self.realsense_subscription


        # Create joint_states subscriber
        self.joint_state_subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            self.qos)
        self.joint_state_subscription

        # Create kinematics_pose subscriber
        self.kinematics_pose_subscription = self.create_subscription(
            KinematicsPose,
            'kinematics_pose',
            self.kinematics_pose_callback,
            self.qos)
        self.kinematics_pose_subscription

        # Create manipulator state subscriber
        self.open_manipulator_state_subscription = self.create_subscription(
            OpenManipulatorState,
            'states',
            self.open_manipulator_state_callback,
            self.qos)
        self.open_manipulator_state_subscription

        # Create Service Clients
        self.goal_joint_space = self.create_client(SetJointPosition, 'goal_joint_space_path')
        self.goal_task_space = self.create_client(SetKinematicsPose, 'goal_task_space_path')
        self.tool_control = self.create_client(SetJointPosition, 'goal_tool_control')
        self.goal_joint_space_req = SetJointPosition.Request()
        self.goal_task_space_req = SetKinematicsPose.Request()
        self.tool_control_req = SetJointPosition.Request()



        #realsense用のdef
    def listener_callback(self, msg):
        #cv2.imshow("Image window", self.mask_rgb(self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8"), self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")))

        self.rgb_image = self.bridge.imgmsg_to_cv2(msg.rgb, 'bgr8')
        self.depth_image = self.bridge.imgmsg_to_cv2(msg.depth, "passthrough")


        self.results_mask = self.model(self.bridge.imgmsg_to_cv2(msg.rgb, "bgr8"))
        #annotated_frame_mask = self.results_mask[0].plot()
        #cv2.imshow("Yolov8_masked", annotated_frame_mask)
        #cv2.waitKey(1)
        self.control_teleop()
   
    def mask_rgb(self, rgb, depth) -> np.ndarray:
        mask = (depth <= min_depth) | (depth >= max_depth)
        return np.where(np.broadcast_to(mask[:, :, None], rgb.shape), 0, rgb).astype(np.uint8)




    def control_teleop(self):
        tmp_image = copy.copy(self.rgb_image)
        results: List[Results] = self.model.predict(self.rgb_image, verbose=False, classes=[0], conf=0.3)


        for result in results:
            boxes = result.boxes.cpu().numpy()
            names = result.names
            print(names)
            if len(boxes.xyxy) == 0:
                continue
            #elif names != 'bottle':
            #    continue
            x1, y1, x2, y2 = map(int, boxes.xyxy[0][:4])
            cls_pred = boxes.cls[0]
            tmp_image = cv2.rectangle(tmp_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            tmp_image = cv2.putText(tmp_image, names[cls_pred], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            #print(names[cls_pred], self.depth_image[cy][cx]/10, "mm")
            obj.append([names[cls_pred],cx,cy,self.depth_image[cy][cx]/10])

        print(obj)
        #tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2BGR)
        #detection_result = self.bridge.cv2_to_imgmsg(tmp_image, "bgr8")
        cv2.imshow("yolo_result",tmp_image)
        cv2.waitKey(1)

    def send_goal_task_space(self):
        self.goal_task_space_req.end_effector_name = 'gripper'
        self.goal_task_space_req.kinematics_pose.pose.position.x = goal_kinematics_pose[0]
        self.goal_task_space_req.kinematics_pose.pose.position.y = goal_kinematics_pose[1]
        self.goal_task_space_req.kinematics_pose.pose.position.z = goal_kinematics_pose[2]
        self.goal_task_space_req.kinematics_pose.pose.orientation.w = goal_kinematics_pose[3]
        self.goal_task_space_req.kinematics_pose.pose.orientation.x = goal_kinematics_pose[4]
        self.goal_task_space_req.kinematics_pose.pose.orientation.y = goal_kinematics_pose[5]
        self.goal_task_space_req.kinematics_pose.pose.orientation.z = goal_kinematics_pose[6]
        self.goal_task_space_req.path_time = path_time

        try:
            send_goal_task = self.goal_task_space.call_async(self.goal_task_space_req)
        except Exception as e:
            self.get_logger().info('Sending Goal Kinematic Pose failed %r' % (e,))

    def send_goal_joint_space(self):
        self.goal_joint_space_req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4' ,'gripper']
        self.goal_joint_space_req.joint_position.position = [goal_joint_angle[0], goal_joint_angle[1], goal_joint_angle[2], goal_joint_angle[3], goal_joint_angle[4]]
        self.goal_joint_space_req.path_time = send_time

        try:
            send_goal_joint = self.goal_joint_space.call_async(self.goal_joint_space_req)
        except Exception as e:
            self.get_logger().info('Sending Goal Joint failed %r' % (e,))


    def send_tool_control_request(self):
        self.tool_control_req.joint_position.joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'gripper']
        self.tool_control_req.joint_position.position = [goal_joint_angle[0], goal_joint_angle[1], goal_joint_angle[2], goal_joint_angle[3], goal_joint_angle[4]]
        self.tool_control_req.path_time = path_time

        try:
            self.tool_control_result = self.tool_control.call_async(self.tool_control_req)

        except Exception as e:
            self.get_logger().info('Tool control failed %r' % (e,))


    def kinematics_pose_callback(self, msg):
        present_kinematics_pose[0] = msg.pose.position.x
        present_kinematics_pose[1] = msg.pose.position.y
        present_kinematics_pose[2] = msg.pose.position.z
        present_kinematics_pose[3] = msg.pose.orientation.w
        present_kinematics_pose[4] = msg.pose.orientation.x
        present_kinematics_pose[5] = msg.pose.orientation.y
        present_kinematics_pose[6] = msg.pose.orientation.z

    def joint_state_callback(self, msg):
        present_joint_angle[0] = msg.position[0]
        present_joint_angle[1] = msg.position[1]
        present_joint_angle[2] = msg.position[2]
        present_joint_angle[3] = msg.position[3]
        present_joint_angle[4] = msg.position[4]

    def open_manipulator_state_callback(self, msg):
        if msg.open_manipulator_moving_state == 'STOPPED':
            for index in range(0, 7):
                goal_kinematics_pose[index] = present_kinematics_pose[index]
            for index in range(0, 5):
                goal_joint_angle[index] = present_joint_angle[index]

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    #print_present_values()
    return key


def print_present_values():
    print('Joint Angle(Rad): [{:.6f}, {:.6f}, {:.6f}, {:.6f} ,{:.6f}]'.format(
        present_joint_angle[0],
        present_joint_angle[1],
        present_joint_angle[2],
        present_joint_angle[3],
        present_joint_angle[4]))
    print('Kinematics Pose(Pose X, Y, Z | Orientation W, X, Y, Z): {:.3f}, {:.3f}, {:.3f} | {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        present_kinematics_pose[0],
        present_kinematics_pose[1],
        present_kinematics_pose[2],
        present_kinematics_pose[3],
        present_kinematics_pose[4],
        present_kinematics_pose[5],
        present_kinematics_pose[6]))

def main():
    settings = None 
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)
    try:
        rclpy.init()
    except Exception as e:
        print(e)

    try:
        teleop_RsSub = TeleopRsSub()
    except Exception as e:
        print(e)

#ここは上のClass内の継承が効かないので、大体の処理は上の中で完結させるか一番上で変数を定義して上げる必要がある
    try:
        while(rclpy.ok()):
            rclpy.spin_once(teleop_RsSub)
            key_value = get_key(settings)
            
            if key_value == 'q':
                goal_kinematics_pose[2] = prev_goal_kinematics_pose[2] + task_position_delta
                teleop_RsSub.send_goal_task_space()
            elif key_value == 'z':
                goal_kinematics_pose[2] = prev_goal_kinematics_pose[2] - task_position_delta
                teleop_RsSub.send_goal_task_space()
            


            #メモ　絶対値はつけず、正負で動かす方向を指定する
            if int(len(obj)) > 1:
                dist1,dist2 = (1280//2) - int(obj[0][1]), 720//2 - int(obj[0][2])
            else:
                continue


            print(dist1,float(dist1/1200))
            
            goal_joint_angle[0] = float(dist1 /1300)
            teleop_RsSub.send_goal_joint_space()

            """
            if  int(dist1) < 640:
                print("aaaaaaaa")
                goal_joint_angle[0] = prev_goal_joint_angle[0] + joint_angle_delta
                teleop_RsSub.send_goal_joint_space()
            elif int(dist1) > 640:
                print("bbbbbbbbbbbbbbb")
                goal_joint_angle[0] = prev_goal_joint_angle[0] - joint_angle_delta
                teleop_RsSub.send_goal_joint_space()
            """    
            

            """
            if dist1 > 640:
                goal_joint_angle[0] = prev_goal_joint_angle[0] + joint_angle_delta
                teleop_RsSub.send_goal_joint_space()
            
            if dist1 < 640:
                goal_joint_angle[0] = prev_goal_joint_angle[0] - joint_angle_delta
                teleop_RsSub.send_goal_joint_space()    
            """ 
            #goal_kinematics_pose[1] = prev_goal_kinematics_pose[1] - task_position_delta
            #teleop_RsSub.send_goal_task_space() 
            obj.clear()

            
            if key_value == '\x03':
                break
            else:
                for index in range(0, 7):
                    prev_goal_kinematics_pose[index] = goal_kinematics_pose[index]
                for index in range(0, 4):
                    prev_goal_joint_angle[index] = goal_joint_angle[index]
            
            

    except Exception as e:
        print(e)

    finally:
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        teleop_RsSub.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()  
