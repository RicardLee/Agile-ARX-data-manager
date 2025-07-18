# -- coding: UTF-8
import os
import time
import numpy as np
import h5py
import argparse
import dm_env

import collections
from collections import deque
import threading
import queue

import rospy
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
import sys
import select
import cv2
import lmdb
import pickle
import imageio
from datetime import datetime
import subprocess

from pynput import keyboard


class KeyboardHandler:
    """Thread-safe keyboard handler using pynput for robust key detection."""

    def __init__(self):
        self.key_queue = queue.Queue()
        self.listener = None
        self.running = False

    def on_press(self, key):
        """Callback for key press events."""
        try:
            # Handle alphanumeric keys
            if hasattr(key, 'char') and key.char:
                self.key_queue.put(key.char.lower())
        except AttributeError:
            # Handle special keys if needed
            pass

    def start_listener(self):
        """Start the keyboard listener in a separate thread."""
        if not self.running:
            self.listener = keyboard.Listener(on_press=self.on_press)
            self.listener.start()
            self.running = True

    def stop_listener(self):
        """Stop the keyboard listener."""
        if self.running and self.listener:
            self.listener.stop()
            self.running = False

    def get_key_non_blocking(self):
        """Get a key press without blocking. Returns None if no key pressed."""
        try:
            return self.key_queue.get_nowait()
        except queue.Empty:
            return None

    def get_key_blocking(self):
        """Get a key press with blocking. Waits until a key is pressed."""
        return self.key_queue.get()

    def clear_queue(self):
        """Clear any pending key presses."""
        while not self.key_queue.empty():
            try:
                self.key_queue.get_nowait()
            except queue.Empty:
                break


def call_ros_service_sequence():
    """
    Execute the ROS service call sequence:
    1. Call go_zero_master services
    2. Wait 3 seconds
    3. Call restore_ms_mode services
    """
    try:
        print("🔧 Executing ROS service sequence...")

        # Step 1: Call go_zero_master services
        print("📞 Calling go_zero_master services...")
        rospy.wait_for_service('/can_left/go_zero_master', timeout=5.0)
        rospy.wait_for_service('/can_right/go_zero_master', timeout=5.0)

        go_zero_left = rospy.ServiceProxy('/can_left/go_zero_master', Trigger)
        go_zero_right = rospy.ServiceProxy('/can_right/go_zero_master', Trigger)

        response_left = go_zero_left()
        response_right = go_zero_right()
        print(f"   Left response: {response_left.message if response_left.success else 'Failed'}")
        print(f"   Right response: {response_right.message if response_right.success else 'Failed'}")
        print("✅ go_zero_master services called successfully")

        # Step 2: Wait 3 seconds
        print("⏳ Waiting 3 seconds...")
        time.sleep(3.0)

        # Step 3: Call restore_ms_mode services
        print("📞 Calling restore_ms_mode services...")
        rospy.wait_for_service('/can_left/restore_ms_mode', timeout=5.0)
        rospy.wait_for_service('/can_right/restore_ms_mode', timeout=5.0)

        restore_left = rospy.ServiceProxy('/can_left/restore_ms_mode', Trigger)
        restore_right = rospy.ServiceProxy('/can_right/restore_ms_mode', Trigger)

        response_left = restore_left()
        response_right = restore_right()
        print(f"   Left response: {response_left.message if response_left.success else 'Failed'}")
        print(f"   Right response: {response_right.message if response_right.success else 'Failed'}")
        print("✅ restore_ms_mode services called successfully")
        print("🎉 ROS service sequence completed!")

    except rospy.ServiceException as e:
        print(f"❌ ROS service call failed: {e}")
    except rospy.ROSException as e:
        print(f"❌ ROS error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error during service calls: {e}")


def get_git_commit_id(repo_path):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        commit_id = result.stdout.strip()
        return commit_id
    except subprocess.CalledProcessError as e:
        print(f"❌ 获取 commit id 失败: {e.stderr.strip()}")
        return None


# 保存数据函数
def save_data(args, timesteps, actions, dataset_path):
    # 数据字典
    data_size = len(actions)
    print("data_size", data_size)
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
        '/base_action': [],
    }

    # 相机字典  观察的图像
    for cam_name in args.camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
        if args.use_depth_image:
            data_dict[f'/observations/images_depth/{cam_name}'] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    # 动作长度 遍历动作
    while actions:
        action = actions.pop(0)   # 动作  当前动作
        ts = timesteps.pop(0)     # 奖励  前一帧

        # 往字典里面添值
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        data_dict['/base_action'].append(ts.observation['base_vel'])

        # 相机数据
        for cam_name in args.camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            if args.use_depth_image:
                data_dict[f'/observations/images_depth/{cam_name}'].append(ts.observation['images_depth'][cam_name])

    t0 = time.time()
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        root.attrs['sim'] = False
        root.attrs['compress'] = False

        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in args.camera_names:
            _ = image.create_dataset(cam_name, (data_size, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
        if args.use_depth_image:
            image_depth = obs.create_group('images_depth')
            for cam_name in args.camera_names:
                _ = image_depth.create_dataset(cam_name, (data_size, 480, 640), dtype='uint16',
                                             chunks=(1, 480, 640), )

        _ = obs.create_dataset('qpos', (data_size, 14))
        _ = obs.create_dataset('qvel', (data_size, 14))
        _ = obs.create_dataset('effort', (data_size, 14))
        _ = root.create_dataset('action', (data_size, 14))
        _ = root.create_dataset('base_action', (data_size, 2))

        for name, array in data_dict.items():  
            root[name][...] = array
    print(f'\033[32m\nSaving: {time.time() - t0:.1f} secs. %s \033[0m\n'%dataset_path)


def save_data_lmdb(args, timesteps, actions, dataset_path, max_size=1 << 40):  # max_size=1TB
    save_dir = dataset_path
    os.makedirs(save_dir, exist_ok=True)
    log_path_lmdb = os.path.join(save_dir, "lmdb")
    env = lmdb.open(log_path_lmdb, map_size=max_size)
    txn = env.begin(write=True)

    step_id_list = list(range(len(actions)))

    # 检查 action 和 qpos 每一维是否一大半为 0
    action_arr = np.array(actions)
    qpos_arr = np.array([timesteps[i].observation["qpos"] for i in step_id_list])

    action_zero_ratio = np.mean(action_arr == 0, axis=0)
    qpos_zero_ratio = np.mean(qpos_arr == 0, axis=0)
    
    # if np.any(action_zero_ratio > 0.9) or np.any(qpos_zero_ratio > 0.9):
    #     env.close()
    #     raise ValueError("Aborted: Some dimensions in action have > 60% zeros")

    meta_info = {}
    meta_info["keys"] = {}
    meta_info["camera_names"] = args.camera_names
    meta_info["keys"]["scalar_data"] = []
    meta_info["keys"]["images"] = {}
    meta_info["language_instruction"] = args.task_name
    meta_info['camera_height'] = args.camera_height
    meta_info["version"] = get_git_commit_id("/home/agilex/Desktop/Agile-ARX-data-manager")

    def put_scalar(name, values):
        meta_info["keys"]["scalar_data"].append(name.encode('utf-8'))
        txn.put(name.encode('utf-8'), pickle.dumps(values))

    # collect scalar data
    put_scalar("action", [actions[i] for i in step_id_list])
    put_scalar("base_action", [timesteps[i].observation["base_vel"] for i in step_id_list])
    put_scalar("qpos", [timesteps[i].observation["qpos"] for i in step_id_list])
    put_scalar("qvel", [timesteps[i].observation["qvel"] for i in step_id_list])
    put_scalar("effort", [timesteps[i].observation["effort"] for i in step_id_list])

    # collect images
    for cam_name in args.camera_names:
        rgb_key_prefix = f"observation/{cam_name}/color_image"
        meta_info["keys"]["images"][rgb_key_prefix] = []
        os.makedirs(os.path.join(save_dir, rgb_key_prefix), exist_ok=True)

        rgb_list = []
        for i in step_id_list:
            step_str = str(i).zfill(4)
            rgb_img = timesteps[i].observation["images"][cam_name]
            rgb_img_ = rgb_img[:,:,::-1]
            rgb_enc = cv2.imencode(".jpg", rgb_img_)[1]
            rgb_key = f"{rgb_key_prefix}/{step_str}".encode('utf-8')
            txn.put(rgb_key, pickle.dumps(rgb_enc))
            meta_info["keys"]["images"][rgb_key_prefix].append(rgb_key)
            rgb_list.append(rgb_img)

        imageio.mimsave(os.path.join(save_dir, rgb_key_prefix, "demo.mp4"), rgb_list, fps=30)

    meta_info["num_steps"] = len(step_id_list)
    txn.commit()
    env.close()

    with open(os.path.join(save_dir, "meta_info.pkl"), "wb") as f:
        pickle.dump(meta_info, f)

    print(f"\033[32mSaved LMDB to {save_dir}, {len(step_id_list)} steps\033[0m")


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.master_arm_right_deque = None
        self.master_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.args = args
        self.keyboard_handler = KeyboardHandler()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.master_arm_left_deque = deque()
        self.master_arm_right_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_left_deque) == 0 or self.master_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.master_arm_right_deque) == 0 or self.master_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.master_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_left_deque.popleft()
        master_arm_left = self.master_arm_left_deque.popleft()

        while self.master_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.master_arm_right_deque.popleft()
        master_arm_right = self.master_arm_right_deque.popleft()

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')
            top, bottom, left, right = 40, 40, 0, 0
            img_left_depth = cv2.copyMakeBorder(img_left_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_right_depth = cv2.copyMakeBorder(img_right_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')
        top, bottom, left, right = 40, 40, 0, 0
        img_front_depth = cv2.copyMakeBorder(img_front_depth, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()
        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def master_arm_left_callback(self, msg):
        if len(self.master_arm_left_deque) >= 2000:
            self.master_arm_left_deque.popleft()
        self.master_arm_left_deque.append(msg)

    def master_arm_right_callback(self, msg):
        if len(self.master_arm_right_deque) >= 2000:
            self.master_arm_right_deque.popleft()
        self.master_arm_right_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def init_ros(self):
        rospy.init_node('record_episodes', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.Subscriber(self.args.master_arm_left_topic, JointState, self.master_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.master_arm_right_topic, JointState, self.master_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)

    def process(self):
        timesteps = []
        actions = []

        prev_qpos = None
        count = 0
        record_started = True
        print("record not started, move arms to start")
        rate = rospy.Rate(self.args.frame_rate)
        print_flag = True
        save_flag = True

        # Start keyboard listener
        self.keyboard_handler.start_listener()

        while not rospy.is_shutdown():
            # Check for key presses using pynput (non-blocking)
            key = self.keyboard_handler.get_key_non_blocking()
            if key == 'q':
                key_input = 'q'
                break
            elif key == 's' or key == 'j':
                # Execute ROS service sequence for both 's' and 'j' keys
                call_ros_service_sequence()
                key_input = 's'  # Treat both as save
                break

            result = self.get_frame()
            if not result:
                if print_flag:
                    print_flag = False
                rate.sleep()
                continue
            print_flag = True

            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
            puppet_arm_left, puppet_arm_right, master_arm_left, master_arm_right, robot_base) = result

            cv2.imshow("img_front", img_front[:, :, ::-1])
            cv2.imshow("img_left", img_left[:, :, ::-1])
            cv2.imshow("img_right", img_right[:, :, ::-1])
            cv2.waitKey(1)

            qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
            base_vel = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z] if self.args.use_robot_base else [0.0, 0.0]

            if prev_qpos is None:
                prev_qpos = qpos
                rate.sleep()
                continue

            delta_qpos = qpos - prev_qpos
            should_start = (np.abs(delta_qpos) > 0.005).any() or (np.abs(base_vel) > 0.005).any()

            if not record_started:
                if should_start:
                    record_started = True
                    print("[INFO] Start recording from this frame.")
                else:
                    prev_qpos = qpos
                    rate.sleep()
                    continue

            prev_qpos = qpos
            count += 1

            image_dict = {
                self.args.camera_names[0]: img_front,
                self.args.camera_names[1]: img_left,
                self.args.camera_names[2]: img_right
            }

            obs = collections.OrderedDict()
            obs['images'] = image_dict
            if self.args.use_depth_image:
                image_dict_depth = {
                    self.args.camera_names[0]: img_front_depth,
                    self.args.camera_names[1]: img_left_depth,
                    self.args.camera_names[2]: img_right_depth
                }
                obs['images_depth'] = image_dict_depth
            obs['qpos'] = qpos
            obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
            obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
            obs['base_vel'] = base_vel

            step_type = dm_env.StepType.FIRST if count == 1 else dm_env.StepType.MID
            ts = dm_env.TimeStep(step_type=step_type, reward=None, discount=None, observation=obs)
            timesteps.append(ts)

            action = np.concatenate((np.array(master_arm_left.position), np.array(master_arm_right.position)), axis=0)
            actions.append(action)

            print("Frame data: ", count)
            
            rate.sleep()

        # if count > self.args.max_timesteps or count < self.args.min_timesteps:
        #     print(f"Recorded {count} frames. Reference frame: {self.args.min_timesteps} to {self.args.max_timesteps}.")
        #     save_flag = False

        # Stop keyboard listener
        self.keyboard_handler.stop_listener()

        print("len(timesteps): ", len(timesteps))
        print("len(actions)  : ", len(actions))
        return timesteps, actions, key_input, save_flag


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    
    parser.add_argument('--task_id', action='store', type=str, help='Task ID.',
                        default="0-0", required=False)

    parser.add_argument('--user_id', action='store', type=int, help='User ID.',
                        default=0, required=False)
    
    # parser.add_argument('--language_instruction', action='store', type=str, help='language_instruction.',
    #                     default="task description", required=False)
    
    parser.add_argument('--camera_height', action='store', type=str, help='camera_height.',
                        default="65.4", required=False)

    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=10000, required=False)
    
    parser.add_argument('--min_timesteps', action='store', type=int, help='Min_timesteps.',
                        default=50, required=False)

    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)

    parser.add_argument('--camera_names', action='store', type=str, help='camera_names',
                        default=['cam_high', 'cam_left_wrist', 'cam_right_wrist'], required=False)
    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--master_arm_left_topic', action='store', type=str, help='master_arm_left_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--master_arm_right_topic', action='store', type=str, help='master_arm_right_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom', required=False)
    
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)

    # Create a separate keyboard handler for the main loop
    main_keyboard_handler = KeyboardHandler()

    while not rospy.is_shutdown():
        timesteps, actions, key_input, save_flag = ros_operator.process()

        if save_flag == True:
            if key_input == 'q':
                print("\033[31m[INFO] Episode discarded. Not saved.\033[0m")
            elif key_input == 's':
                date_str = datetime.now().strftime("%Y%m%d")
                dataset_dir = os.path.join(args.dataset_dir, f"{args.task_name.replace(' ', '_')}/set{args.task_id}_collector{args.user_id}_{date_str}")
                dataset_dir = dataset_dir.replace(",", "_")  # 确保路径格式正确

                os.makedirs(dataset_dir, exist_ok=True)

                dataset_path_lmdb = os.path.join(dataset_dir, f"{str(args.episode_idx).zfill(7)}")
                save_data_lmdb(args, timesteps.copy(), actions.copy(), dataset_path_lmdb)

                print(f"\033[32m[INFO] Episode {args.episode_idx} saved.\033[0m")
                args.episode_idx += 1
        else:
            print("\033[31m[INFO] Episode discarded. Not saved.\033[0m")

        print("\n是否开始下一次采集？在窗口中按 y 继续，按 n 退出。")

        # Start keyboard listener for user input
        main_keyboard_handler.clear_queue()  # Clear any previous key presses
        main_keyboard_handler.start_listener()

        while True:
            # Use pynput for blocking key input (blocking behavior)
            key = main_keyboard_handler.get_key_blocking()
            if key == 'y':
                break
            elif key == 'n':
                print("采集终止。")
                main_keyboard_handler.stop_listener()
                return

        main_keyboard_handler.stop_listener()


if __name__ == '__main__':
    main()