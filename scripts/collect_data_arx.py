# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))

import time
import h5py
import argparse
import rospy
import cv2
import yaml
import threading
import pyttsx3
from datetime import datetime
import numpy as np
import lmdb
import pickle
import imageio

from copy import deepcopy

from utils.ros_operator import RosOperator

np.set_printoptions(linewidth=200)


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None


def collect_information(args, ros_operator):
    timesteps = []
    actions = []
    actions_eef = []
    action_bases = []
    count = 0
    rate = rospy.Rate(args.frame_rate)

    # 初始化机器人基础位置
    ros_operator.init_robot_base_pose()

    gripper_idx = [6, 13]
    gripper_close = 3.0
    prev_qpos = None
    record_started = False
    key_input = None

    while not rospy.is_shutdown():
        obs_dict = ros_operator.get_observation(ts=count)
        action_dict = ros_operator.get_action()

        # 同步帧检测
        if obs_dict is None or action_dict is None:
            print("Synchronization frame")
            rate.sleep()

            continue

        # 获取动作和观察值
        action = deepcopy(obs_dict['qpos'])
        action_eef = deepcopy(action_dict['eef'])
        action_base = obs_dict['robot_base']


        # 夹爪动作处理
        for idx in gripper_idx:
            action[idx] = 0 if action[idx] < gripper_close else action[idx]
        action_eef[6] = 0 if action_eef[6] < gripper_close else action_eef[6]
        action_eef[13] = 0 if action_eef[13] < gripper_close else action_eef[13]


        if prev_qpos is None:
            prev_qpos = deepcopy(action)
            rate.sleep()
            continue

        delta_qpos = action - prev_qpos
        should_start = (np.abs(delta_qpos) > 0.001).any() or (np.abs(action_base) > 0.01).any()

        if not record_started:
            if should_start:
                record_started = True
                print("[INFO] Start recording from this frame.")
            else:
                prev_qpos = deepcopy(action)
                rate.sleep()
                continue

        # 收集数据
        timesteps.append(obs_dict)
        actions.append(action)
        actions_eef.append(action_eef)
        action_bases.append(action_base)

        #vis
        cv2.imshow("img_front", obs_dict['images']['head'][:,:,::-1])
        cv2.imshow("img_left", obs_dict['images']['left_wrist'][:,:,::-1])
        cv2.imshow("img_right", obs_dict['images']['right_wrist'][:,:,::-1])
        cv2.waitKey(1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            key_input = 's'
            break
        elif key == ord('q'):
            key_input = 'q'
            break

        count += 1
        print(f"Frame data: {count}")

        if rospy.is_shutdown():
            exit(-1)

        rate.sleep()

    print(f"\nlen(timesteps): {len(timesteps)}")
    print(f"len(actions)  : {len(actions)}")

    return timesteps, actions, actions_eef, action_bases, key_input


def save_data_lmdb(args, timesteps, actions, actions_eef, actions_bases, dataset_path, max_size=1 << 40):  # max_size=1TB
    save_dir = dataset_path
    os.makedirs(save_dir, exist_ok=True)
    log_path_lmdb = os.path.join(save_dir, "lmdb")
    env = lmdb.open(log_path_lmdb, map_size=max_size)
    txn = env.begin(write=True)

    meta_info = {}
    meta_info["keys"] = {}
    meta_info["camera_names"] = args.camera_names
    meta_info["keys"]["scalar_data"] = []
    meta_info["keys"]["images"] = {}
    meta_info["language_instruction"] = args.task_name
    step_id_list = list(range(len(actions)))

    def put_scalar(name, values):
        meta_info["keys"]["scalar_data"].append(name.encode('utf-8'))
        txn.put(name.encode('utf-8'), pickle.dumps(values))

    # collect scalar data
    put_scalar("action", [actions[i] for i in step_id_list])
    put_scalar("action_bases", [actions_bases[i] for i in step_id_list])
    put_scalar("action_eef", [actions_eef[i] for i in step_id_list])
    put_scalar("/observations/qpos", [timesteps[i]["qpos"] for i in step_id_list])
    put_scalar("/observations/qvel", [timesteps[i]["qvel"] for i in step_id_list])
    put_scalar("/observations/effort", [timesteps[i]["effort"] for i in step_id_list])
    put_scalar("/observations/eef", [timesteps[i]["eef"] for i in step_id_list])
    put_scalar("/observations/robot_base", [timesteps[i]["robot_base"] for i in step_id_list])


    # collect images
    for cam_name in args.camera_names:
        rgb_key_prefix = f"observation/{cam_name}/color_image"
        meta_info["keys"]["images"][rgb_key_prefix] = []
        os.makedirs(os.path.join(save_dir, rgb_key_prefix), exist_ok=True)

        rgb_list = []
        for i in step_id_list:
            step_str = str(i).zfill(4)
            rgb_img = timesteps[i]["images"][cam_name]
            # rgb_img = rgb_img[:,:,::-1]
            rgb_enc = cv2.imencode(".jpg", rgb_img)[1]
            rgb_key = f"{rgb_key_prefix}/{step_str}".encode('utf-8')
            txn.put(rgb_key, pickle.dumps(rgb_enc))
            meta_info["keys"]["images"][rgb_key_prefix].append(rgb_key)
            rgb_list.append(rgb_img)

        imageio.mimsave(os.path.join(save_dir, rgb_key_prefix, "demo.mp4"), rgb_list, fps=args.frame_rate)

    meta_info["num_steps"] = len(step_id_list)
    txn.commit()
    env.close()

    with open(os.path.join(save_dir, "meta_info.pkl"), "wb") as f:
        pickle.dump(meta_info, f)

    print(f"\033[32mSaved LMDB to {save_dir}, {len(step_id_list)} steps\033[0m")


def main(args):
    config = load_yaml(args.config)
    ros_operator = RosOperator(args, config, in_collect=True)

    current_episode = args.episode_idx

    while not rospy.is_shutdown():
        print(f"Start to record episode {current_episode}")
        timesteps, actions, actions_eef, action_bases, key_input = collect_information(args, ros_operator)  # ros_operator.process()

        if key_input == 'q':
            print("\033[31m[INFO] Episode discarded. Not saved.\033[0m")
        
        elif key_input == 's':
            date_str = datetime.now().strftime("%Y%m%d")
            dataset_dir = os.path.join(args.dataset_dir, f"{args.task_name.replace(' ', '_')}/set{args.task_id}_collector{args.user_id}_{date_str}")
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_path_lmdb = os.path.join(dataset_dir, f"{str(current_episode).zfill(7)}")
            
            # threading.Thread(target=save_data, args=(args, timesteps, actions, actions_eef, action_bases, dataset_path,)
            #                 ).start()  # 执行指令单独的线程,，可以边说话边执行，多线程操作
            
            threading.Thread(target=save_data_lmdb, args=(args, timesteps, actions, actions_eef, action_bases, dataset_path_lmdb)
                            ).start()  # 执行指令单独的线程,，可以边说话边执行，多线程操作
            print(f"\033[32m[INFO] Episode {current_episode} saved.\033[0m")
            current_episode = current_episode + 1
        
        print("\n是否开始下一次采集？在窗口中按 y 继续，按 n 退出。")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('y'):
                break
            elif key == ord('n'):
                print("采集终止。")
                return



def parse_arguments(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="arx_dummy", required=False)
    
    parser.add_argument('--task_id', action='store', type=str, help='Task ID.',
                        default="0-0", required=False)

    parser.add_argument('--user_id', action='store', type=int, help='User ID.',
                        default=0, required=False)

    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    
    parser.add_argument('--episode_idx', type=int, default=0, help='episode index')
    parser.add_argument('--frame_rate', type=int, default=30, help='frame rate')

    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=10000, required=False)
    
    parser.add_argument('--min_timesteps', action='store', type=int, help='Min_timesteps.',
                        default=50, required=False)

    # 配置文件
    parser.add_argument('--config', type=str,
                        default=Path.joinpath(ROOT, 'data/config.yaml'),
                        help='config file')

    # 图像处理选项
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist'],
                        default=['head', 'left_wrist', 'right_wrist'], help='camera names')
    parser.add_argument('--use_depth_image', action='store_true', help='use depth image')
    parser.add_argument('--is_compress', action='store_true', help='compress image')

    # 机器人选项
    parser.add_argument('--use_base', action='store_true', help='use robot base')

    # 数据采集选项
    parser.add_argument('--key_collect', action='store_true', help='use key collect')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
