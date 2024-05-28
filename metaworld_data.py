import os
import pickle
import shutil
from typing import Tuple, List, Type, Optional, Any
import multiprocessing
import argparse
import logging
import math

from tqdm import tqdm
import numpy as np
import cv2
from metaworld.policies import *
import metaworld.envs.mujoco.env_dict as _env_dict


def get_policy_names(env_names: List[str]) -> List[str]:
    """Generate policy names based on environment names.

    Args:
        env_names (List[str]): List of environment names.

    Returns:
        List[str]: List of corresponding policy names.
    """
    policy_names = []
    for env_name in env_names:
        base = "Sawyer"
        res = env_name.split("-")
        for substr in res:
            base += substr.capitalize()
        policy_name = base + "Policy"
        if policy_name == "SawyerPegInsertSideV2Policy":
            policy_name = "SawyerPegInsertionSideV2Policy"
        policy_names.append(policy_name)
    
    return policy_names


def get_all_envs(env_names: List[str] = None, n_envs: int = 50) -> Tuple[List[str], List[Type]]:
    """Get Metaworld environments.

    Args:
        env_names (List[str]): List of environment names.
        n_envs (int): Use number of environments

    Returns:
        Tuple: A tuple containing a list of environment names and a list of environment classes.
    """
    envs = []
    env_names = ([] if env_names is None else env_names)
    # get env names
    if env_names == []:
        for env_name in _env_dict.MT50_V2:
            env_names.append(env_name)
    if n_envs > len(env_names):
        raise ValueError(f"n_envs={n_envs} should less then "
                         f"len(env_names)={len(env_names)} with env_names={env_names}")
    # get envs
    for env_name in env_names:
        envs.append(_env_dict.MT50_V2[env_name])

    return env_names[:n_envs], envs[:n_envs]


def make_folder(folder: Optional[str] = None):
    """Create a folder and initialize the data directory.

    Args:
        folder (Optional[str]): Name of the folder to be created within the data directory.
                                If None, the data directory is initialized.
    """
    # Make folder
    if folder is not None:
        path = os.path.join(args.data_dir, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    # Init data dir
    elif not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        

def save_state(tag: str, data: Any):
    """Save data to a binary file.

    Args:
        tag (str): Name of the file (without extension) to save the data.
        data: Data to be saved.
    """
    # with open(os.path.join(args.data_dir, f"state/{tag}.bin"), "wb") as f:
    #     pickle.dump(data, f)

    for key in ["obss", "acts", "rews"]:
        np.savez_compressed(
            os.path.join(args.data_dir, f"state/{tag}-{key}"),
            np.array(data[key])
        )


def write_video(tag, fps, res):
    return cv2.VideoWriter(
        os.path.join(args.data_dir, f"video/{tag}.avi"),
        cv2.VideoWriter_fourcc("M","J","P","G"),
        fps, res
    )


def collect_trail(env, policy, env_name: str, out_state: bool, use_rgb: bool, 
                  resolution: tuple = (224, 224), camera_name: str = "corner", out_video: bool = False,
                  num_trail: int = 200, num_step: int = 501):
    """Collect data for a single environment using the given policy."""
    flip = False  # TODO flip automatic
    for trail in range(num_trail):
        obs = env.reset()
        tag = env_name + f"-{trail}"
        if out_state:
            obss, acts, rews = [], [], []
        if out_video:
            writer = write_video(tag, env.metadata["video.frames_per_second"], resolution)

        for i in range(num_step):
            act = policy.get_action(obs)
            
            # obs
            if out_state:
                if use_rgb:
                    obs = env.sim.render(*resolution, mode="offscreen", camera_name=camera_name)[:,:,::-1]
                    if flip: obs = cv2.rotate(obs, cv2.ROTATE_180)
                obss.append(obs)
            
            # video
            if out_video:
                if not out_state or not use_rgb:  # render first time
                    obs = env.sim.render(*resolution, mode="offscreen", camera_name=camera_name)[:,:,::-1]

                if flip: obs = cv2.rotate(obs, cv2.ROTATE_180)
                writer.write(obs)     
                       
            obs, rew, done, info = env.step(act+0.1*np.random.randn(4,))
            
            # action and reward
            if out_state:
                acts.append(act)
                rews.append(rew)

            if info['success'] or done:
                break
            
        if out_state:
            save_state(tag, {"obss": obss, "acts": acts, "rews": rews})
            del obss, acts, rews
        if out_video:
            del writer

def collect_demos(env_names: List[str], envs: List[Type], policy_names: List[str], 
                  out_state: bool, use_rgb: bool, resolution: tuple = (224, 224), 
                  camera_name: str = "corner", out_video: bool = False, 
                  num_trail: int = 200, num_step: int = 501, num_workers: int = 1) -> None:
    """Parallel collect demonstration data for the specified environments using the given policies."""
    if out_state:
        make_folder("state")
    if out_video:
        make_folder("video")

    # process bar 
    pbar = tqdm(total=len(env_names))
    pbar.set_description(f"Total {len(env_names)} environment")
    update = lambda *x: pbar.update()
    
    pool = multiprocessing.Pool(processes=num_workers)

    for i in range(len(env_names)):
        env_name = env_names[i]
        env = envs[i]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        
        policy_name = policy_names[i]
        policy = globals()[policy_name]()
        
        # parallel    
        pool.apply_async(
            func=collect_trail,
            args=(env, policy, env_name, 
                  out_state, use_rgb, resolution, camera_name, 
                  out_video, num_trail, num_step),
            callback=update
        )
        # collect_trail(env, policy, env_name, out_state, use_rgb, resolution, camera_name, 
        #           out_video, num_trail, num_step)

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect meta-world demonstration.")

    parser.add_argument("--env_names", nargs="+", help="List of environment names.")
    parser.add_argument("--n_envs", type=int, default=50, help="Number of collect environment.")
    parser.add_argument("--out_state", action="store_true", help="Flag indicating whether to output state.")
    parser.add_argument("--use_rgb", action="store_true", help="Flag indicating whether to use RGB observations.")
    parser.add_argument("--resolution", nargs=2, type=int, default=[224, 224], help="Resolution of RGB observations.")
    parser.add_argument("--camera_name", choices=["corner", "topview", "behindGripper", "gripperPOV"], default="corner", help="Name of the camera view.") 
    parser.add_argument("--out_video", action="store_true", help="Flag indicating whether to output video.")
    parser.add_argument("--num_trail", type=int, default=2000, help="Number of trials for data collection.")
    parser.add_argument("--num_step", type=int, default=501, help="Number of steps per trial.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--data_dir", type=str, default="metaworld_data", help="Directory to store collected data.")
    
    args = parser.parse_args()

    if not (args.out_state or args.out_video):
        raise ValueError("Should at list output one of the state or video!")
    
    env_names, envs = get_all_envs(args.env_names, args.n_envs)
    policy_names = get_policy_names(env_names)

    logging.basicConfig(level=logging.DEBUG,  # level DEBUG
                        format="%(asctime)s - %(levelname)s - %(message)s")
    
    logging.info(f"n_envs {args.n_envs}")
    logging.info(f"env_names {env_names}")
    logging.info(f"policy_names {policy_names}")
    logging.info(f"out_state {args.out_state}")   
    logging.info(f"use_rgb {args.use_rgb}")   
    logging.info(f"resolution {tuple(args.resolution)}")
    logging.info(f"camera_name {args.camera_name}")
    logging.info(f"out_video {args.out_video}")
    logging.info(f"num_trail {args.num_trail}")
    logging.info(f"num_step {args.num_step}")
    logging.info(f"num_workers {args.num_workers}")
    logging.info(f"data_dir {args.data_dir}")

    collect_demos(env_names, envs, policy_names, out_state=args.out_state, 
                  use_rgb=args.use_rgb, resolution=tuple(args.resolution), 
                  camera_name=args.camera_name, out_video=args.out_video, 
                  num_trail=args.num_trail, num_step=args.num_step, 
                  num_workers=args.num_workers)
    
    logging.info("All finished!")
    