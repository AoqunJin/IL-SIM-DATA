import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import CloseDoor, OpenDrawer, PickAndLift, PushButton, ReachTarget
from transformers import set_seed
from functions import _keypoint_discovery, _get_action

class ImitationLearning(object):

    def predict_action(self, batch):
        return np.random.uniform(size=(len(batch), 7))

    def behaviour_cloning_loss(self, ground_truth_actions, predicted_actions):
        return 1

############################### Config ###############################
# To use 'saved' demos, set the path below, and set live_demos=False
live_demos = False
DATASET = '/home/jinaoqun/workspace/data/train'
seed = 42
rlbench_scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
voxel_sizes = [100]
rotation_resolution = 5
crop_augmentation = False
############################### Config ###############################

set_seed(seed)
obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    action_mode=MoveArmThenGripper(arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    dataset_root=DATASET,
    obs_config=ObservationConfig(),
    headless=True)
env.launch()


task = env.get_task(OpenDrawer)

il = ImitationLearning()

demos = task.get_demos(10, live_demos=live_demos)  # -> List[List[Observation]]
demos = np.array(demos).flatten()


# An example of using the demos to 'train' using behaviour cloning loss.
for i in range(10):
    print("'training' iteration %d" % i)
    # demo = np.random.choice(demos, replace=False)
    demo = demos[i]
    episode_keypoints = _keypoint_discovery(demo)
    for k, keypoint in enumerate(episode_keypoints):
        if keypoint == 0: raise IndexError
        clip_points = demo[episode_keypoints[k-1]: keypoint] if k != 0 else demo[:keypoint]
        for point in clip_points:
            obs_tp1 = demo[keypoint]
            obs_tm1 = point
            trans_indicies, rot_grip_indicies, ignore_collisions, action, attention_coordinates = _get_action(
                obs_tp1, obs_tm1, rlbench_scene_bounds, voxel_sizes, rotation_resolution, crop_augmentation
            )
            item = {
                "feature": obs_tm1,
                "label": [trans_indicies, rot_grip_indicies, obs_tp1.gripper_pose]
            }
            # Add
    item = {
        "feature": demo[keypoint],
        "label": [trans_indicies, rot_grip_indicies, obs_tp1.gripper_pose]
    }
    
    # demo_images = [obs.left_shoulder_rgb for obs in demo]
    # predicted_actions = il.predict_action(demo_images)
    # ground_truth_actions = [obs.joint_velocities for obs in demo]
    # loss = il.behaviour_cloning_loss(ground_truth_actions, predicted_actions)
    # _keypoint_discovery(demo)
    
    break

print('Done')
env.shutdown()
