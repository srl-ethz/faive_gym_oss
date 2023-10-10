from faive_gym.robot_hand import RobotHand
from isaacgymenvs.utils.torch_jit_utils import (
    quat_rotate_inverse,
    quat_conjugate,
    quat_mul,
    quat_to_angle_axis,
)
import torch

"""
not a serious task, just intended as an example of
- how to make a custom task that is derived from the RobotHand class
- how to create environments where the robot hand doesn't have a fixed base

The robot hand learns to crawl on the floor like a horror film - except it doesn't even learn to crawl right now, see if you can make it work...
Before, running, manually delete the "base" mesh in the MJCF file, since the fingers won't even touch the ground that much with the base.
"""

class Crawl(RobotHand):

    def check_termination(self):
        # override the function defined in RobotHand
        
        # check if hand is upside down (z axis is facing up)
        # do a bit hacky but easy to implement check
        hand_rot = self.hand_pose[:, 3:]
        angle, axis = quat_to_angle_axis(hand_rot)
        self.upside_down_buf = (torch.norm(axis[:, :2], dim=1) > 0.9) & (angle > 0.8 * 3.14)

        timeout_buf = self.progress_buf > self.max_episode_length - 1

        self.reset_goal_buf[:] = 0
        self.reset_buf[:] = self.upside_down_buf | timeout_buf

 
    def _init_buffers(self):
        super()._init_buffers()
        # move forward (in the direction of the fingertip, which is the y axis) 10 cm/s
        self.target_vel = torch.tensor([0., 0.07, -0.07], dtype=torch.float32, device=self.device)

    def _reward_crawl_penalty_upsidedown(self):
        return self.upside_down_buf

    def _reward_crawl_forward_vel(self):
        quat = self.hand_pose[:, 3:]
        lin_vel = self.hand_vel[:, :3]
        local_vel = quat_rotate_inverse(quat, lin_vel)
        vel_error = torch.sum(torch.square(self.target_vel - local_vel), dim=1)
        return torch.exp(-vel_error/0.05)

    def _observation_hand_quat(self):
        return self.hand_pose[:, 3:]
    
    def _observation_hand_vel(self):
        return self.hand_vel
    
