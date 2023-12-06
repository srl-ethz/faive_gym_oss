# robot_hand.py
# Generic class for robot hand RL environments
# Copyright 2023 Soft Robotics Lab, ETH Zurich
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import (
    to_torch,
    quat_mul,
    quat_conjugate,
    tensor_clamp,
    scale,
    unscale,
    quat_apply,
    torch_rand_float,
    quat_from_angle_axis,
)
import numpy as np
import os
import datetime
import torch



from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import (
    quat_to_angle_axis,
)

"""
Define the IsaacGym environment for a somewhat generic robot hand
Different versions of the Faive Hand can be implemented by specifying the model in the config yaml file, as well as other similar robot hands
Despite this goal, currently it is somewhat hardcoded specifically to the Faive Hand- if some features are too specific to the Faive Hand, they can be moved to a child class

if the task or the robot deviates too much from what is implemented here, consider creating a child class
"""


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class RobotHand(VecTask):
    def __init__(
        self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render
    ):
        """
        Args:
            cfg: Configuration object
            sim_params (gymapi.SimParams): Simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            sim_device (str): "cuda" or "cpu"
            headless: True if running headless
        """
        self.cfg = cfg
        self._parse_cfg(self.cfg)
        # overwrite config to have the correct number of observations for actor and critic
        self.cfg["env"]["numObservations"], self.cfg["env"]["numStates"] = self._prepare_observations()

        # define names of relevant body parts
        self.hand_base_name = "root"
        self.sim_device_id = sim_device

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        self.control_dt = self.control_freq_inv * self.dt  # dt for policy
        self.max_episode_length = np.ceil(cfg["env"]["episode_length_s"] / self.control_dt)
        
        if not self.headless:
            # set camera
            self.cam_pos = gymapi.Vec3(
                *self.cfg["visualization"]["camera_pos_start"]
            )
            self.cam_target = gymapi.Vec3(
                *self.cfg["visualization"]["camera_target_start"]
            )
            # set camera speed direction
            self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)
            if self.cfg["visualization"]["move_camera"]:
                self.cam_movement_per_step = \
                    gymapi.Vec3(*self.cfg["visualization"]["camera_movement_vector"])
        self._init_buffers()
        self._prepare_reward_function()
        if self.cfg["logging"]["rt_plt"]:
            self._prepare_logged_functions()

    def _init_buffers(self):
        """
        Initialize buffers (torch tensors) that will contain simulation states
        """
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        # fetch the data from the sim
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # save as appropriately shaped torch tensors
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
    
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
            self.num_envs, self.num_hand_dofs
        )

        # next, define new member variables that make it easier to access the state tensors
        # if arrays are not used for indexing, the sliced tensors will be views of the original tensors, and thus their values will be automatically updated
        self.hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_hand_dofs
        ]
        
        self.hand_dof_pos = self.hand_dof_state[..., 0]
        self.hand_dof_vel = self.hand_dof_state[..., 1]
        self.prev_hand_dof_vel = torch.zeros_like(self.hand_dof_state[..., 1])
        
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # Add base position for debugging/etc.
        self.hand_base_pos = self.rigid_body_states[:, self.hand_base_handle][:, 0:3]
        
        self.pose_sensor_state = self.rigid_body_states[:, self.pose_sensor_handles][
            :, :, 0:13
        ]
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
            self.num_envs, -1
        )
        assert self.vec_sensor_tensor.shape[1] % 6 == 0  # sanity check
        self.num_bodies = self.rigid_body_states.shape[1]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # current position control targets for each joint (joints with no actuators should be set to 0)
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        # add up the number of successes in each env (resets when the env is reset)
        self.successes = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        # smoothed value keeping track of how many successes before drop / timeout in each env
        self.consecutive_successes = torch.zeros(
            1, dtype=torch.float, device=self.device
        )

        all_observation_names = self.cfg["observations"]["actor_observations"]
        if self.cfg["observations"]["asymmetric_observations"]:
            all_observation_names += self.cfg["observations"]["critic_observations"]
        all_observation_names = list(set(all_observation_names))  # remove duplicates

        # reserve space for previous observation buffer (for object pose and robot dof)
        len_obj_pose_buffer = self.obs_dims["obj_pose_history"]
        assert len_obj_pose_buffer % 7 == 0, \
            "obj_pose_buffer length must be a multiple of 7"
        assert len_obj_pose_buffer >=  7 * 2, \
            "obj_pose_buffer length must be equal to or greater than 14 to save more than one step of history"

        self.obj_pose_buffer = torch.zeros(
            (self.num_envs, len_obj_pose_buffer),
            dtype=torch.float,
            device=self.device,
        )

        len_dof_pos_buffer = self.obs_dims["dof_pos_history"]
        assert len_dof_pos_buffer % self.num_actuated_dofs == 0, \
            "dof_pos_buffer length must be a multiple of the " + \
                    f"actuated dofs ({self.num_actuated_dofs})"
        assert len_dof_pos_buffer >= self.num_actuated_dofs * 2, \
            f"dof_pos_buffer length must be equal to or greater than\
            {self.num_actuated_dofs * 2} to save more than one step of history"

        self.dof_pos_buffer = torch.zeros(
            (self.num_envs, len_dof_pos_buffer),
            dtype=torch.float,
            device=self.device,
        )

        self.debug_obs_buf = torch.zeros_like(self.obs_buf)
        
        # obs_buf is initialized in parent class but set the buffer for student observation here
        # this is trained in a separate framework from rl_games, so it is treated a bit differently from the other observations
        self.student_obs_buf = torch.zeros(
            (self.num_envs, self.student_obs_dim), device=self.device, dtype=torch.float32
        )

        # joint and sensor readout recording buffers
        self.record_dof_poses = self.cfg["logging"]["record_dofs"]
        self.record_length = self.cfg["logging"]["record_length"]
        self.record_observations = self.cfg["logging"]["record_observations"]
        if self.record_dof_poses:
            self.dof_pose_recording = torch.zeros(
                (self.num_envs, self.record_length, self.num_actuated_dofs),
                dtype=torch.float,
                device=self.device
            )
        if self.record_observations:
            self.observation_recording = torch.zeros(
                (self.num_envs, self.record_length, self.obs_buf.shape[1]),
                dtype=torch.float,
                device=self.device
            )
        self.num_recorded_steps = 0
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.recording_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'recordings')    
        if not os.path.exists(self.recording_dir):
            os.makedirs(self.recording_dir)
        self.recording_save_path = os.path.join(
            self.recording_dir,
            f'{timestamp_str}')

    def pre_physics_step(self, actions):
        """
        convert actions to commands applicable to the robot and set them
        """
        env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).flatten()
        goal_ids_to_reset = self.reset_goal_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids=env_ids_to_reset, goal_env_ids=goal_ids_to_reset)

        clip_actions = self.cfg["actions"]["clip_value"]
        actions = torch.clip(actions, -clip_actions, clip_actions)
        self.actions = actions.to(self.device)
        # print(f"{self.actions.mean(dim=0)=}\t{self.actions.std(dim=0)=}")
        if self.cfg["env"]["use_relative_control"]:
            targets = (
                self.prev_targets[:, self.actuated_dof_indices]
                + self.cfg["env"]["relative_control_speed_scale"] * self.control_dt * self.actions
            )
        else:
            targets = scale(
                self.actions,
                self.actuated_dof_lower_limits,
                self.actuated_dof_upper_limits,
            )
        self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
            targets, self.actuated_dof_lower_limits, self.actuated_dof_upper_limits
        )
        self.prev_targets[:] = self.cur_targets[:]
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.cur_targets)
        )

    def post_physics_step(self):
        self.progress_buf += 1

        # update the state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # compute tensors for accessing specific parts of the state tensors
        # since it uses an array for indexing, it's not possible (afaik -Yasu) to make them reference the same memory and have them update automatically, like how self.hand_dof_pos is done
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        
        # compute finger states
        self.pose_sensor_state = self.rigid_body_states[:, self.pose_sensor_handles][
            :, :, 0:13
        ]

        # update the history buffers
        self.obj_pose_buffer[:,:-7] = self.obj_pose_buffer[:,7:]
        self.obj_pose_buffer[:,-7:] = self.object_pose.clone()
        self.obj_pose_buffer[:,-7:-4] -= self.goal_init_states[:, :3]  # try to have zero mean

        self.dof_pos_buffer[:,:-self.num_actuated_dofs] = self.dof_pos_buffer[:,self.num_actuated_dofs:]
        self.dof_pos_buffer[:,-self.num_actuated_dofs:] = unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )

        # compute dof and object velocity numerically
        # this may be more useful in some cases where the velocity reported by isaacgym is not accurate (due to impulses applied for contact?)
        # https://forums.developer.nvidia.com/t/inaccuracy-of-dof-state-readings/197373
        current_hand_dof = self.dof_pos_buffer[:,-self.num_actuated_dofs:]
        previous_hand_dof = self.dof_pos_buffer[:,-self.num_actuated_dofs*2:-self.num_actuated_dofs]
        self.hand_dof_vel_numerical = (current_hand_dof - previous_hand_dof) / self.control_dt
        current_obj_pos = self.obj_pose_buffer[:,-7:-4]
        previous_obj_pos = self.obj_pose_buffer[:,-14:-7][:,-7:-4]
        self.object_linvel_numerical = (current_obj_pos - previous_obj_pos) / self.control_dt
        # compute object rotational velocity
        current_quat = self.obj_pose_buffer[:,-4:]
        previous_quat = self.obj_pose_buffer[:,-14:-7][:,-4:]
        angle, axis = quat_to_angle_axis(quat_mul(current_quat, quat_conjugate(previous_quat)))
        self.object_angvel_numerical = angle.unsqueeze(1) * axis / self.control_dt

        # compute acceleration by taking finite difference of velocity
        self.dof_acceleration = (self.hand_dof_vel - self.prev_hand_dof_vel) / self.control_dt
        self.prev_hand_dof_vel = self.hand_dof_vel.clone()

        self.check_termination()
        self.compute_reward()

        # if recording is avtivate, register dof poses/observations
        if self.record_dof_poses or self.record_observations:
            if self.num_recorded_steps <= self.record_length:
                self.record_step()

        # add rewards_dict to extras
        self.extras.update(self.rewards_dict)
        # log additional curriculum info
        self.extras["consecutive_successes"] = self.consecutive_successes.item()
        self.extras["average_rotvel_x"] = self.object_angvel_numerical[:,0].mean().item()
        self.extras["min_rotvel_x"] = self.object_angvel_numerical[:,0].min().item()
        self.extras["max_rotvel_x"] = self.object_angvel_numerical[:,0].max().item()
        self.extras["std_rotvel_x"] = self.object_angvel_numerical[:,0].std().item()
        self.compute_observations()

        # visualize
        if not self.headless and self.cfg["env"]["enable_debug_viz"]:
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                hand_pos = self.hand_base_pos.cpu().numpy()
                x, y, z = hand_pos[i]
                self._draw_sphere(i, x, y, z)
                self._draw_frame_axes(i, self.goal_pos[i], self.goal_rot[i])
                self._draw_frame_axes(i, self.object_pos[i], self.object_rot[i])

        if not self.headless and self.cfg["env"]["enable_contact_viz"]:
            if not self.cfg["env"]["enable_debug_viz"]:
                # clear lines if it has not been cleared already
                self.gym.clear_lines(self.viewer)
            assert not self.cfg["sim"]["use_gpu_pipeline"], "contact visualization can be only done with CPU pipeline"
            assert self.cfg["sim"]["physx"]["contact_collection"] in [1, 2], "contact_collection must be set to 1 or 2"
            assert not self.cfg["task"]["randomize"], "contact visualization is not supported with randomization, since the code for applying DR seems to be hardcoded to use GPU somewhere"
            self.gym.draw_env_rigid_contacts(self.viewer, self.envs[0], gymapi.Vec3(1,0.2,0.2), 1, False)

        # if specified, update the camera position
        if self.cfg["visualization"]["move_camera"]:
            self.cam_pos += self.cam_movement_per_step
            self.cam_target += self.cam_movement_per_step
            self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, self.cam_target)

        # update logger
        if self.cfg["logging"]["rt_plt"]:
            self.get_logs()

    def record_step(self):
        '''
        Records the dof and/or observation buffers and saves them to a .npy file.
        '''
        print("Recording!")
        if self.num_recorded_steps < self.record_length:
            if self.record_dof_poses:
                self.dof_pose_recording[:,self.num_recorded_steps, :] = self.dof_pos_buffer[:,-self.num_actuated_dofs:]
            if self.record_observations:
                self.observation_recording[:,self.num_recorded_steps, :] = self.obs_buf
        else:
            if self.record_dof_poses:
                np.save(self.recording_save_path + "_dof_poses.npy", self.dof_pose_recording.numpy(force=True))
                print('dof poses saved')
            if self.record_observations:
                np.save(self.recording_save_path + "_observation.npy", self.observation_recording.numpy(force=True))
            print("all recordings saved, exiting")
            exit()
        self.num_recorded_steps += 1

    def check_termination(self):
        """
        check termination conditions for each env and set the corresponding buffers
        """
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        rot_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
        )
        dist = torch.norm(self.object_pos - self.goal_pos, dim=-1)
        # envs whose object orientation is within success tolerance of goal orientation
        self.success_buf = rot_dist < self.cfg["rewards"]["success_tolerance"]
        self.successes += self.success_buf
        # envs where the cube was dropped
        self.dropped_buf = dist > self.cfg["rewards"]["fall_dist_threshold"]
        # the parent class sets the self.timeout_buf in the step() function so use a local variable here
        timeout_buf = self.progress_buf > self.max_episode_length - 1

        self.reset_goal_buf = self.success_buf | self.dropped_buf | timeout_buf
        # for the successful envs, the robot does not need to be reset
        self.reset_buf = self.dropped_buf | timeout_buf

        # calculate the consecutive successes
        finished_consecutive_successes = torch.sum(self.reset_buf * self.successes)
        num_successes = torch.sum(self.reset_buf)
        av_factor = 0.1  # smoothing factor
        self.consecutive_successes = torch.where(num_successes > 0, av_factor * finished_consecutive_successes / num_successes + (1.0 - av_factor) * self.consecutive_successes, self.consecutive_successes)
        self.successes[self.reset_buf] = 0

    def reset_goal_states(self, env_ids):
        """
        reset the goal states to the initial states for env_ids by overwriting self.goal_states.
        this implementation is for the in-hand reorientation task, but override it in your class if the reset procedure is different
        """
        rand_floats_goal = torch_rand_float(
            -1.0, 1.0, (len(env_ids), 3), self.device
        )
        # reset goal position
        self.goal_states[env_ids, 0:3] = self.goal_init_states[
            env_ids, 0:3
        ]
        # reset goal orientation
        self.goal_states[env_ids, 3:7] = randomize_rotation(
            rand_floats_goal[:, 0],
            rand_floats_goal[:, 1],
            self.x_unit_tensor[env_ids],
            self.y_unit_tensor[env_ids],
        )
    
    def custom_reset(self):
        """
        if you want to set up your own specific code to override the pose of objects, do so here
        (e.g. keep the object still in the air for the first few moments to help the hand grab it)
        The function should modify self.root_state_tensor and return a tensor of indices of the objects whose status should be reset.
        Then the reset_idx function will call gym.set_actor_root_state_tensor_indexed() to actually reset the objects in IsaacGym.
        """
        return torch.zeros(0, device=torch.device(self.device))

    def reset_idx(self, env_ids, goal_env_ids=[]):
        """
        Reset the envs (the robot dofs and the object pose) in env_ids and
        the goals in goal_env_ids. The former forcibly resets all the dofs
        of the hand, which isn't ideal for joints constrained by tendons
        #TODO ... fix it if it becomes a problem
        """
        if self.cfg["task"]["randomize"]:
            self.apply_randomizations(self.cfg["task"]["randomization_params"])

        # keep track of which indices of the root state tensor should be reset at the end of this function
        reset_indices = torch.zeros(0, device=torch.device(self.device)).to(torch.int32)

        # handle any custom reset procedures and add the indices of those objects to reset_indices
        custom_reset_indices = self.custom_reset()
        reset_indices = torch.cat(
            (reset_indices, custom_reset_indices.to(torch.int32))
        )
 
        if len(goal_env_ids) > 0:
            # overwrite self.goal_states in this function
            self.reset_goal_states(goal_env_ids)

            # set the goal states in the sim
            self.root_state_tensor[self.goal_object_indices[goal_env_ids], 0:3] = (
                self.goal_states[goal_env_ids, 0:3] + self.goal_visual_displacement_tensor
            )
            self.root_state_tensor[
                self.goal_object_indices[goal_env_ids], 3:7
            ] = self.goal_states[goal_env_ids, 3:7]
            self.root_state_tensor[
                self.goal_object_indices[goal_env_ids], 7:13
            ] = 0  # zero velocity
            reset_indices = torch.cat(
                (reset_indices, self.goal_object_indices[goal_env_ids].to(torch.int32))
            )

        if len(env_ids) > 0:
            # draw rand floats
            rand_floats = torch_rand_float(
                -1.0, 1.0, (len(env_ids), self.num_hand_dofs * 2 + 5), self.device
            )
            # reset hand state
            dof_range = self.hand_dof_upper_limits - self.hand_dof_lower_limits
            dof_pos = (
                self.hand_dof_default_pos
                + rand_floats[:, 5 : 5 + self.num_hand_dofs]
                * self.cfg["reset_noise"]["dof_pos"]
                * dof_range
            )
            dof_pos = torch.clamp(
                dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            dof_vel = (
                self.hand_dof_default_vel
                + rand_floats[:, 5 + self.num_hand_dofs : 5 + self.num_hand_dofs * 2]
                * self.cfg["reset_noise"]["dof_vel"]
            )
            self.hand_dof_pos[env_ids] = dof_pos
            self.hand_dof_vel[env_ids] = dof_vel
            # TODO: may have to set nonactuated to 0
            self.prev_targets[env_ids] = dof_pos
            self.cur_targets[env_ids] = dof_pos

            hand_indices = self.hand_indices[env_ids].to(torch.int32)
            # set the dof targets in the sim
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(hand_indices),
                len(env_ids),
            )
            # set the dof states in the sim
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(hand_indices),
                len(env_ids),
            )

            # reset object position
            object_states = self.object_init_states[env_ids].clone()
            object_states[:, 0:3] += (
                rand_floats[:, 0:3] * self.cfg["reset_noise"]["object_pos"]
            )
            # reset object rotation
            object_states[:, 3:7] = randomize_rotation(
                rand_floats[:, 3],
                rand_floats[:, 4],
                self.x_unit_tensor[env_ids],
                self.y_unit_tensor[env_ids],
            )
            
            # set the object state in the sim
            self.root_state_tensor[self.object_indices[env_ids]] = object_states

            reset_indices = torch.cat(
                (reset_indices, self.object_indices[env_ids].to(torch.int32))
            )
            # reset buffers
            self.progress_buf[env_ids] = 0

        if len(goal_env_ids) or len(env_ids):
            # apparently this can only be called once per step?
            # will return False if command fails
            assert self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state_tensor),
                gymtorch.unwrap_tensor(reset_indices),
                len(reset_indices),
            )

    def reset(self):
        """Reset all robots and goals"""
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids=all_envs, goal_env_ids=all_envs)
        obs, _, _, _ = self.step(
            torch.zeros(
                self.num_envs, self.num_actions, device=self.device, requires_grad=False
            )
        )
        return obs

    def _prepare_reward_function(self):
        """
        Prepare a list of reward functons, which will be called to compute the total reward
        looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are the nonzero entries in self.cfg["rewards"]["scales"]
        """
        # prepare list of reward functions
        self.reward_functions = []
        self.reward_names = []
        for name, _ in self.reward_scales.items():
            self.reward_names.append(name)
            func_name = "_reward_" + name
            # find member function with name func_name
            try:
                self.reward_functions.append(getattr(self, func_name))
            except AttributeError:
                raise AttributeError(
                    f"Reward function {func_name} not found, remove reward {name} or implement member function {func_name}"
                )
        # init dict for logging each reward value (averaged across all environments) separately
        self.rewards_dict = {}

    def _prepare_observations(self):
        """
        Prepare a list of observation functons, which will be called to compute the full observation
        looks for self._observation_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are the entries defined
        in actor_observations (and critic_observations, if asymmetric_observations is set to True)
        returns the observation dimension for the actor and critic (latter is 0 if asymmetric_observations is False)
        """

        def collect_observation_functions_compute_dim(obs_names):
            """
            create a list of the observation functions for the list obs_names,
            and also compute the total dimension of that set of observations
            """
            obs_functions = []
            obs_dim = 0
            for obs_name in obs_names:
                try:
                    obs_dim += self.obs_dims[obs_name]
                except KeyError:
                    raise KeyError(f"could not find obs_dims for observation {obs_name}, check config file")
                func_name = "_observation_" + obs_name
                # find member function with name func_name
                try:
                    obs_functions.append(getattr(self, func_name))
                except AttributeError:
                    raise AttributeError(
                        f"Observation function {func_name} not found, remove observation {obs_name} or implement member function {func_name}"
                    )
            return obs_functions, obs_dim

        self.actor_obs_functions, actor_obs_dim = collect_observation_functions_compute_dim(
            self.cfg["observations"]["actor_observations"]
        )
        if self.cfg["observations"]["asymmetric_observations"]:
            self.critic_obs_functions, critic_obs_dim = collect_observation_functions_compute_dim(
                self.cfg["observations"]["critic_observations"]
            )
        else:
            critic_obs_dim = 0
        # student observations
        self.student_obs_functions, self.student_obs_dim = collect_observation_functions_compute_dim(
            self.cfg["observations"]["student_observations"]
        )
        return actor_obs_dim, critic_obs_dim

    def _prepare_logged_functions(self):
        """
        Prepare a list of observation, reward and custom functions, which will be called to
        compute the plots of the visualization. Plot a canvas on which
        """
        # add a logger for online logging and plotting of states
        # TODO: this has been just ported from faive_gym, make it fully work with faive-isaac
        from isaacgymenvs.utils.logger import Logger
        self.logger = Logger(dt = self.control_dt, 
            buf_len_s = self.cfg["logging"]["buf_len_s"],
            rows = self.cfg["logging"]["num_rows"],
            cols = self.cfg["logging"]["num_cols"],)
        
        self.logger.measurement_names = self.cfg["logging"]["measurements"]
        self.logger.measurement_units = self.cfg["logging"]["units"]
        self.logging_functions = []
        self.logging_indices = []
        for name in self.cfg["logging"]["measurements"]:
            try:
                if "dof" in name:
                    func_name = ("_").join(name.split("_")[:-2])
                    dof_name = ("_").join(name.split("_")[-2:])
                    self.logging_indices.append(self.logger.dof_names.index(dof_name))
                    if "pos" in name:
                        self.logger.num_lines_per_subplot.append(2)
                    else:
                        self.logger.num_lines_per_subplot.append(1)
                elif "fingertip" in name:
                    func_name = ("_").join(name.split("_")[:-1])
                    finger = name.split("_")[-1]
                    self.logging_indices.append(self.logger.finger_names.index(finger))
                    obs_name = ("_").join(name.split("_")[2:-1])
                    if obs_name.endswith("stat"):
                        obs_name = obs_name[:-5]
                    self.logger.num_lines_per_subplot.append(
                            getattr(self.cfg["observations"]["obs_dims"], obs_name)//5)
                else:
                    if "reward" in name:
                        self.logger.num_lines_per_subplot.append(1)
                    if "observation" in name:
                        obs_name = ("_").join(name.split("_")[2:])
                        if obs_name.endswith("stat"):
                            obs_name = obs_name[:-5]
                        self.logger.num_lines_per_subplot.append(
                            self.obs_dims[obs_name])
                    self.logging_indices.append(None)
                    func_name = name
                # check for std/mean
                if func_name.endswith("_stat"):
                    func_name = func_name[:-5]
                    self.logger.num_lines_per_subplot[-1] *= 2
                self.logging_functions.append(getattr(self, func_name))
            except AttributeError:
                raise AttributeError(
                    f"Logging function {func_name} not found, remove measurement {name} or implement member function {func_name}"
                )
        # start plot process
        self.logger.plot_process.start()
    
    def _get_observation_to_log(self, obs_function, env_idx, obs_name, num_lines, obs_idx = None):
        """
        Calls the observation function and returns the observation to log
        """
        #print("Getting obs ", obs_name, " for env ", env_idx, " with idx ", obs_idx, " and num_lines ", num_lines)
        obs_tensor = obs_function()
        if "_stat" in obs_name:
                #print(obs_tensor.shape)
                if obs_idx is None:
                    if num_lines == 2:
                        return [torch.mean(obs_tensor).item(), torch.std(obs_tensor).item()]
                    else:
                        values = []
                        for i in range(num_lines//2):
                            values += [torch.mean(obs_tensor, dim=0)[i].item(), 
                                torch.std(obs_tensor, dim=0)[i].item()]
                        return values
                else:
                    if "pos" in obs_name and "dof" in obs_name:
                        pos_mean = torch.mean(obs_tensor, dim=0)[obs_idx].item()
                        target_mean = torch.mean(self.cur_targets, dim=0)[obs_idx].item()
                        pos_std = torch.std(obs_tensor, dim=0)[obs_idx].item()
                        target_std = torch.std(self.cur_targets, dim=0)[obs_idx].item()
                        return [pos_mean, target_mean, pos_std, target_std]
                    elif "fingertip" in obs_name:
                        values = []
                        for i in range(num_lines//2):
                            values += [
                                torch.mean(obs_tensor, dim=0)[obs_idx*5+i].item(),
                                torch.std(obs_tensor, dim=0)[obs_idx*5+i].item()
                            ]
                        return values
                    elif "proxim" in obs_name:
                        values = []
                        for i in range(num_lines//2):
                            values += [
                                torch.mean(obs_tensor, dim=0)[obs_idx*5+i].item(),
                                torch.std(obs_tensor, dim=0)[obs_idx*5+i].item()
                            ]
                        return values
                    else:
                        return [
                            torch.mean(obs_tensor, dim=0)[obs_idx].item(),
                            torch.std(obs_tensor, dim=1)[obs_idx].item()
                        ]
        else:
            if obs_idx is None:
                if num_lines == 1:
                    return obs_tensor[env_idx].item()
                else:
                    return [obs_tensor[env_idx,i].item() for i in range(num_lines)]
            else:
                if "pos" in obs_name and "dof" in obs_name:
                    pos = self.hand_dof_pos[:, self.actuated_dof_indices][env_idx, obs_idx].item()
                    target = self.cur_targets[:, self.actuated_dof_indices][env_idx, obs_idx].item()
                    return [pos, target]
                elif "fingertip" in obs_name:
                    #print("Taking idx ", obs_idx*5, " to ", obs_idx*5+4, " from fingertip obs tensor")
                    return [obs_tensor[env_idx,obs_idx*5+i].item() for i in range(num_lines)]
                elif "proxim" in obs_name:
                    
                    return [obs_tensor[env_idx,obs_idx*5+i].item() for i in range(num_lines)]
                else:
                    return obs_tensor[env_idx, obs_idx].item()

    def compute_reward(self):
        """
        Calls each reward function which has a non-zero scale (processed in self._prepare_reward_function)
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0
        for reward_name, reward_func in zip(self.reward_names, self.reward_functions):
            if self.reward_scales[reward_name] == 0:
                continue  # ignore zero-scaled rewards
            reward = reward_func() * self.reward_scales[reward_name]
            self.rew_buf += reward
            self.rewards_dict[f"rew_{reward_name}"] = reward.mean()

    def _fill_obs(self, obs_tensor, obs_names, obs_functions):
        """
        convenience function fill up the observation tensor
        obs_tensor: tensor that will be filled with the observations
        obs_names: list of observation names
        obs_functions: list of observation functions
        """
        obs_start = 0
        for obs_name, obs_func in zip(obs_names, obs_functions):
            obs_dim = self.obs_dims[obs_name]
            obs_end = obs_start + obs_dim
            obs = obs_func()
            assert (
                obs_dim == obs.shape[1]
            ), f"set correct observation dimension for [{obs_name}] in cfg"
            scale = self.obs_scales.get(obs_name, 1.0)
            assert scale != 0, f"set nonzero observation scale for [{obs_name}] in cfg"
            obs_tensor[:, obs_start:obs_end] = obs * scale
            obs_start = obs_end

    def compute_observations(self):
        """
        updates the observation buffer with the current observations
        """
        self._fill_obs(self.obs_buf, self.cfg["observations"]["actor_observations"], self.actor_obs_functions)
        if self.cfg["observations"]["asymmetric_observations"]:
            self._fill_obs(self.states_buf, self.cfg["observations"]["critic_observations"], self.critic_obs_functions)
        if self.cfg["observations"]["clip"]:
            clip_obs = self.cfg["observations"]["clip_value"]
            self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
            self.states_buf = torch.clip(self.states_buf, -clip_obs, clip_obs)
    
    def compute_student_observations(self):
        """
        computes the observations sent to student
        """
        self._fill_obs(self.student_obs_buf, self.cfg["observations"]["student_observations"], self.student_obs_functions)


    def get_logs(self):
        """
        Captures variables from the current environment to the logger,
        which plots the real time on the screen.
        """
        for log_name, log_func, num_lines, obs_idx in zip(
            self.cfg["logging"]["measurements"],
            self.logging_functions,
            self.logger.num_lines_per_subplot,
            self.logging_indices
        ):
            self.logger.log_state(
                key = log_name,
                value=self._get_observation_to_log(log_func,
                    self.logger.logged_env_idx,
                    log_name,
                    num_lines,
                    obs_idx)
            )
        self.logger._state_send()

    def create_sim(self):
        """
        create the simulation environment (called within the super class's __init__)
        """
        self.sim = self.gym.create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.cfg["task"]["randomize"]:
            # apply randomization once before first sim step
            self.apply_randomizations(self.cfg["task"]["randomization_params"])

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        env_lower = gymapi.Vec3(
            -self.cfg["env"]["env_spacing"], -self.cfg["env"]["env_spacing"], 0.0
        )
        env_upper = gymapi.Vec3(self.cfg["env"]["env_spacing"], self.cfg["env"]["env_spacing"], 0.0)
        num_per_row = int(np.sqrt(self.num_envs))

        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../assets")
        )
        hand_asset_file = os.path.normpath(self.cfg["asset"]["model_file"])
        asset_files_dict = {
            "block": "urdf/cube_multicolor.urdf",
            "sphere": "urdf/sphere.urdf",
            "pyramid": "objects_dext_manip/pyramid.xml",
            "prism": "objects_dext_manip/trig_prism.xml",
            "hex_prism": "objects_dext_manip/hex_prism.xml",
            "flat_pyr": "objects_dext_manip/flat_pyr.xml",
            "octahedron": "objects_dext_manip/octahedron.xml",
            "tetrahedron": "objects_dext_manip/tetrahedron.xml",
            "pentaprism": "objects_dext_manip/pentaprism.urdf",
            "dodecahedron": "objects_dext_manip/dodecahedron.xml",
            "stell_dodeca": "objects_dext_manip/stell_dodeca.urdf",
            "stairs": "objects_dext_manip/stairs.urdf",
            "block_pyr": "objects_dext_manip/block_pyr.urdf",
        }
        for i in range(len(self.cfg["env"]["object_type"])):
            try:
                os.path.normpath(
                    asset_files_dict[self.cfg["env"]["object_type"][i]]
                )
            except KeyError:
                raise ValueError(
                    f'Invalid object type: {self.cfg["env"]["object_type"][i]}, must be one of {asset_files_dict.keys()}'
                )
        
        # load Faive Hand asset with these options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        hand_asset = self.gym.load_asset(
            self.sim, asset_root, hand_asset_file, asset_options
        )
        # set up friction and restitution params
        # sphere rotation is somewhat brittle to these params...
        hand_props = self.gym.get_asset_rigid_shape_properties(hand_asset)
        for p in hand_props:
            p.friction = self.cfg["env"]["hand_friction"]
            p.torsion_friction = self.cfg["env"]["hand_friction"]
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(hand_asset, hand_props)

        # define some variables based on the asset
        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.num_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.num_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        self.num_hand_actuators = self.gym.get_asset_actuator_count(hand_asset)
        self.num_hand_tendons = self.gym.get_asset_tendon_count(hand_asset)
        # set up tendons
        # tendons are used for simulating rolling contact joints using two hinge joints.
        # make tendon stiffer than the default stiffness of 1.
        # 30 is the value used in shadow_hand.py
        # however, scale fixed/joint coef of the Shadow Hand model is around 0.007, instead of 1 of the Hand.
        # https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon-fixed
        # So 30 must be scaled to achieve the same tendon stiffness, i.e. 30 / 0.007 ~ 4000
        # scale the damping as well
        limit_stiffness = 4000
        t_damping = 10
        tendon_props = self.gym.get_asset_tendon_properties(hand_asset)

        # go through all tendons in the robot model and set their properties
        for i in range(self.num_hand_tendons):
            tendon_name = self.gym.get_asset_tendon_name(hand_asset, i)
            tendon_props[i].limit_stiffness = limit_stiffness
            tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(hand_asset, tendon_props)

        actuated_dof_names = [
            self.gym.get_asset_actuator_joint_name(hand_asset, i)
            for i in range(self.num_hand_actuators)
        ]
        actuated_dof_indices = []
        for name in actuated_dof_names:
            dof_index = self.gym.find_asset_dof_index(hand_asset, name)
            assert dof_index != -1, f"Could not find dof index for {name}"
            print(f"{name}\t->\t{dof_index}")
            actuated_dof_indices.append(dof_index)

        # get hand dof properties, loaded by Isaac Gym from the MJCF file
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)

        # load joint range information
        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []
        self.hand_dof_default_pos = []
        self.hand_dof_default_vel = []
        for i in range(self.num_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props["lower"][i])
            self.hand_dof_upper_limits.append(hand_dof_props["upper"][i])
            self.hand_dof_default_pos.append(0.0)
            self.hand_dof_default_vel.append(0.0)

        # convert to torch tensors so they can be computed in GPU
        self.actuated_dof_indices = to_torch(
            actuated_dof_indices, dtype=torch.long, device=self.device
        )  # indices of dofs with actuators attached to them
        self.hand_dof_lower_limits = to_torch(
            self.hand_dof_lower_limits, device=self.device
        )
        self.hand_dof_upper_limits = to_torch(
            self.hand_dof_upper_limits, device=self.device
        )
        self.hand_dof_default_pos = to_torch(
            self.hand_dof_default_pos, device=self.device
        )
        self.hand_dof_default_vel = to_torch(
            self.hand_dof_default_vel, device=self.device
        )
        self.num_actuated_dofs = len(self.actuated_dof_indices)
        # if using tendons to simulate rolling contact joints, only some of the dofs have actuators
        # they should be all you need to reconstruct the hand's full state, so use these dofs for observations as well
        self.actuated_dof_lower_limits = self.hand_dof_lower_limits[
            self.actuated_dof_indices
        ]
        self.actuated_dof_upper_limits = self.hand_dof_upper_limits[
            self.actuated_dof_indices
        ]
        # if actuated dof have overridden range, use that instead
        for i in range(self.num_actuated_dofs):
            actuated_dof_lower_limit = self.actuated_dof_lower_limits[i]
            actuated_dof_upper_limit = self.actuated_dof_upper_limits[i]
            actuated_dof_range_override = self.cfg["env"]["actuated_dof_range_override"]
            if actuated_dof_range_override != "None":
                # if dof range override is set, use that instead
                # check that the override is within the range of the hand
                assert actuated_dof_lower_limit - 1e-3 <= actuated_dof_range_override[i][0]
                assert actuated_dof_range_override[i][1] <= actuated_dof_upper_limit + 1e-3
                assert actuated_dof_range_override[i][0] < actuated_dof_range_override[i][1]
                self.actuated_dof_lower_limits[i] = actuated_dof_range_override[i][0]
                self.actuated_dof_upper_limits[i] = actuated_dof_range_override[i][1]

        # create handles to access body parts of interest and force sensors
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        
        pose_sensor_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.cfg["asset"]["pose_sensor_names"]
        ]
        force_sensor_handles = [
            self.gym.find_asset_rigid_body_index(hand_asset, name)
            for name in self.cfg["asset"]["force_sensor_names"]
        ]
        for fs_handle in force_sensor_handles:
            self.gym.create_asset_force_sensor(hand_asset, fs_handle, sensor_pose)
    
        hand_base = self.gym.find_asset_rigid_body_index(
            hand_asset, self.hand_base_name
        )
        

        # load manipulated object and goal assets
        object_asset_list = []
        goal_asset_list = []
        for i in range(len(self.cfg["env"]["object_type"])):
            object_asset_file = os.path.normpath(
                    asset_files_dict[self.cfg["env"]["object_type"][i]]
                )
            object_asset_options = gymapi.AssetOptions()
            object_asset_list.append(self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            ))
            object_asset_options.disable_gravity = True
            goal_asset_list.append(self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            ))
            

        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(self.cfg['env']['hand_start_p'][0],
                                        self.cfg['env']['hand_start_p'][1],
                                        self.cfg['env']['hand_start_p'][2])
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        # rotate 200 degrees around x axis to make palm face up, and slightly tilt it downwards
        hand_start_pose.r = gymapi.Quat(self.cfg['env']['hand_start_r'][0],
                                        self.cfg['env']['hand_start_r'][1],
                                        self.cfg['env']['hand_start_r'][2],
                                        self.cfg['env']['hand_start_r'][3])
        [pose_dx, pose_dy, pose_dz] = self.cfg["env"]["object_start_offset"]  # position the object above the palm

        object_start_pose.p.x = hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = hand_start_pose.p.z + pose_dz
        
    
        goal_visual_displacement = gymapi.Vec3(-0.2, -0.06, 0.08)
        # the goal object within the rendered scene will be displaced by this amount from the actual goal
        self.goal_visual_displacement_tensor = to_torch(
            [
                goal_visual_displacement.x,
                goal_visual_displacement.y,
                goal_visual_displacement.z,
            ],
            device=self.device,
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + goal_visual_displacement

        # compute aggregate size
        max_agg_bodies = self.num_hand_bodies + 2
        max_agg_shapes = self.num_hand_shapes + 2

        self.envs = []
        object_init_states = []
        hand_indices = []
        object_indices = []
        goal_object_indices = []
        # one-hot encoding which saves the object type loaded in each environment
        self.object_type = torch.zeros([self.num_envs, len(self.cfg["env"]["object_type"])])
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.cfg["env"]["aggregate_mode"]:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            actor_handle = self.gym.create_actor(
                env_ptr, hand_asset, hand_start_pose, "hand", i, -1, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, hand_dof_props)

            self.gym.enable_actor_dof_force_sensors(
                env_ptr, actor_handle
            )  # need to be explicitly enabled for torque sensors to work

            # set the first body to be black (base of the hand) to match real robot
            self.gym.set_rigid_body_color(
                env_ptr, actor_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.25, 0.25, 0.25))

            # add object, seg_id = 1
            object_handle = self.gym.create_actor(
                env_ptr, object_asset_list[i%len(self.cfg["env"]["object_type"])], object_start_pose, "object", i, 0, 1
            )
            self.object_type[i][i%len(self.cfg["env"]["object_type"])] = 1
            object_init_states.append(
                [
                    object_start_pose.p.x,
                    object_start_pose.p.y,
                    object_start_pose.p.z,
                    object_start_pose.r.x,
                    object_start_pose.r.y,
                    object_start_pose.r.z,
                    object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )

            # add goal object
            # by setting the fifth argument to not coincide with the others, the goal object does not collide with anything else
            goal_handle = self.gym.create_actor(
                env_ptr,
                goal_asset_list[i%len(self.cfg["env"]["object_type"])],
                goal_start_pose,
                "goal_object",
                i + self.num_envs,
                0,
                0,
            )

            # save the indices of each item in the environment
            hand_idx = self.gym.get_actor_index(
                env_ptr, actor_handle, gymapi.DOMAIN_SIM
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            goal_object_idx = self.gym.get_actor_index(
                env_ptr, goal_handle, gymapi.DOMAIN_SIM
            )
            hand_indices.append(hand_idx)
            object_indices.append(object_idx)
            goal_object_indices.append(goal_object_idx)

            if self.cfg["env"]["aggregate_mode"]:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
        
        # used for resetting the object
        self.object_init_states = to_torch(
            object_init_states, dtype=torch.float, device=self.device
        ).view(self.num_envs, 13)
        # this tensor sets the goal state of the object
        self.goal_states = self.object_init_states.clone()
        # lower it to match the height of the hand
        self.goal_states[:, 2] -= 0.04
        
        self.goal_init_states = self.goal_states.clone()
        self.hand_indices = to_torch(hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(
            object_indices, dtype=torch.long, device=self.device
        )
    
        self.goal_object_indices = to_torch(
            goal_object_indices, dtype=torch.long, device=self.device
        )

        self.hand_base_handle = to_torch(
            hand_base, dtype=torch.long, device=self.device
        )
        
        self.force_sensor_handles = to_torch(
            force_sensor_handles, dtype=torch.long, device=self.device
        )
        self.pose_sensor_handles = to_torch(
            pose_sensor_handles, dtype=torch.long, device=self.device
        )


    def _parse_cfg(self, cfg):
        """
        Parse the configuration object and save relevant parameters
        """
        self.reward_scales = class_to_dict(cfg["rewards"]["scales"])
        self.obs_dims = class_to_dict(cfg["observations"]["obs_dims"])
        self.obs_scales = class_to_dict(cfg["observations"]["obs_scales"])

    # ------------- debug visualizer functions -----------------
    def _draw_sphere(self, env_idx, x, y, z, radius=0.05, color=(1, 1, 0)):
        """
        draw a sphere designated location for the environment with index env_idx
        """
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 26, 26, None, color=color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(
            sphere_geom, self.gym, self.viewer, self.envs[env_idx], sphere_pose
        )

    def _draw_frame_axes(self, env_idx, pos, rot, ax_len=0.2):
        """
        draw xyz axes for the given frame
        """
        x_tip = pos + quat_apply(rot, to_torch([ax_len, 0, 0], device=self.device))
        y_tip = pos + quat_apply(rot, to_torch([0, ax_len, 0], device=self.device))
        z_tip = pos + quat_apply(rot, to_torch([0, 0, ax_len], device=self.device))
        x_tip = x_tip.cpu().numpy()
        y_tip = y_tip.cpu().numpy()
        z_tip = z_tip.cpu().numpy()
        pos = pos.cpu().numpy()
        self.gym.add_lines(
            self.viewer,
            self.envs[env_idx],
            1,
            [pos[0], pos[1], pos[2], x_tip[0], x_tip[1], x_tip[2]],
            [0.85, 0.1, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            self.envs[env_idx],
            1,
            [pos[0], pos[1], pos[2], y_tip[0], y_tip[1], y_tip[2]],
            [0.1, 0.85, 0.1],
        )
        self.gym.add_lines(
            self.viewer,
            self.envs[env_idx],
            1,
            [pos[0], pos[1], pos[2], z_tip[0], z_tip[1], z_tip[2]],
            [0.1, 0.1, 0.85],
        )

    # ------------- reward functions -----------------
    # the reward functions below may not be called depending on the reward
    # configuration, so they must not contain computation that is used in
    # other functions i.e. they should only compute the reward term and
    # nothing else.
    # for readability, rewards specific to a task should be named [task_name]task_[reward_name]

    # first define generic reward functions that can be used for any task

    def _reward_dof_acc_penalty(self):
        """
        Penalize joint acceleration, could remove shaking
        """
        return torch.norm(self.dof_acceleration, p=2, dim=-1)

    def _reward_dof_vel_penalty(self):
        """
        Penalize speed of the joints, smooth out movement
        """
        return torch.norm(self.hand_dof_vel, p=2, dim=-1)

    def _reward_action_penalty(self):
        """
        Penalize the magnitude of the action
        """
        return torch.norm(self.actions, p=2, dim=-1)

    def _reward_dof_trq_penalty(self):
        """
        Penalize the magnitude of the joint torque
        """
        return torch.norm(self.dof_force_tensor, p=2, dim=-1)

    def _reward_success(self):
        """
        Reward the agent for success (success_buf is computed in check_termination(), its definition is different for each task)
        """
        return self.success_buf

    def _reward_drop_penalty(self):
        """
        Penalize the agent for falling over
        """
        return self.dropped_buf

    def _reward_simple_hand_flat(self):
        """
        simple reward function that rewards the joint pos being close to zero
        useful for debugging policies, since it is such an easy task
        """
        dist_from_zero = torch.norm(self.hand_dof_pos, p=2, dim=-1)
        return 1.0 / (dist_from_zero + 0.1)

    # ---------------------------------------------------------------------
    # define reward functions specific to the in-hand reorientation task

    def _reward_reorienttask_obj_dist(self):
        """
        Reward the agent based on the distance between the object and the goal
        """
        return torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)

    def _reward_reorienttask_obj_rot(self):
        """
        Orientation alignment for the cube in hand and goal cube
        """
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        rot_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)
        )
        rot_eps = 0.1
        return 1.0 / (torch.abs(rot_dist) + rot_eps)

    # ---------------------------------------------------------------------
    # define reward functions specific to the in-hand sphere rotation task

    def _reward_rottask_obj_xrotvel(self):
        """
        reward the rotational velocity in the X axis of the object
        numerically computed velocity is used to avoid instability from isaacgym
        
        note: if the reward tapers down to 0 (instead of keep going negative like in this implementation), the agent tends to oscillate the ball instead of rotating
        (why? because it rotates in the desired direction at desired speed, then rotates back with a quick motion in a short amount of time)

        This reward was used to generate the motions used in TEDx demo, but it may be more stable to set a goal object with that slowly rotates, like in "Circus ANYmal" paper
        """
        rotvel = self.object_angvel_numerical
        # give max reward when rotvel magnitude is larger than 1 rad/s
        # optionally flip this sign back to positive for the
        # ablation study
        direction = - self.cfg["env"]["x_rotation_dir"]
        a = direction * rotvel[:, 0] + 1
        b = torch.ones_like(a) * 2
        # return the smallest of the two
        return torch.min(a, b)

    # ------------- observation functions -----------------
    # the observation functions below may not be called depending on the
    # reward configuration, so they must not contain computation that is
    # used in other functions i.e. they should only compute the observation
    # and nothing else.

    def _observation_dof_position(self):
        """
        Returns the position of actuated DoFs in the hand, scaled to [-1, 1]
        """
        return unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )
    
    def _observation_dof_pos_target(self):
        """
        position control target of the joints, scaled to [-1, 1]
        recommended when using relative control (as the policy won't know the joint control target otherwise)
        """
        dof_pos_target = self.cur_targets[:, self.actuated_dof_indices]
        dof_pos_target_normalized = unscale(
            dof_pos_target,
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )
        return dof_pos_target_normalized
    
    def _observation_obj_type(self):
        """
        Returns object type
        """
        return self.object_type
        
    def _observation_obj_pose_history(self):
        """
        Returns the history of object poses (7 DoF) (first element is the
        current pose, so using this with obj pos/quat is redundant)
        """
        return self.obj_pose_buffer
        
    def _observation_dof_pos_history(self):
        """
        Returns the history of joint positions, scaled to [-1, 1]
        (rightmost element is the current position, so using this with dof_pos is redundant)
        """
        return self.dof_pos_buffer
    
    def _observation_dof_speed(self):
        """
        Returns the speed of the actuated DoFs in the hand
        """
        return unscale(
            self.hand_dof_vel[:, self.actuated_dof_indices],
            self.actuated_dof_lower_limits,
            self.actuated_dof_upper_limits,
        )

    def _observation_dof_speed_numerical(self):
        """
        speed of the actuated DoFs in the hand computed numerically
        """
        return unscale(self.hand_dof_vel_numerical,
                       self.actuated_dof_lower_limits,
                       self.actuated_dof_upper_limits)

    def _observation_dof_force(self):
        """
        Returns the forces/torques measured in each DoF
        """
        return self.dof_force_tensor[:, self.actuated_dof_indices]

    def _observation_obj_pos(self):
        """
        Returns the observed object pos in the env's
        coordinate system (TODO wrt hand base)
        """
        obj_pos = self.object_pose.clone()[:, :3]
        obj_pos -= self.goal_init_states[:, :3]
        return obj_pos

    def _observation_obj_quat(self):
        """
        Returns the observed object orientation in the env's
        coordinate system (TODO wrt hand base) represented by
        a quaternion
        """
        obj_quat = self.object_pose.clone()[:, 3:7]
        return obj_quat

    def _observation_obj_linvel(self):
        """
        Returns the linear velocity of the manipulated
        object
        """
        return self.object_linvel

    def _observation_obj_angvel(self):
        """
        Returns the angular velocity of the manipulated
        object
        """
        return self.object_angvel

    def _observation_obj_linvel_numerical(self):
        """
        the linear velocity computed numerically with finite differences
        """
        return self.object_linvel_numerical

    def _observation_obj_angvel_numerical(self):
        """
        the angular velocity computed numerically with finite differences
        """
        return self.object_angvel_numerical

    def _observation_goal_pos(self):
        """
        Returns the goal object position
        """
        goal_pos = self.goal_pose.clone()[:, :3]
        goal_pos -= self.goal_init_states[:, :3]
        return goal_pos

    def _observation_goal_quat(self):
        """
        Returns the goal object orientation, represented by 
        a quaternion
        """
        goal_quat = self.goal_pose.clone()[:, 3:7]
        return goal_quat

    def _observation_goal_quat_diff(self):
        """
        Returns the difference in rotation between the
        current and the goal quaternion
        """
        return quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
    
    def _observation_pose_sensor_pos(self):
        """
        Returns the pose_sensor position for each pose_sensor
        """
        pose_sensor_pos = self.pose_sensor_state.clone()[:, :, :3]
        pose_sensor_pos -= self.goal_init_states[:, :3].unsqueeze(1)
        return pose_sensor_pos.reshape(self.num_envs, -1)

    def _observation_pose_sensor_quat(self):
        """
        Returns the orientation for each pose_sensor, represented
        by a quaternion
        """
        pose_sensor_quats = self.pose_sensor_state.clone()[:, :, 3:7]
        return pose_sensor_quats.reshape(self.num_envs, -1)

    def _observation_pose_sensor_linvel(self):
        """
        Returns the linear velocity of each pose_sensor
        """
        pose_sensor_linvel = self.pose_sensor_state.clone()[:, :, 7:10]
        return pose_sensor_linvel.reshape(self.num_envs, -1)

    def _observation_pose_sensor_angvel(self):
        """
        Returns the 13-DoF full state (pos, quat, linvel, angvel)
        of each pose_sensor
        """
        pose_sensor_angvel = self.pose_sensor_state.clone()[:, :, 10:13]
        return pose_sensor_angvel.reshape(self.num_envs, -1)

    def _observation_force_sensor_force(self):
        """
        Returns 6 DoF force + torque measurements from
        the proxims
        """
        
        return self.vec_sensor_tensor

    def _observation_actions(self):
        """
        Returns the latest actions from the policy
        """
        
        return self.actions
    


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
        quat_from_angle_axis(rand1 * np.pi, y_unit_tensor),
    )
