# Script that saves torch jit compiled policies for sim-to-real transfer, made
# by Benedek Forrai (bforrai@student.ethz.ch). A modified version of train.py,
# the header of which is pasted below.
#
# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Export a trained RL policy
# This will output a .onnx and .pt file to the same directory, which can be loaded in faive_franka_control or other solutions to run the policy on the real robot.
# The output files, created in the folder `faive_gym/exported_policies`, will have the names `[policy_name]_[timestamp]`, where `policy_name` is set by the `wandb_name`
# parameter. The export can be ran as follows:
# ```bash
# python export_policy.py task=FaiveHandP0 checkpoint=/path/to/checkpoint/FaiveHand.pth wandb_name=policy_name
# ```
# To check if the `.onnx` outputs are correct, you can use [netron](https://netron.app/). An exported `.onnx` should produce the following archtiecture:
# ![](onnx_export_sample.png)

import datetime
import isaacgym

import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import gym
import torch
import onnx
import onnxruntime as ort

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# register custom tasks for faive_gym here
from isaacgymenvs.tasks import isaacgym_task_map
from faive_gym.robot_hand import RobotHand
isaacgym_task_map["RobotHand"] = RobotHand

## ModelWrapper class from https://colab.research.google.com/github/Denys88/rl_games/blob/master/notebooks/train_and_export_onnx_example_discrete.ipynb
class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        
        
    def forward(self,input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        #print(input_dict)
        #output_dict = self._model.a2c_network(input_dict)
        #input_dict['is_train'] = False
        #return output_dict['logits'], output_dict['values']
        return self._model.a2c_network(input_dict)

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
@hydra.main(config_name="config", config_path="../cfg")
def launch_rlg_hydra(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)
    # force running on cpu
    cfg.sim_device = 'cpu'

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        return envs

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    # load agent from the path we'd like to convert
    agent = runner.create_player()
    agent.restore(cfg.checkpoint)
    # run tracing on cpu
    print("Current device of the agent:")
    print(agent.device)
    # run torch.jit.trace on the model
    import rl_games.algos_torch.flatten as flatten
    inputs = {
        'obs' : torch.zeros((1,) + agent.obs_shape).to(agent.device),
        'rnn_states' : agent.states,
    }

    with torch.no_grad():
        adapter = flatten.TracingAdapter(ModelWrapper(agent.model), inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        flattened_outputs = traced(*adapter.flattened_inputs)
        print(flattened_outputs)
    # export onnx and torchscript
    faive_gym_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(faive_gym_path,"exported_policies")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path_base = os.path.join(save_dir, run_name)
    traced.save(save_path_base + '.pt')
    torch.onnx.export(traced, *adapter.flattened_inputs, save_path_base + ".onnx", verbose=True, input_names=['obs'], output_names=['mu','log_std', 'value'])
    onnx_model = onnx.load(save_path_base + ".onnx")

    # Check that the model is well formed
    onnx.checker.check_model(onnx_model)

if __name__ == "__main__":
    launch_rlg_hydra()