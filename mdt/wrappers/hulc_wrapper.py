import logging
import os
import time
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch

from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id
from mdt.datasets.utils.episode_utils import process_depth, process_rgb, process_state

logger = logging.getLogger(__name__)


class HulcWrapper(gym.Wrapper):
    def __init__(self, dataset_loader, device, show_gui=False, **kwargs):
        self.set_egl_device(device)
        env = get_env(
            dataset_loader.abs_datasets_dir, show_gui=show_gui, obs_space=dataset_loader.observation_space, **kwargs
        )
        super(HulcWrapper, self).__init__(env)
        self.observation_space_keys = dataset_loader.observation_space
        self.transforms = dataset_loader.transforms
        self.proprio_state = dataset_loader.proprio_state
        self.device = device
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        logger.info(f"[HulcWrapper] Initialized PlayTableEnv for device {self.device}")
        print(f'[DEBUG] Rank={os.environ.get("LOCAL_RANK")}: HulcWrapper initialized! device={self.device}')

    @staticmethod
    def set_egl_device(device):
        if "EGL_VISIBLE_DEVICES" in os.environ:
            logger.warning("Environment variable EGL_VISIBLE_DEVICES is already set. Is this intended?")
        cuda_id = device.index if device.type == "cuda" else 0
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to calvin env README"
            )
            egl_id = 0
        print(f'[DEBUG] Rank={os.environ.get("LOCAL_RANK")}: cuda_id={cuda_id}, egl_id={egl_id}')
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def transform_observation(self, obs: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        import time
        s = time.time()
        state_obs = process_state(obs, self.observation_space_keys, self.transforms, self.proprio_state)
        # print(f'[DEBUG]: process_state:', time.time() - s)

        s = time.time()
        rgb_obs = process_rgb(obs["rgb_obs"], self.observation_space_keys, self.transforms, on_gpu=True)
        # print(f'[DEBUG]: process_rgb:', time.time() - s)

        s = time.time()
        depth_obs = process_depth(obs["depth_obs"], self.observation_space_keys, self.transforms)
        # print(f'[DEBUG]: process_depth:', time.time() - s)

        s = time.time()
        state_obs["robot_obs"] = state_obs["robot_obs"].to(self.device).unsqueeze(0)
        rgb_obs.update({"rgb_obs": {k: v.to(self.device).unsqueeze(0) for k, v in rgb_obs["rgb_obs"].items()}})
        depth_obs.update({"depth_obs": {k: v.to(self.device).unsqueeze(0) for k, v in depth_obs["depth_obs"].items()}})
        # print(f'[DEBUG]: update_dict:', time.time() - s)

        s = time.time()
        obs_dict: Dict = {
            **rgb_obs,
            **state_obs,
            **depth_obs,
            "robot_obs_raw": torch.from_numpy(obs["robot_obs"]).to(self.device),
        }
        # print(f'[DEBUG]: torch.from_numpy:', time.time() - s)
        return obs_dict

    def step(
        self, action_tensor: torch.Tensor
    ) -> Tuple[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], int, bool, Dict]:
        import time
        start = time.time()
        if self.relative_actions:
            action = action_tensor.squeeze().cpu().detach().numpy()
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor.squeeze().cpu().detach().numpy(), slice_ids)
        # print(f'[DEBUG]: detach.np:', time.time() - start)
        action[-1] = 1 if action[-1] > 0 else -1
        start = time.time()
        o, r, d, i = self.env.step(action)
        # print(f'[DEBUG]: env.step:', time.time() - start)

        start = time.time()
        obs = self.transform_observation(o)
        # obs = None
        # print(f'[DEBUG]: transform_obs:', time.time() - start)
        return obs, r, d, i

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()

        return self.transform_observation(obs)

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        obs = self.env.get_obs()
        return self.transform_observation(obs)
