from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
import random
import os

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import numpy as np

from mdt.datasets.base_dataset import BaseDataset
from mdt.datasets.utils.episode_utils import lookup_naming_pattern

logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class DiskDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_text = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + window_size
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key in keys}
        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
        return episode

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_text = lang_data["language"]["ann"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_text

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)


class ExtendedDiskDataset(DiskDataset):
    def __init__(
        self,
        *args: Any,
        obs_seq_len: int,
        action_seq_len: int,
        future_range: int,
        img_gen_frame_diff: int = 3,
        use_extracted_rel_actions: bool = False,
        extracted_dir: str = 'extracted/',
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.future_range = future_range  # Number of steps into the future to sample goals
        self.ep_start_end_ids = np.load(self.abs_datasets_dir / "ep_start_end_ids.npy")  # Load sequence boundaries
        self.img_gen_frame_diff = img_gen_frame_diff
        self.random_frame_diff = False if img_gen_frame_diff > -1 else True 
        # self.min_window_size = self.action_seq_len
        # self.max_window_size = self.action_seq_len + self.future_range

        # Using extracted npy to reduce bandwidth of data loading
        self.use_extracted_rel_actions = use_extracted_rel_actions
        if use_extracted_rel_actions:
            self.extracted_dir = extracted_dir
            if not os.path.exists(extracted_dir):  # maybe a relative path
                self.extracted_dir = os.path.join(self.abs_datasets_dir, "extracted")  # convert to abs path
                assert os.path.exists(self.extracted_dir), "extracted dir not found!"
            with open(os.path.join(self.extracted_dir, "ep_npz_names.list"), "r") as f:
                self.extracted_ep_npz_names = [int(x.strip()) for x in f.readlines()]
                self.extracted_ep_npz_name_to_npy_idx = {self.extracted_ep_npz_names[i]: i
                                                         for i in range(len(self.extracted_ep_npz_names))}
                # key: int, original episode fn's index; value: int, extracted npy's inner index
            self.extracted_ep_rel_actions: np.ndarray = np.load(os.path.join(self.extracted_dir, "ep_rel_actions.npy"))
            logger.info(f"Extracted files loaded from {self.extracted_dir}")
        
    def find_sequence_boundaries(self, idx: int) -> Tuple[int, int]:
        for start_idx, end_idx in self.ep_start_end_ids:
            if start_idx <= idx < end_idx:
                return start_idx, end_idx
        raise ValueError(f"Index {idx} does not belong to any sequence.")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + self.action_seq_len + self.obs_seq_len-1
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        # keys:['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper', 'robot_obs', 'rel_actions', 'scene_obs']

        # Modify the episode dict to only include the specified sequence lengths
        if self.random_frame_diff:
            img_gen_frame_diff = random.randint(0, self.action_seq_len - 1)
        else:
            img_gen_frame_diff = self.img_gen_frame_diff
        gen_img_idx = start_idx + self.obs_seq_len + img_gen_frame_diff - 1

        if not self.use_extracted_rel_actions:
            # Op1. original reading actions from episode_xxx.npz one-by-one
            episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        else:
            # Op2. reading actions from a single ep_rel_actions.npy file
            episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in
                        range(start_idx, start_idx + self.obs_seq_len)]
            gen_img_episode = self.load_file(self._get_episode_name(gen_img_idx))
            ex_indices = [self.extracted_ep_npz_name_to_npy_idx[file_idx] for file_idx in range(start_idx, end_idx)]
            ex_actions = self.extracted_ep_rel_actions[ex_indices, :]

        episode = {}
        for key in keys:
            if 'gen' in key:
                continue
            
            stacked_data = np.stack([ep[key] for ep in episodes])
            if not self.use_extracted_rel_actions:
                # Op1. original reading actions from episode_xxx.npz one-by-one
                if key == "rel_actions" or key == 'actions':
                    episode[key] = stacked_data[(self.obs_seq_len-1):((self.obs_seq_len-1) + self.action_seq_len), :]
                else:
                    if key == 'rgb_static':
                       gen_img_static = stacked_data[self.obs_seq_len + img_gen_frame_diff - 1, :]
                    elif key == 'rgb_gripper':
                        gen_img_gripper = stacked_data[self.obs_seq_len + img_gen_frame_diff -1, :]
                    episode[key] = stacked_data[:self.obs_seq_len, :]
            else:
                # Op2. reading actions from a single ep_rel_actions.npy file
                if key == "rel_actions" or key == 'actions':
                    episode[key] = ex_actions[(self.obs_seq_len - 1):((self.obs_seq_len - 1) + self.action_seq_len), :]
                else:
                    if key == 'rgb_static':
                        gen_img_static = gen_img_episode[key]
                    elif key == 'rgb_gripper':
                        gen_img_gripper = gen_img_episode[key]

                    episode[key] = stacked_data[:self.obs_seq_len, :]

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
            episode["language_text"] = self.lang_text[self.lang_lookup[idx]] #[0]  # TODO check  [0]
        
        # get the random future state as goal
        goal_idx = end_idx + window_size
        # print(start_idx, end_idx, goal_idx)
        eps_start_idx, eps_end_idx = self.find_sequence_boundaries(end_idx)

        # Check if future goal can be sampled

        if eps_end_idx < goal_idx:
            goal_idx = eps_end_idx
        
        goal_episodes = self.load_file(self._get_episode_name(goal_idx))
        goal_episode = {}
        for key in keys:
            if 'gen' in key:
                continue
            goal_stacked_data = np.stack([goal_episodes[key]])
            if key == "rel_actions" or key == 'actions':
                pass
            else:
                goal_episode[key] = goal_stacked_data[:self.obs_seq_len, :]
        # store for merging
        
        episode = self.merge_episodes(episode, goal_episode)
        episode['gen_static'] = gen_img_static
        episode['gen_gripper'] = gen_img_gripper
        episode['future_frame_diff'] = np.array(img_gen_frame_diff)
        return episode
       
    
    def merge_episodes(self, episode1: Dict[str, np.ndarray], episode2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_episode = {}
        all_keys = set(episode1.keys()).union(set(episode2.keys()))
        for key in all_keys:
            if key in episode1 and key in episode2:
                # Merge logic here, for example:
                merged_episode[key] = np.concatenate([episode1[key], episode2[key]], axis=0)
            elif key in episode1:
                merged_episode[key] = episode1[key]
            else:
                merged_episode[key] = episode2[key]
        return merged_episode
    
    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)

