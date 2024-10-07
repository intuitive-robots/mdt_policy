import os
import time
import copy
import argparse
from typing import List
import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')


class DiskDataset(Dataset):
    def __init__(self, in_root,
                 task='task_D_D',
                 split='training',
                 target_key='rel_actions'
                 ):
        self.ori_path = os.path.join(in_root, task, split)
        print(f'[DiskDataset] Reading data from disk, this might take a while...')
        ep_npz_names = [int(x.split('_')[1].split('.')[0]) for x in
                        os.listdir(self.ori_path) if 'episode' in x]
        ep_npz_names.sort()  # To make sure the order is same across different systems
        self.ep_npz_names = ep_npz_names
        self.target_key = target_key
    def __len__(self):
        return len(self.ep_npz_names)
    def __getitem__(self, idx):
        npz = np.load(
            os.path.join(self.ori_path,
                         f"episode_{self.ep_npz_names[idx]:07d}.npz"), allow_pickle=True)
        s_data = copy.deepcopy(npz[self.target_key])
        del npz
        return {
            "data": s_data,
            "idx": idx,
            "ep_npz_idx": self.ep_npz_names[idx],
        }


@torch.no_grad()
def diskdata_extract(
        root,
        task='task_D_D',  # calvin_debug_dataset, task_D_D
        split='training',  # training, validation
        extract_key='rel_actions',
        out_dir='',
        ep_npz_names_fn: str = 'ep_npz_names.list',
        bs: int = 128,
        force_overwrite: bool = False,
):
    """ Extract key data from disk dataset """
    # 1. Read data
    in_abs = os.path.join(root, task, split)
    if not os.path.exists(in_abs):
        print(f'[diskdata_extract][Warning] {in_abs} does not exist, skipping!')
        return
    os.makedirs(out_dir, exist_ok=True)
    extracted_save_path = os.path.join(root, task, split, 'extracted', f"ep_{extract_key}.npy")
    if os.path.exists(extracted_save_path):
        if not force_overwrite:
            print(f'[diskdata_extract][Warning] {extracted_save_path} already exists, skipping!')
            return
        print(f'[diskdata_extract][Warning] {extracted_save_path} already exists, going to cover it!')

    disk_dataset = DiskDataset(in_root=root, task=task, split=split, target_key=extract_key)
    dataset_len = len(disk_dataset)
    print(f'[diskdata_extract] dataset_len={dataset_len}, in_abs={in_abs}, out_abs={out_dir}')

    ep_npz_names = [str(x) for x in disk_dataset.ep_npz_names]
    with open(os.path.join(out_dir, ep_npz_names_fn), 'w') as f:
        f.write('\n'.join(ep_npz_names))
    print(f'[diskdata_extract] ep_npz_names saved to:{os.path.join(out_dir, ep_npz_names_fn)}, len={len(ep_npz_names)}')

    train_dataloader = DataLoader(
        disk_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=24,
    )

    # 2. Iterate data
    all_data = []
    start_time = time.time()
    for idx, data in enumerate(tqdm(train_dataloader)):
        data = data['data']
        targets = data
        for b_idx in range(targets.shape[0]):
            target: torch.Tensor = targets[b_idx]  # remove batch-dim
            target: np.ndarray = target.cpu().numpy()
            all_data.append(copy.deepcopy(target))
            del target
    print(f'[diskdata_extract] DiskData avg Speed (bs={bs}): '
          f'{len(all_data) / (time.time() - start_time)}data/s, total={len(all_data)}')

    # 3. Save data
    all_data = np.stack(all_data, axis=0)
    np.save(extracted_save_path, all_data)
    print(f'[diskdata_extract] Extracted Data: {all_data.shape}, saved_to={extracted_save_path}')

    # 4. Check data
    with open(os.path.join(out_dir, ep_npz_names_fn), 'r') as f:
        out_ep_npz_names = f.readlines()  # out_dir, str increasing order
    ep_npz_name_to_npy_idx = {int(out_ep_npz_names[i].strip()): i for i in range(len(out_ep_npz_names))}  # order mapping

    in_root_ep_npz_indices = [int(x.split('_')[1].split('.')[0]) for x in
                            os.listdir(in_abs) if 'episode' in x]  # episode_{x:07d}.npz
    check_sources = [0, len(in_root_ep_npz_indices) // 2, len(in_root_ep_npz_indices) - 1] \
                     + np.random.randint(0, len(in_root_ep_npz_indices), size=10).tolist()  # in_root, directory order
    for i in range(len(check_sources)):
        si = check_sources[i]
        s_ep_npz_name = in_root_ep_npz_indices[si]
        s_fn = f'episode_{s_ep_npz_name:07d}.npz'
        s_data = np.load(os.path.join(in_abs, s_fn), allow_pickle=True)[extract_key]
        t_data = np.load(extracted_save_path)[ep_npz_name_to_npy_idx[s_ep_npz_name]]
        if (s_data - t_data).sum() != 0.:
            print(f'[diskdata_extract] Check ERROR@{i}! saved.npy != ori/ep_*.npz, error={(s_data - t_data).sum()}')
    print(f'[diskdata_extract] Check OK on {check_sources}!')
    print('*' * 30 + ' End ' + '*' * 30)


if __name__ == '__main__':
    '''
    Usage example:
        python mdt/datasets/preprocess/extract_by_key.py -i /home/geyuan/local_soft/ \
            --in_task all
    Params:
        in_root: /YOUR/PATH/TO/CALVIN/, e.g /home/geyuan/datasets/CALVIN/dataset, /data3/geyuan/datasets/CALVIN
        extract_key: A key of 'dict(episode_xxx.npz)'
        out_dir: if None, output will be saved to {in_root}/{in_task}/{in_split}/extracted/ep_{extract_key}.npy
        force: whether to overwrite existing extracted data
    '''
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--in_root', type=str,
                      help='This path contains task_XXX_D/ directories.')
    args.add_argument('--in_task', type=str, default='all',
                      choices=['all', 'task_D_D', 'task_ABC_D', 'task_ABCD_D', 'calvin_debug_dataset'])
    args.add_argument('--in_split', type=str, default='all',
                      choices=['all', 'training', 'validation'])
    args.add_argument('-k', '--extract_key', type=str, default='rel_actions',
                      help='Which key to extract from episode_xxx.npz .')
    args.add_argument('-o', '--out_dir', type=str, default=None,
                      help='The extracted *.npy will be saved under this directory. '
                           'If None, *.npy will be saved '
                           'as {in_root}/{in_task}/{in_split}/extracted/ep_{extract_key}.npy .'
                           'Otherwise, the saving dir will '
                           'be {out_dir}/{in_task}/{in_split}/extracted}')
    args.add_argument('-f', '--force', action='store_true',
                      help='Force overwrite existing npy files.')
    opts = args.parse_args()

    # preprocess input args
    if opts.in_task == 'all':
        tasks = ['calvin_debug_dataset', 'task_D_D', 'task_ABC_D', 'task_ABCD_D']
    else:
        tasks = [opts.in_task]

    if opts.in_split == 'all':
        splits = ['training', 'validation']
    else:
        splits = [opts.in_split]

    # start extracting
    for task in tasks:
        for split in splits:
            if opts.out_dir is None:
                out_dir = str(os.path.join(opts.in_root, task, split, 'extracted'))
                # i.e {in_root}/{in_task}/{in_split}/extracted/
            else:
                out_dir = str(os.path.join(opts.out_dir, task, split, 'extracted'))
                # i.e {out_dir}/{in_task}/{in_split}/extracted/
            diskdata_extract(opts.in_root, task, split,
                             opts.extract_key,
                             out_dir,
                             force_overwrite=opts.force,
                             )
