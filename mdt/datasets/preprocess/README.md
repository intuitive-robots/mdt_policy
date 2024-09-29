# Preprocess Tools

Since MDT needs to load 10 episodes' action data in each inference, it 
requires a large bandwidth (usually ~2000MB/iteration). 
This significantly reduces the GPU utilization rate during training. 
Therefore, you can use the script `extract_by_key.py` to 
extract the data into a single file, avoiding 
opening too many episode files when using CALVIN dataset.

### Usage example:

```shell
python mdt/datasets/preprocess/extract_by_key.py -i /home/geyuan/local_soft/ \
    --in_task all
```

### Params:

Run this command to see more detailed information: 

```shell
python mdt/datasets/preprocess/extract_by_key.py -h
```

Important params:

* `--in_root`: `/YOUR/PATH/TO/CALVIN/`, e.g `/data3/geyuan/datasets/CALVIN/`
* `--extract_key`: A key of `dict(episode_xxx.npz)`, default is **'rel_actions'**, the saved file name depends on this (i.e `ep_{extract_key}.npy`)

Optional params:

* `--in_task`: default is **'all'**, meaning all task folders (e.g `task_ABCD_D/`) of CALVIN
* `--in_split`: default is **'all'**, meaning both `training/` and `validation/`
* `--out_dir`: optional, default is **'None'**, and will be converted to `{in_root}/{in_task}/{in_split}/extracted/`
* `--force`: whether to overwrite existing extracted data