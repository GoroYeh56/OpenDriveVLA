This is the steps to reproduce (infererence) this repository

1. Install packages: ✅
    
    ```bash
    conda create -n drivevla python=3.10 -y
    conda activate drivevla
    pip install --upgrade pip 
    
    # First install torch & torch-vision
    pip install torch==2.1.2
    pip install torchvision==0.16.2
        
    # Then, run pip install (comment out torch & numpy & flash-attn)
    pip install -e ".[train]" --no-build-isolation
    
    # Then, install flash-attention
    # Note: You'll have to set PIP_CACHE at /tmp to bypass the network volume error.
    mkdir -p /tmp/pip_cache
    export PIP_CACHE=/tmp/pip_cache
    pip install flash-attn==2.5.7 --no-build-isolation
    ```
    
2. Install MMCV with CUDA Support (Note: you **MUST specify CUDA_HOME**!) ✅
    - Thanks for this issue: https://github.com/DriveVLA/OpenDriveVLA/issues/21.for providing the command to install MMCV with CUDA (OPS)
    
    ```bash
    export CUDA_HOME=/usr/local/cuda
    
    cd OpenDriveVLA # cd to the root of this repo
    cd third_party/mmcv_1_7_2
    pip install -r requirements/optional.txt
    MMCV_WITH_OPS=1 MMCV_WITH_CUDA=1 pip install -v -e . --no-build-isolation
    ```
    
    Check if your MMCV has CUDA support
    
    ```bash
    python -c 'import mmcv; import mmcv.ops; print("MMCV Ops are installed.")'
    ```
    
    [Nov 26, 2025] YES!!! Finally have MMCV installed!!!
    
    ![Screen Shot 2025-11-26 at 4.08.01 PM.png](attachment:8e63a31d-fa7e-45bd-96f7-74c5fb90f592:Screen_Shot_2025-11-26_at_4.08.01_PM.png)
    
    Verify MMCV Custom Operations (ops) Import
    
    `python -c 'import mmcv.ops'` → Succeeded!
    
3. Install `mmdet` & `mmseg` (detection & segmentation libraries) ✅
    
    ```dart
    pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmengine==0.9.0 motmetrics==1.4.0 casadi==3.6.0
    ```
    
4. **Install mmdet3d from source code : In Progress** ⚠️
    
    ```bash
    cd DriveVLA # cd to the root of this repo
    cd third_party/mmdetection3d_1_0_0rc6
    pip install scipy==1.10.1 scikit-image==0.19.3 fsspec
    pip install . --no-build-isolation 
    # Key: --no-build-isolation is a MUST! Otherwise, you'll run into issue of "no module found as torch" since pip creates an isolated virtual environment by default. This is a GREAT lesson learned!
    
    ```
    
    Installing in progress …  (Nov 26, 2025) and Done!
    
    ![Screen Shot 2025-11-26 at 4.48.25 PM.png](attachment:2977a18a-eb18-44a8-a548-bfaefa95baca:Screen_Shot_2025-11-26_at_4.48.25_PM.png)
    
    ![Screen Shot 2025-11-26 at 4.49.15 PM.png](attachment:295a81b7-2633-4334-92a6-556b57d81dd2:Screen_Shot_2025-11-26_at_4.49.15_PM.png)
    
5. Download HuggingFace Checkopints ✅
    1. My HuggingFace [token](https://www.notion.so/My-Tasks-Tracker-298bd19b1e3880cba6caf039c9b20d3d?pvs=21)
    
    ```bash
    cd OpenDriveVLA/
    mkdir checkpoints
    # Download checkopints here
    ---
    ### Using GitHub Large File System (git LFS)
    # 1. First time: need to apt install
    sudo apt-get install git-lfs
    # password: av
    
    # 2. Initialize LFS
    git lfs install
    
    cd ~/OpenDriveVLA/checkpoints
    # 3. Clone the repository into a temporary folder named 'temp_repo'
    git clone https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B
    # prompt your Username: GoroYeh56 & huggingface token (Important!)
    ```
    
    Check: `models.savetensors` is there (~1.4 GB)
    
    ![Screen Shot 2025-11-26 at 4.28.35 PM.png](attachment:423fe196-fec7-410e-ba58-038ac673e6ad:Screen_Shot_2025-11-26_at_4.28.35_PM.png)
    
6. Download nuScenes Dataset (use mini as a practice) ✅
    - Follow [instructions](https://github.com/DriveVLA/OpenDriveVLA/blob/main/docs/2_DATA_PREP.md) here.
    
    ```bash
    cd OpenDriveVLA
    mkdir data && cd data
    
    wget https://www.nuscenes.org/data/v1.0-mini.tgz
    
    tar -xf v1.0-mini.tgz
    # -x: extract
    # -f: specify the filename (v1.0-mini.tgz)
    ```
    
    - Folder Structure
        
        ```bash
        OpenDriveVLA
        ├── data/
        │   ├── infos/
        │   │   ├── nuscenes_infos_temporal_train.pkl
        │   │   ├── nuscenes_infos_temporal_val.pkl
        │   ├── nuscenes/
        │   │   ├── can_bus/
        │   │   ├── maps/
        │   │   ├── samples/
        │   │   ├── sweeps/
        │   │   ├── v1.0-test/
        │   │   ├── v1.0-trainval/
        │   │   ├── cached_nuscenes_info.pkl
        ```
        
7. Run Inference Script
    
    ```bash
    cd OpenDriveVLA
    conda activate drivevla
    bash scripts/eval_drivevla.sh <CKPT_PATH> <NUM_GPU>
    
    For example:
    cd DriveVLA
    conda activate drivevla
    bash scripts/eval_drivevla.sh checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct 1
    ```
    
    ```bash
    cd OpenDriveVLA
    conda activate drivevla
    bash scripts/eval_drivevla.sh <CKPT_PATH> <NUM_GPU>
    
    # For example:
    bash scripts/eval_drivevla.sh checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct 1
    ```
    
    ### ⚠️ In Progress
    
    Nov 26, 2025
    
    ### Troubleshooting:
    
    1. ModuleNotFoundError: No module named `'llava.train'`
        1. Github issue https://github.com/DriveVLA/OpenDriveVLA/issues/24
    
    ```bash
    Traceback (most recent call last):
      File "/home/goro/OpenDriveVLA/drivevla/inference_drivevla.py", line 18, in <module>
        from llava.train.train import DataArguments
    ModuleNotFoundError: No module named 'llava.train'
    [2025-11-27 00:49:42,116] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 38152) of binary: /home/goro/miniconda3/envs/drivevla/bin/python3.10
    Traceback (most recent call last):
      File "/home/goro/miniconda3/envs/drivevla/bin/torchrun", line 7, in <module>
        sys.exit(main())
      File "/home/goro/miniconda3/envs/drivevla/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 346, in wrapper
        return f(*args, **kwargs)
      File "/home/goro/miniconda3/envs/drivevla/lib/python3.10/site-packages/torch/distributed/run.py", line 806, in main
        run(args)
      File "/home/goro/miniconda3/envs/drivevla/lib/python3.10/site-packages/torch/distributed/run.py", line 797, in run
        elastic_launch(
      File "/home/goro/miniconda3/envs/drivevla/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 134, in __call__
        return launch_agent(self._config, self._entrypoint, list(args))
      File "/home/goro/miniconda3/envs/drivevla/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 264, in launch_agent
        raise ChildFailedError(
    torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
    ============================================================
    
    ```
    
    ![Screen Shot 2025-11-26 at 4.50.41 PM.png](attachment:82158655-09c7-44ff-ac8b-99074753272c:Screen_Shot_2025-11-26_at_4.50.41_PM.png)