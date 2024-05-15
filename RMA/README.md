## Requirements
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
    - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
4. Install other dependence
   -  `pip install -r requirements.txt` 
5. Troubleshooting
   - `sudo apt-get update`
   - `sudo apt-get install build-essential --fix-missing`
   - `pip install setuptools==59.5.0`

## Train Teacher
```
python legged_gym/scripts/train_teacher.py --task=a1 --headless --sim_device=cuda:0
```

## Train Student
**Please make sure you have trained the teacher before training Student.**
```
python legged_gym/scripts/train_student.py --task=a1 --headless --sim_device=cuda:0
```
