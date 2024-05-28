#!/bin/bash
python legged_gym/scripts/train_teacher.py --headless --sim_device=cuda:0 --max_iterations 3000 --checkpoint 3000 --resume