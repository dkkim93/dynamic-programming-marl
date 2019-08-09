#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Tensorboard
pkill tensorboard
# rm -rf logs/tb*
tensorboard --logdir logs/ &

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Train tf 
print_header "Training network"
cd $DIR

python3.6 main.py \
--env-name "Gridworld-v0" \
--ep-max-timesteps 100 \
--future-max-timesteps 0 \
--estimate-option "montecarlo" \
--decay-max-timesteps 100 \
--row 9 \
--prefix ""

python3.6 main.py \
--env-name "Gridworld-v0" \
--ep-max-timesteps 100 \
--future-max-timesteps 0 \
--estimate-option "naive" \
--decay-max-timesteps 100 \
--row 9 \
--prefix ""

# Begin experiment
for i in {1..20}
do
    python3.6 main.py \
    --env-name "Gridworld-v0" \
    --ep-max-timesteps 100 \
    --future-max-timesteps $i \
    --estimate-option "ours" \
    --decay-max-timesteps 100 \
    --row 9 \
    --prefix ""
done
