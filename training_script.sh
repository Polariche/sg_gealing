 python train.py --outdir=~/training-runs --cfg=stylegan2 \
 --resume=https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-cat-config-f.pkl \
 --gpus=1 --batch=8 --gamma=10 --mirror=1 --aug=noaug --kimg=5000 \
 --glr 0.001