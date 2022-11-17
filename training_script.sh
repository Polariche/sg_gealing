 python train.py --outdir=~/training-runs --cfg=stylegan2 \
 --resume=./pretrained/stylegan2-cat-config-f.pkl \
 --gpus=1 --batch=4 --gamma=10 --mirror=1 --aug=noaug --kimg=5000 \
 --glr 0.002 --tick 0.1 --snap 10