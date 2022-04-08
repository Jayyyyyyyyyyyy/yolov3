#!/bin/sh
# nohup /home/jiangchenxi/anaconda3/envs/openjay/bin/python trainvim.py runserver 0.0.0.0:9002

#nohup python -m torch.distributed.launch --nproc_per_node 8 train.py --batch 64 --imgsz 320 --data coco.yaml --cfg models/yolov3-tiny.yaml --weights '' --device 0,1,2,3,4,5,6,7 --sync-bn >>nohupout.log 2>&1 &
nohup python -m torch.distributed.launch --nproc_per_node 8 train.py --epochs 100 --batch 128 --imgsz 320 --data dataset.yaml --cfg models/yolov3-tiny.yaml --adam --single-cls --weights '' --device 0,1,2,3,4,5,6,7 --sync-bn --single-channel >>nohupout.log 2>&1 &

# watch -n 0.1 -d nvidia-smi
~



