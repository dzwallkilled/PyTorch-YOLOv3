#!/bin/bash

export PYTHONPATH="$(pwd)"

fold=3

name="rip_cv5-$fold"
dirname="/data2/data2/zewei/exp/RipData/YOLOv3/CV5-$fold"

echo $name
echo "output" $dirname

if [ ! -d "$dirname" ]
then
    echo "$dirname doesn't exist. Creating now"
    mkdir $dirname
    echo "File created"
else
    echo "$dirname exists"
fi

python -u train.py \
  --data_config config/rip/rip_cv5/$name.data \
  --model_def config/rip/yolov3-rip.cfg \
  --pretrained_weights checkpoints/yolov3_ckpt_490.pth \
  --output $dirname \
  2>&1 | tee -a /data2/data2/zewei/exp/RipData/YOLOv3/CV5-1/$name.log \
  "$@"