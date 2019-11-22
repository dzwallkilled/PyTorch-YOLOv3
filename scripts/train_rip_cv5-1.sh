#!/bin/bash

export PYTHONPATH="$(pwd)"

fold=1
level=full

name="rip_cv5-$fold"
dirname="/data2/data2/zewei/exp/RipData/YOLOv3/$level/CV5-$fold"

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
  --pretrained_weights weights/yolov3.weights \
  --output $dirname \
  2>&1 | tee -a $dirname/$name.log \
  "$@"