from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()
    dataset = JSONDataset('/data2/data2/zewei/data/RipData/RipTrainingAllData', path, True,
                          augment=False, multiscale=False, img_size=img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn
    )
    predicts = []
    for batch_i, (img_ids, imgs, _) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        imgs = imgs.to(device)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        predicts += get_batch_statistics_coco(outputs, img_ids, iou_threshold=iou_thres)

    pred_file = 'results.json'
    json.dump(predicts, open(pred_file, 'w'))
    coco_eval = do_coco_evaluation(gt_file=path, pred_file=pred_file)

    return


def xyxy2coco(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def get_batch_statistics_coco(outputs, img_ids, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    predicts = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        image_id = img_ids[sample_i]

        output = outputs[sample_i]
        pred_boxes = xyxy2coco(output[:, :4])
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        for pred_i, (pred_box, pred_label, pred_score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            predicts.append({'image_id': image_id,
                             'category_id': int(pred_label.item()),
                             'bbox': pred_box.tolist(),
                             'score': pred_score.item()})

    return predicts


def do_coco_evaluation(gt_file, pred_file):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(gt_file)  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(pred_file)  # initialize COCO pred api

    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/rip/yolov3-rip-detect.cfg",
                        help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/rip/rip_cv5/rip_cv5-1.data",
                        help="path to data config file")
    parser.add_argument("--weights_path", type=str,
                        default="/home/zd027/exp/RipData/YOLOv3/CV5-1/checkpoints/yolov3_ckpt_90.pth",
                        help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/rip/rip.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--gpus", type=int, default=1, help="the num of GPUs to be used")
    opt = parser.parse_args()
    print(opt)

    device = get_available_device(opt.gpus)

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=8,
    )

