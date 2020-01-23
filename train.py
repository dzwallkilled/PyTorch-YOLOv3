from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate
from test_coco import evaluate_coco

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1, help="the num of GPUs to be used")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/rip/rip_model/yolov3-rip-level1.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/rip/rip_data_patches/rip_level1.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=800, help="size of each image dimension")
    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=10, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--output", type=str, default='output', help="the output folder")
    opt = parser.parse_args()
    opt.checkpoints = os.path.join(opt.output, "checkpoints")
    opt.logs = os.path.join(opt.output, "logs")
    print(opt)

    os.makedirs(opt.output, exist_ok=True)
    os.makedirs(opt.checkpoints, exist_ok=True)
    os.makedirs(opt.logs, exist_ok=True)

    logger = Logger(opt.logs)
    if opt.gpus > 1:
        raise NotImplementedError('multiple GPUs not supported yet.')
    device = get_available_device(opt.gpus)
    # device = get_available_device(0)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    data_root = data_config["root"]
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    # Get dataloader
    if 'rip' in opt.data_config:
        dataset = JSONDataset(data_root, train_path, remove_images_without_annotations=True,
                              img_size=opt.img_size,
                              augment=True, multiscale=opt.multiscale_training)
    else:
        dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    if opt.gpus > 1:
        model = nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.module.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(1, opt.epochs + 1):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, _ = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------
            if (batch_i % opt.print_freq == 0 and batch_i != 0) \
                    or batch_i == len(dataloader) - 1:
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
                yolo_layers = model.module.yolo_layers if opt.gpus > 1 else model.yolo_layers
                metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(yolo_layers))]]]

                # Log metrics at each YOLO layer
                for i, metric in enumerate(metrics):
                    formats = {m: "%.6f" for m in metrics}
                    formats["grid_size"] = "%2d"
                    formats["cls_acc"] = "%.2f%%"
                    row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in yolo_layers]
                    metric_table += [[metric, *row_metrics]]

                    # Tensorboard logging
                    tensorboard_log = []
                    for j, yolo in enumerate(yolo_layers):
                        for name, metric in yolo.metrics.items():
                            if name != "grid_size":
                                tensorboard_log += [(f"{name}_{j + 1}", metric)]
                    tensorboard_log += [("loss", loss.item())]
                    logger.list_of_scalars_summary(tensorboard_log, batches_done)

                log_str += AsciiTable(metric_table).table
                log_str += f"\nTotal loss {loss.item()}"

                # Determine approximate time left for epoch
                epoch_batches_left = len(dataloader) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                log_str += f"\n---- ETA {time_left}"

                print(log_str)

            if opt.gpus > 1:
                model.module.seen += imgs.size(0)
            else:
                model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0 or epoch == opt.epochs:
            print("\n---- Evaluating Model ----")
            evaluate_coco(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
                device=device
            )

            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=opt.batch_size,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0:
            if opt.gpus > 1:
                torch.save(model.module.state_dict(), f"{opt.checkpoints}/yolov3_ckpt_%d.pth" % epoch)
            else:
                torch.save(model.state_dict(), f"{opt.checkpoints}/yolov3_ckpt_%d.pth" % epoch)
