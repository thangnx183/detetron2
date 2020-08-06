#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import cv2
from pycocotools import coco
import numpy as np
import hypertune
import argparse


import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetCatalog,
)

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, launch,DefaultPredictor
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask

logger = logging.getLogger("detectron2")

def get_gt_mask(json_coco,id):
    img = json_coco.loadImgs(id)[0]
    ann_ids = json_coco.getAnnIds(img['id'],catIds=[0])
    anns = json_coco.loadAnns(ann_ids)

    gt_mask = np.zeros((img['height'],img['width']))

    check = 0
    for i in range(len(anns)):
        check += anns[i]['image_id'] - anns[0]['image_id']

    if check != 0:
        return np.array([-1])
    else :
        for i in range(len(anns)):
            gt_mask += json_coco.annToMask(anns[i])
        
    return (gt_mask > 0)

def dice_coe(mask, pred):
    intersect = mask[pred == 1]
    dice = (2*intersect.sum()+1e-12) / (mask.sum()+ pred.sum()+1e-12)

    return dice

def do_test(cfg,model):
    val_metadata = MetadataCatalog.get('scratch_val')
    val_dicts = DatasetCatalog.get('scratch_val')
    
    fs = coco.COCO('./datasets/coco/annotations/instances_val.json')
    predictor = DefaultPredictor(cfg)

    dice_sum = 0
    print('evaluating .......')
    print('len validation:',len(val_dicts))
    num = 0
    #print(val_dicts[302]['annotations'])
    #return 0
    for i in val_dicts:
        img = cv2.imread(i['file_name'])
        gt_mask = get_gt_mask(fs,i['image_id'])

        if (gt_mask.sum() == -1):
            continue
        
        out = predictor(img)
        pred = torch.sum(out['instances'].pred_masks,dim=0) > 0
        pred = pred.cpu().detach().numpy()

        dice_sum += dice_coe(gt_mask,pred)
        num += 1

        if 0:
            visualizer = Visualizer(img[:,:,::-1], metadata=val_metadata,instance_mode=ColorMode.IMAGE_BW)
            #vis = visualizer.draw_dataset_dict(i)
            vis = visualizer.draw_instance_predictions(out['instances'].to('cpu'))
            #print('mask :',torch.sum(out['instances'].pred_masks,dim=0).shape)
            cv2.imwrite('./result/'+i['file_name'][23:],vis.get_image()[:,:,::-1])
    print('test samples : 'num)
    dice = dice_sum/num
    print('dice coefficent: ',dice)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='dice_tuning_tag',
        metric_value=dice,
        global_step=1
    )

    return dice


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg,model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_file("./config/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file('./config/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    cfg.merge_from_list(args.opts)

    #cfg.TEST.EVAL_PERIOD =  2000

    #---- my own config
    num_classes = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    
    #---- tuning config
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MOMENTUM = args.momentum

    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN  = args.PRE_NMS_TOPK_TRAIN
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST   = args.PRE_NMS_TOPK_TEST

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = args.POST_NMS_TOPK_TRAIN
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = args.POST_NMS_TOPK_TEST

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.NMS_THRESH_TEST

    cfg.freeze()
    default_setup(
        cfg, args
    )  
    return cfg


def main(args):


    cfg = setup(args)

    #---- train set
    name_train = 'scratch_train'
    json_train_file  = './datasets/coco/annotations/instances_train.json'
    image_train_root = './datasets/coco/images'

    #---- validation set
    name_val = 'scratch_val'
    json_val_file ='./datasets/coco/annotations/instances_val.json'
    image_val_root = './datasets/coco/images'

    #---- test set
    #name_test = 'scratch_test'
    #json_test_file ='./datasets/coco/annotations/instances_test.json'
    #image_test_root = './datasets/coco/images'

    #---- register all dataset
    register_coco_instances(name_train,{},json_train_file,image_train_root)
    register_coco_instances(name_val,{},json_val_file,image_val_root)
    #register_coco_instances(name_test,{},json_test_file,image_test_root)

    model = build_model(cfg)
    #logger.info("Model:\n{}".format(model))

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        #print('+'*20)
        return do_test(cfg,model)
        
    #print('/'*20)
    
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg)


if __name__ == "__main__":
    #args = default_argument_parser().parse_args()
    args = default_argument_parser()

    # args.add_argument(
    #     "--config_file",
    #     metavar="FILE",
    #     type=str,
    #     default= "./config/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    # )

    #args.config_file.default = "./config/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    #args.num_gpus.default = 2

    # args.add_argument(
    #     "--num-gpus",
    #     type=int,
    #     default=2,
    # )

    args.add_argument(
        "--lr",
        type =float,
        default=0.00025, 
    )

    args.add_argument(
        "--momentum",
        type = float,
        default = 0.9,
    )

    args.add_argument(
        "--PRE_NMS_TOPK_TRAIN",
        type=int,
        default=2000,
    )

    args.add_argument(
        "--PRE_NMS_TOPK_TEST",
        type=int,
        default=1000,
    )

    args.add_argument(
        "--POST_NMS_TOPK_TRAIN",
        type=int,
        default=1000,
    )

    args.add_argument(
        "--POST_NMS_TOPK_TEST",
        type=int,
        default=1000, 
    )

    args.add_argument(
        "--SCORE_THRESH_TEST",
        type=float,
        default=0.25,
    )

    args.add_argument(
        "--NMS_THRESH_TEST",
        type=float,
        default=0.3,
    )

    args = args.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
