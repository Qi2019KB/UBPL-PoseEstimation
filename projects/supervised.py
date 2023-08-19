# -*- coding: utf-8 -*-
import os
import random
import argparse
import datetime
import numpy as np
import torch
from torch.optim.adamw import AdamW as TorchAdam
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import GLOB as glob
import datasources
import datasets
import models as modelClass

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc
from utils.parameters import consWeight_increase
from utils.losses import JointMSELoss, AvgCounter, AvgCounters
from projects.tools import ProjectTools as proj
from utils.evaluation import EvaluationUtils as eval


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args.start_epoch = 0
    args.best_acc = -1.
    args.best_epoch = 0

    # region 1. Initialize
    # Data loading
    dataSource = datasources.__dict__[args.dataSource]()
    semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs, means, stds = dataSource.getSemiData(args.trainCount, args.validCount, args.labelRatio)
    args.kpsCount, args.imgType, args.pck_ref, args.pck_thr, args.inpRes, args.outRes = dataSource.kpsCount, dataSource.imgType, dataSource.pck_ref, dataSource.pck_thr, dataSource.inpRes, dataSource.outRes

    # Model initialize
    model = modelClass.__dict__["PoseModel"](modelType=args.model, kpsCount=args.kpsCount, mode=args.feature_mode).to(args.device)
    optim = TorchAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model_pNum = sum(p.numel() for p in model.parameters())
    args.nStack = model.nStack
    logc = "=> initialized {} model (params: {})".format(args.model, format(model_pNum / (1024**2), ".2f"))
    logger.print("L1", logc)

    # Dataloader initialize
    # region trainDS initialize by first way
    # trainDS = datasets.__dict__["DS"]("train", labeledData, means, stds, isAug=True, isDraw=args.debug, **vars(args))
    # trainLoader = TorchDataLoader(trainDS, batch_size=args.trainBS, shuffle=True, pin_memory=True, drop_last=False)
    # endregion

    # region trainDS initialize by second way
    sampler = SubsetRandomSampler(labeledIdxs)
    trainDS = datasets.__dict__["DS"]("train", semiTrainData, means, stds, isAug=True, isDraw=args.debug, **vars(args))
    batchSampler = BatchSampler(sampler, args.trainBS, drop_last=True)
    trainLoader = torch.utils.data.DataLoader(trainDS, batch_sampler=batchSampler, pin_memory=True)
    # endregion

    validDS = datasets.__dict__["DS"]("valid", validData, means, stds, isAug=False, isDraw=args.debug, **vars(args))
    validLoader = TorchDataLoader(validDS, batch_size=args.inferBS, shuffle=True, pin_memory=True, drop_last=False)
    logger.print("L1", "=> initialized {} Dataloaders".format(args.dataSource))
    # endregion

    # region 2. Iteration
    logger.print("L1", "=> training start")
    for epo in range(args.epochs):
        epoTM = datetime.datetime.now()
        args.epo = epo

        # region 2.1 model training and validating
        startTM = datetime.datetime.now()
        pec_loss = train(trainLoader, model, optim, args)
        logger.print("L3", "model training finished...", start=startTM)
        startTM = datetime.datetime.now()
        predsArray, accs, errs = validate(validLoader, model, args)
        logger.print("L3", "model validating finished...", start=startTM)
        # endregion

        # region 2.2 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = accs[-1] > args.best_acc
        if is_best:
            args.best_epoch = epo
            args.best_acc = accs[-1]
        # model storage
        checkpoint = {"current_epoch": epo, "best_acc": args.best_acc, "best_epoch": args.best_epoch,
                      "model": args.model, "model_state": model.state_dict(), "optim_state": optim.state_dict()}
        comm.ckpt_save(checkpoint, is_best, ckptPath="{}/ckpts".format(args.basePath))
        logger.print("L3", "model storage finished...", start=startTM)
        # endregion

        # region 2.3 log storage
        startTM = datetime.datetime.now()
        # Initialization parameter storage
        if epo == args.start_epoch:
            save_args = vars(args).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)

        # Log data storage
        log_data = {"pec_loss": pec_loss, "accs": accs, "errs": errs}
        comm.json_save(log_data, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)

        # Pseudo-labels data storage
        pseudo_data = {"predsArray": predsArray}
        comm.json_save(pseudo_data, "{}/logs/pseudoData/pseudoData_{}.json".format(args.basePath, epo+1), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 2.4 output result
        fmtc = "[{}/{} | lr: {}] | pec_loss: {} | best acc: {} (epo: {}) | acc: {}, err: {} | kps_acc:[{}], kps_err:[{}]"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"),
                           format(optim.state_dict()['param_groups'][0]['lr'], ".8f"),
                           format(pec_loss, ".5f"),
                           format(args.best_acc, ".5f"), format(args.best_epoch + 1, "3d"),
                           format(accs[-1], ".5f"), format(errs[-1], ".3f"),
                           proj.setContent(accs[0:len(accs)-1], ".5f"),
                           proj.setContent(errs[0:len(errs)-1], ".3f"))
        logger.print("L1", logc)

        # Epoch line
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
        fmtc = "[{}/{} | {}] ---------- ---------- ---------- ---------- ---------- ---------- ----------"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), time_interval)
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion

    logger.print("L1", "[{}, all executing finished...]".format(args.experiment), start=allTM)


def train(trainLoader, model, optim, args):
    # region 1. Preparation
    pec_counter = AvgCounter()
    pose_criterion = JointMSELoss(nStack=args.nStack).to(args.device)
    model.train()
    # endregion

    # region 2. Training
    for bat, (imgMap, heatmap, meta) in enumerate(trainLoader):
        optim.zero_grad()

        # region 2.1 data organize
        imgMap = proj.setVariable(imgMap, args.device)  # [bs, 3, 256, 256]
        heatmap = proj.setVariable(heatmap, args.device)
        bs, k, _, _ = heatmap.shape
        # endregion

        # region 2.2 Test
        # endregion

        # region 2.3 model forward
        outs = model(imgMap) if args.feature_mode == "default" else model(imgMap)[0]
        # endregion

        # region 2.4 pose estimation constraint
        pec_sum, pec_count = pose_criterion(outs, heatmap)
        pec_loss = args.poseWeight * ((pec_sum / pec_count) if pec_count > 0 else pec_sum)
        pec_counter.update(pec_loss.item(), pec_count)
        # endregion

        # region 2.5 calculate total loss & update model
        total_loss = pec_loss
        total_loss.backward()
        optim.step()
        # endregion

        # region 2.6 clearing the GPU Cache
        del outs, pec_loss, total_loss
        # endregion
    # endregion
    return pec_counter.avg


def validate(validLoader, model, args):
    # region 1. Preparation
    acc_counters = AvgCounters()
    err_counters = AvgCounters()
    predsArray = []
    model.eval()
    # endregion

    # region 2. Validating
    with torch.no_grad():
        for bat, (imgMap, heatmap, meta) in enumerate(validLoader):
            # region 2.1 data organize
            imgMap = proj.setVariable(imgMap, args.device, False)
            bs, k, _, _ = heatmap.shape
            # endregion

            # region 2.2 model predict
            # model forward
            outs = model(imgMap) if args.feature_mode == "default" else model(imgMap)[0]
            preds, _ = proc.kps_fromHeatmap(outs[:, -1].cpu().detach(), meta["center"], meta["scale"], [args.outRes, args.outRes])
            predsArray += preds.clone().data.numpy().tolist()
            # endregion

            # region 2.3 calculate the error and accuracy
            errs, accs = eval.acc_pck(preds, meta['kpsMap'], args.pck_ref, args.pck_thr)
            for idx in range(k+1):
                acc_counters.update(idx, accs[idx].item(), bs if idx < k else bs*k)
                err_counters.update(idx, errs[idx].item(), bs if idx < k else bs*k)
            # endregion

            # region 2.4 clearing the GPU Cache
            del outs
            # endregion
    return predsArray, acc_counters.avg(), err_counters.avg()


def setArgs(args, params):
    dict_args = vars(args)
    if params is not None:
        for key in params.keys():
            if key in dict_args.keys():
                dict_args[key] = params[key]
    for key in dict_args.keys():
        if dict_args[key] == "True": dict_args[key] = True
        if dict_args[key] == "False": dict_args[key] = False
    return argparse.Namespace(**dict_args)


def exec(expMark="MT", params=None):
    random_seed = 1388
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = initArgs(params)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.experiment = "{}({}_{})_{}_{}".format(args.dataSource, args.trainCount, args.labelRatio, expMark, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    args.basePath = "{}/{}".format(glob.expr, args.experiment)
    glob.setValue("logger", Logger(args.experiment, consoleLevel="L1"))
    main(args)


def initArgs(params=None):
    # region 1. Parameters
    parser = argparse.ArgumentParser(description="Pose Estimation")

    # Model setting
    parser.add_argument("--model", default="HG3", choices=["HG3", "HG2", "LitePose"])
    parser.add_argument("--feature_mode", default="AvgPool", choices=["default", "MaxPool", "AvgPool", "ConvOne"])

    # Dataset setting
    parser.add_argument("--dataSource", default="Mouse", choices=["Mouse", "FLIC", "LSP"])
    parser.add_argument("--trainCount", default=100, type=int)
    parser.add_argument("--validCount", default=500, type=int)
    parser.add_argument("--labelRatio", default=0.3, type=float)

    # Training strategy
    parser.add_argument("--epochs", default=100, type=int, help="the number of total epochs")
    parser.add_argument("--trainBS", default=4, type=int, help="the batchSize of training")
    parser.add_argument("--inferBS", default=128, type=int, help="the batchSize of infering")
    parser.add_argument("--lr", default=2.5e-4, type=float, help="initial learning rate")
    parser.add_argument("--wd", default=0, type=float, help="weight decay (default: 0)")
    parser.add_argument("--power", default=0.9, type=float, help="power for learning rate decay")


    # Data augment
    parser.add_argument("--useFlip", default="True", help="whether add flip augment")
    parser.add_argument("--scaleRange", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rotRange", default=30.0, type=float, help="rotation factor")
    parser.add_argument("--useOcclusion", default="False", help="whether add occlusion augment")
    parser.add_argument("--numOccluder", default=8, type=int, help="number of occluder to add in")

    # Data augment (to teacher in Mean-Teacher)
    parser.add_argument("--scaleRange_ema", default=0.25, type=float, help="scale factor")
    parser.add_argument("--rotRange_ema", default=30.0, type=float, help="rotation factor")
    parser.add_argument("--useOcclusion_ema", default="False", help="whether add occlusion augment")
    parser.add_argument("--numOccluder_ema", default=8, type=int, help="number of occluder to add in")

    # Hyper-parameter
    parser.add_argument("--poseWeight", default=10.0, type=float, help="the weight of pose loss (default: 10.0)")

    # misc
    parser.add_argument("--debug", default="False")
    parser.add_argument("--program", default="SSL-Pose3_v1.0.20230805.1")
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()
