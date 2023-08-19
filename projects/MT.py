# -*- coding: utf-8 -*-
import os
import random
import argparse
import datetime
import numpy as np
import torch
from torch.optim.adamw import AdamW as TorchAdam
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from utils.mt.data import TwoStreamBatchSampler

import GLOB as glob
import datasources
import datasets
import models as modelClass

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc
from utils.augment import AugmentUtils as aug
from utils.parameters import update_ema_variables, consWeight_increase
from utils.losses import JointMSELoss, JointDistLoss, AvgCounter, AvgCounters
from projects.tools import ProjectTools as proj
from utils.evaluation import EvaluationUtils as eval


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args.start_epoch = 0
    args.best_acc = [-1., -1.]
    args.best_epoch = [0, 0]
    args.brCompareCount = [0, 0]

    # region 1. Initialize
    # Data loading
    dataSource = datasources.__dict__[args.dataSource]()
    semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs, means, stds = dataSource.getSemiData(args.trainCount, args.validCount, args.labelRatio)
    args.kpsCount, args.imgType, args.pck_ref, args.pck_thr, args.inpRes, args.outRes = dataSource.kpsCount, dataSource.imgType, dataSource.pck_ref, dataSource.pck_thr, dataSource.inpRes, dataSource.outRes

    # Model initialize
    model_pNum = 0
    model = modelClass.__dict__["PoseModel"](modelType=args.model, kpsCount=args.kpsCount, mode=args.feature_mode).to(args.device)
    model_ema = modelClass.__dict__["PoseModel"](modelType=args.model, kpsCount=args.kpsCount, mode=args.feature_mode, nograd=True).to(args.device)
    optim = TorchAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    model_pNum += (sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model_ema.parameters()))
    args.nStack = model.nStack
    logc = "=> initialized MT ({}) Structure (params: {})".format(args.model, format(model_pNum / (1024**2), ".2f"))
    logger.print("L1", logc)

    # Dataloader initialize
    # region trainDS initialize by first way
    # trainDS = datasets.__dict__["DS_mds"]("train", semiTrainData, means, stds, augCount=args.brNum*args.br_augNum, gtCount=args.br_gtNum, isAug=True, isDraw=args.debug, **vars(args))
    # trainLoader = TorchDataLoader(trainDS, batch_size=args.trainBS, shuffle=True, pin_memory=True, drop_last=False)
    # endregion

    # region trainDS initialize by second way
    trainDS = datasets.__dict__["DS_mds"]("train", semiTrainData, means, stds, augCount=args.brNum * args.br_augNum, gtCount=args.br_gtNum, isAug=True, isDraw=args.debug, **vars(args))
    batchSampler = TwoStreamBatchSampler(unlabeledIdxs, labeledIdxs, args.trainBS, args.trainBS_labeled)
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

        # region 2.1 update dynamic parameters
        args.consWeight = consWeight_increase(epo, args)
        # endregion

        # region 2.2 model training and validating
        startTM = datetime.datetime.now()
        pec_loss, mtc_loss = train(trainLoader, model, model_ema, optim, args)
        logger.print("L3", "model training finished...", start=startTM)
        startTM = datetime.datetime.now()
        predsArraies, accsArraies, errsArraies = validate(validLoader, model, model_ema, args)
        logger.print("L3", "model validating finished...", start=startTM)
        # endregion

        # region 2.3 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = []
        for idx in range(len(args.best_epoch)):
            flag = accsArraies[idx][-1] > args.best_acc[idx]
            is_best.append(flag)
            if flag:
                args.best_epoch[idx] = epo
                args.best_acc[idx] = accsArraies[idx][-1]
        # model storage
        checkpoint = {"current_epoch": epo, "best_acc": args.best_acc, "best_epoch": args.best_epoch,
                      "model": args.model, "feature_mode": args.feature_mode,
                      "model_state": model.state_dict(), "model_ema_state": model_ema.state_dict(), "optim_state": optim.state_dict()}
        comm.ckpt_save(checkpoint, is_best[-1], ckptPath="{}/ckpts".format(args.basePath))
        logger.print("L3", "model storage finished...", start=startTM)
        # endregion

        # region 2.4 log storage
        startTM = datetime.datetime.now()
        # Initialization parameter storage
        if epo == args.start_epoch:
            save_args = vars(args).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)

        # Log data storage
        log_data = {"pec_loss": pec_loss, "mtc_loss": mtc_loss, "accsArraies": accsArraies, "errsArraies": errsArraies}
        comm.json_save(log_data, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)

        # Pseudo-labels data storage
        pseudo_data = {"predsArraies": predsArraies}
        comm.json_save(pseudo_data, "{}/logs/pseudoData/pseudoData_{}.json".format(args.basePath, epo+1), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 2.5 output result
        marks = ["stu", "ema"]
        # Training performance
        for idx in range(args.brNum):
            fmtc = "[{}/{} | lr: {} | ConsW: {} | {}] pec_loss: {}, mtc_loss: {}"
            logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"),
                               format(optim.state_dict()['param_groups'][0]['lr'], ".8f"),
                               format(args.consWeight, ".2f"), marks[idx],
                               format(pec_loss, ".5f"), format(mtc_loss, ".10f"))
            logger.print("L1", logc)

        # Validating performance
        if accsArraies[0][-1] >= accsArraies[1][-1]:
            args.brCompareCount[0] += 1
        else:
            args.brCompareCount[1] += 1
        for idx in range(len(args.best_epoch)):
            fmtc = "[{}/{} | ConsW: {} | {} ({})] best acc: {} (epo: {}) | acc: {}, err: {} | kps_acc:[{}], kps_err:[{}]"
            logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.consWeight, ".2f"),
                               marks[idx], format(args.brCompareCount[idx], "3d"),
                               format(args.best_acc[idx], ".5f"), format(args.best_epoch[idx] + 1, "3d"),
                               format(accsArraies[idx][-1], ".5f"), format(errsArraies[idx][-1], ".3f"),
                               proj.setContent(accsArraies[idx][0:len(accsArraies[idx])-1], ".5f"),
                               proj.setContent(errsArraies[idx][0:len(errsArraies[idx])-1], ".3f"))
            logger.print("L1", logc)

        # Epoch line
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
        fmtc = "[{}/{} | {}] ---------- ---------- ---------- ---------- ---------- ---------- ----------"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), time_interval)
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion

    logger.print("L1", "[{}, all executing finished...]".format(args.experiment), start=allTM)


def train(trainLoader, model, model_ema, optim, args):
    # region 1. Preparation
    pec_counter = AvgCounter()
    mtc_counter = AvgCounter()
    pose_criterion = JointMSELoss(nStack=args.nStack, useKPsGate=True, useSampleWeight=True).to(args.device)
    consistency_criterion = JointDistLoss().to(args.device)
    model.train()
    model_ema.train()
    # endregion

    # region 2. Training
    for bat, (augs_imgMap, augs_heatmaps, meta) in enumerate(trainLoader):
        optim.zero_grad()

        # region 2.1 Data organizing
        augs_heatmaps = [[proj.setVariable(aug_heatmap, args.device) for aug_heatmap in aug_heatmaps] for aug_heatmaps in augs_heatmaps]  # [2, 2] * [bs, 9, 256, 256]
        augs_kpsGate = [[proj.setVariable(aug_kpsWeight, args.device) for aug_kpsWeight in aug_kpsWeights] for aug_kpsWeights in meta['kpsWeights']]   # [2, 2] * [bs, 9]
        augs_imgMap = [proj.setVariable(aug_imgMap, args.device) for aug_imgMap in augs_imgMap]  # 2 * [bs, 3, 256, 256]
        augs_warpmat = [proj.setVariable(aug_warpmat, args.device) for aug_warpmat in meta['warpmat']]  # 2 * [bs, 2, 3]
        augs_isFlip = [aug_isFlip.to(args.device, non_blocking=True) for aug_isFlip in meta['isflip']]  # 2 * [bs]
        augs_samplesWeight = proj.getSampleWeight(meta['islabeled'], args)  # 2 * [bs]
        # endregion

        # region 2.2 Test
        # region 2.2.1 test_affine_back
        if args.debug:
            for augIdx in range(args.br_augNum):
                for gtIdx in range(args.br_gtNum):
                    kpsHeatmap_test = aug.affine_back2(augs_heatmaps[augIdx][gtIdx], augs_warpmat[augIdx], augs_isFlip[augIdx])
                    preds_test, _ = proc.kps_fromHeatmap(kpsHeatmap_test.detach().cpu(), meta["center"][augIdx], torch.tensor([1 for idx in range(len(meta["center"][augIdx]))]), [args.outRes, args.outRes])  # 使用aug_scale
                    bs, k, _, _ = augs_heatmaps[augIdx][gtIdx].shape
                    for bsIdx in range(bs):
                        lsLabeled = meta['islabeled'][gtIdx][bsIdx]
                        if lsLabeled:
                            imgID = meta["imageID"][bsIdx]
                            img = proc.image_load(meta["imagePath"][bsIdx])
                            h, w, _ = img.shape
                            gtArray = preds_test[bsIdx].cpu().data.numpy().tolist()
                            for pred in gtArray:
                                gt = [pred[0] * w / args.inpRes, pred[1] * h / args.inpRes]
                                img = proc.draw_point(img, gt, radius=3, thickness=-1, color=(0, 95, 191))
                            proc.image_save(img, "{}/draw/test/test_affine_back/epo_{}/{}_aug{}_gt{}.{}".format(args.basePath, args.epo+1, imgID, augIdx+1, gtIdx+1, args.imgType))
        # endregion

        # region 2.2.2 test_dataloader_output
        if args.debug:
            for augIdx in range(args.br_augNum):
                for gtIdx in range(args.br_gtNum):
                    preds_test, _ = proc.kps_fromHeatmap(augs_heatmaps[augIdx][gtIdx].detach().cpu(), meta["center"][augIdx], meta['ori_scale'], [args.outRes, args.outRes])  # 使用ori_scale
                    bs, k, _, _ = augs_heatmaps[augIdx][gtIdx].shape
                    for bsIdx in range(bs):
                        lsLabeled = meta['islabeled'][gtIdx][bsIdx]
                        if lsLabeled:
                            imgID = meta["imageID"][bsIdx]
                            img = proc.image_tensor2np(augs_imgMap[augIdx][bsIdx].detach().cpu().data * 255).astype(np.uint8)
                            h, w, _ = img.shape
                            gtArray = preds_test[bsIdx].cpu().data.numpy().tolist()
                            for pred in gtArray:
                                gt = [pred[0] * w / args.inpRes, pred[1] * h / args.inpRes]
                                img = proc.draw_point(img, gt, radius=3, thickness=-1, color=(0, 95, 191))
                            proc.image_save(img, "{}/draw/test/test_dataloader_output/epo_{}/{}_aug{}_gt{}.{}".format(args.basePath, args.epo+1, imgID, augIdx+1, gtIdx+1, args.imgType))
        # endregion
        # endregion

        # region 2.3 model forward
        outs, outs_ema = [], []
        for aIdx, aug_imgMap in enumerate(augs_imgMap):
            out = model(aug_imgMap) if args.feature_mode == "default" else model(aug_imgMap)[0]
            outs.append(out)
            with torch.no_grad():
                out_ema = model_ema(aug_imgMap) if args.feature_mode == "default" else model_ema(aug_imgMap)[0]
                outs_ema.append(out_ema)
        outs = torch.stack(outs, dim=0)
        outs_ema = torch.stack(outs_ema, dim=0)
        # endregion

        # region 2.4 mean-teacher consistency constraint
        mtc_sum, mtc_count = 0., 0
        for aIdx in range(len(outs)):  # [augNum, bs, k]
            loss, n = consistency_criterion(outs[aIdx, :, -1], outs_ema[aIdx, :, -1])
            mtc_sum += loss
            mtc_count += n
        mtc_loss = args.consWeight * ((mtc_sum / mtc_count) if mtc_count > 0 else mtc_sum)
        mtc_counter.update(mtc_loss.item(), mtc_count)
        # endregion

        # region 2.5 pose estimation constraint
        pec_sum, pec_count = 0., 0
        for aIdx in range(len(outs)):  # [augNum, bs, k]
            loss, n = pose_criterion(outs[aIdx], augs_heatmaps[aIdx][0], augs_kpsGate[aIdx][0], augs_samplesWeight[0])
            pec_sum += loss
            pec_count += n
        pec_loss = args.poseWeight * ((pec_sum / pec_count) if pec_count > 0 else pec_sum)
        pec_counter.update(pec_loss.item(), pec_count)
        # endregion

        # region 2.6 calculate total loss & update model
        total_loss = pec_loss + mtc_loss
        total_loss.backward()
        optim.step()
        update_ema_variables(model, model_ema, args)
        # endregion

        # region 2.10 clearing the GPU Cache
        del outs, outs_ema, pec_loss, mtc_loss
        # endregion
    # endregion
    return pec_counter.avg, mtc_counter.avg


def validate(validLoader, model, model_ema, args):
    # region 1. Preparation
    accs_counters = [AvgCounters() for mIdx in range(len(args.best_epoch))]
    errs_counters = [AvgCounters() for mIdx in range(len(args.best_epoch))]
    predsArray = [[] for mIdx in range(len(args.best_epoch))]
    model.eval()
    model_ema.eval()
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
            outs = model(imgMap) if args.feature_mode == "default" else model(imgMap)[0]  # [b, n, k, 64, 64]
            outs_ema = model_ema(imgMap) if args.feature_mode == "default" else model_ema(imgMap)[0]  # [b, n, k, 64, 64]

            # predict
            preds, _ = proc.kps_fromHeatmap(outs[:, -1].cpu().detach(), meta["center"], meta["scale"], [args.outRes, args.outRes])
            predsArray[0] += preds.clone().data.numpy().tolist()
            preds_ema, _ = proc.kps_fromHeatmap(outs_ema[:, -1].cpu().detach(), meta["center"], meta["scale"], [args.outRes, args.outRes])
            predsArray[1] += preds_ema.clone().data.numpy().tolist()
            # endregion

            # region 2.3 calculate the error and accuracy
            preds_model = [preds, preds_ema]
            for mIdx in range(len(args.best_epoch)):
                errs, accs = eval.acc_pck(preds_model[mIdx], meta['kpsMap'], args.pck_ref, args.pck_thr)
                for idx in range(k+1):
                    accs_counters[mIdx].update(idx, accs[idx].item(), bs if idx < k else bs*k)
                    errs_counters[mIdx].update(idx, errs[idx].item(), bs if idx < k else bs*k)
            # endregion

            # region 2.4 clearing the GPU Cache
            del outs_ema
            # endregion

    # region 3. records neaten
    accs_records = [accs_counter.avg() for accs_counter in accs_counters]
    errs_records = [errs_counter.avg() for errs_counter in errs_counters]
    # endregion
    return predsArray, accs_records, errs_records


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
    parser.add_argument("--brNum", default=2, type=int, help="the number of ensemble branch in Structure")
    parser.add_argument("--br_augNum", default=1, type=int, help="the number of augment samples in each branch")
    parser.add_argument("--br_gtNum", default=1, type=int, help="the number of ground-truth in each branch")

    # Dataset setting
    parser.add_argument("--dataSource", default="Mouse", choices=["Mouse", "FLIC", "LSP"])
    parser.add_argument("--trainCount", default=100, type=int)
    parser.add_argument("--validCount", default=500, type=int)
    parser.add_argument("--labelRatio", default=0.3, type=float)

    # Training strategy
    parser.add_argument("--epochs", default=100, type=int, help="the number of total epochs")
    parser.add_argument("--trainBS", default=4, type=int, help="the batchSize of training")
    parser.add_argument("--trainBS_labeled", default=2, type=int, help="the batchSize of training")
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

    parser.add_argument("--consWeight_max", default=10.0, type=float)
    parser.add_argument("--consWeight_min", default=0.0, type=float)
    parser.add_argument("--consWeight_rampup", default=5, type=int)

    # mean-teacher
    parser.add_argument("--ema_decay", default=0.999, type=float, help="ema variable decay rate (default: 0.999)")

    # misc
    parser.add_argument("--debug", default="False")
    parser.add_argument("--program", default="SSL-Pose3_v1.0.20230805.1")
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()
