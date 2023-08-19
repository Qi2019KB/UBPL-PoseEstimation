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
from utils.parameters import update_ema_variables, consWeight_increase, pseudoWeight_increase, FDLWeight_decrease
from utils.losses import JointMSELoss, JointDistLoss, JointFeatureDistLoss, JointPseudoLoss3, AvgCounter, AvgCounters
from projects.tools import ProjectTools as proj
from utils.evaluation import EvaluationUtils as eval


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args.start_epoch = 0
    args.best_acc = [-1.] if args.brNum == 1 else [-1. for idx in range(args.brNum + 1)]
    args.best_epoch = [0] if args.brNum == 1 else [0 for idx in range(args.brNum + 1)]

    # region 1. Initialize
    # Data loading
    dataSource = datasources.__dict__[args.dataSource]()
    semiTrainData, validData, labeledData, unlabeledData, labeledIdxs, unlabeledIdxs, means, stds = dataSource.getSemiData(args.trainCount, args.validCount, args.labelRatio)
    args.kpsCount, args.imgType, args.pck_ref, args.pck_thr, args.inpRes, args.outRes = dataSource.kpsCount, dataSource.imgType, dataSource.pck_ref, dataSource.pck_thr, dataSource.inpRes, dataSource.outRes

    # Model initialize
    models, models_ema, optims, model_pNum = [], [], [], 0
    for bIdx in range(args.brNum):
        model = modelClass.__dict__["PoseModel"](modelType=args.model, kpsCount=args.kpsCount, mode=args.feature_mode).to(args.device)
        models.append(model)
        model_ema = modelClass.__dict__["PoseModel"](modelType=args.model, kpsCount=args.kpsCount, mode=args.feature_mode, nograd=True).to(args.device)
        models_ema.append(model_ema)
        optim = TorchAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        optims.append(optim)
        model_pNum += (sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model_ema.parameters()))
    args.nStack = models[0].nStack
    logc = "=> initialized MDSs ({}) Structure (params: {})".format(args.model, format(model_pNum / (1024**2), ".2f"))
    logger.print("L1", logc)

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
        args.FDLWeight = FDLWeight_decrease(epo, args)
        args.pseudoWeight = pseudoWeight_increase(epo, args)
        # endregion

        # region 2.2 model training and validating
        startTM = datetime.datetime.now()
        pec_losses, mtc_losses, epc_losses, fdc_loss = train(trainLoader, models, models_ema, optims, args)
        logger.print("L3", "model training finished...", start=startTM)
        startTM = datetime.datetime.now()
        predsArraies, accsArraies, errsArraies = validate(validLoader, models_ema, args)
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
        checkpoint = {"current_epoch": epo, "best_acc": args.best_acc, "best_epoch": args.best_epoch}
        for bIdx in range(args.brNum):
            checkpoint["model{}_state".format(bIdx+1)] = models[bIdx].state_dict()
            checkpoint["model{}_ema_state".format(bIdx+1)] = models_ema[bIdx].state_dict()
            checkpoint["optim{}_state".format(bIdx+1)] = optims[bIdx].state_dict()
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
        log_data = {"pec_losses": pec_losses, "mtc_losses": mtc_losses, "fdc_loss": fdc_loss, "epc_losses": epc_losses, "accsArraies": accsArraies, "errsArraies": errsArraies}
        comm.json_save(log_data, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)

        # Pseudo-labels data storage
        pseudo_data = {"predsArraies": predsArraies}
        comm.json_save(pseudo_data, "{}/logs/pseudoData/pseudoData_{}.json".format(args.basePath, epo+1), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 2.5 output result
        marks = ["  mds1", "  mds2", "  mean", "h_mean"]
        # Training performance
        for idx in range(args.brNum):
            fmtc = "[{}/{} | lr: {} | ConsW: {}, FDLW: {} | {}] pec_loss: {}, mtc_loss: {}, epc_loss: {}, fdc_loss: {}"
            logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(optims[idx].state_dict()['param_groups'][0]['lr'], ".8f"), format(args.consWeight, ".2f"),
                               format(args.FDLWeight, ".4f"), marks[idx],
                               format(pec_losses[idx], ".5f"), format(mtc_losses[idx], ".10f"),
                               format(epc_losses[idx], ".10f"), format(fdc_loss, ".10f"))
            logger.print("L1", logc)

        # Validating performance
        for idx in range(len(args.best_epoch)):
            fmtc = "[{}/{} | ConsW: {} | {}] best acc: {} (epo: {}) | acc: {}, err: {} | kps_acc:[{}], kps_err:[{}]"
            logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), format(args.consWeight, ".2f"), marks[idx],
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


def train(trainLoader, models, models_ema, optims, args):
    # region 1. Preparation
    pec_counters = [AvgCounter() for mIdx in range(len(models))]
    mtc_counters = [AvgCounter() for mIdx in range(len(models))]
    epc_counters = [AvgCounter() for mIdx in range(len(models))]
    fdc_counter = AvgCounter()
    pose_criterion = JointMSELoss(nStack=args.nStack, useKPsGate=True, useSampleWeight=True).to(args.device)
    consistency_criterion = JointDistLoss().to(args.device)
    feature_distance_criterion = JointFeatureDistLoss().to(args.device)
    pseudo_criterion2 = JointPseudoLoss3(nStack=args.nStack, scoreThr=args.pseudoScoreThr).to(args.device)

    for model in models: model.train()
    for model_ema in models_ema: model_ema.train()
    # endregion

    # region 2. Training
    for bat, (augs_imgMap, augs_heatmaps, meta) in enumerate(trainLoader):
        for optim in optims: optim.zero_grad()

        # region 2.1 Data organizing
        augs_heatmaps = [[proj.setVariable(aug_heatmap, args.device) for aug_heatmap in aug_heatmaps] for aug_heatmaps in augs_heatmaps]  # [2, 2] * [bs, 9, 256, 256]
        augs_kpsGate = [[proj.setVariable(aug_kpsWeight, args.device) for aug_kpsWeight in aug_kpsWeights] for aug_kpsWeights in meta['kpsWeights']]   # [2, 2] * [bs, 9]
        augs_imgMap = [proj.setVariable(aug_imgMap, args.device) for aug_imgMap in augs_imgMap]  # 2 * [bs, 3, 256, 256]
        augs_warpmat = [proj.setVariable(aug_warpmat, args.device) for aug_warpmat in meta['warpmat']]  # 2 * [bs, 2, 3]
        augs_isFlip = [aug_isFlip.to(args.device, non_blocking=True) for aug_isFlip in meta['isflip']]  # 2 * [bs]
        augs_samplesWeight = proj.getSampleWeight(meta['islabeled'], args)  # 2 * [bs]
        nega_samplesWeight = proj.getSampleWeight_nega(meta['islabeled'], args)
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
        outs, features, outs_ema = [], [], []
        for mIdx in range(len(models)):
            outs_aug, features_aug, outs_aug_ema = [], [], []
            for iIdx, imgMap in enumerate(augs_imgMap):
                out, feature = models[mIdx](imgMap)
                outs_aug.append(out)
                features_aug.append(feature)
                with torch.no_grad():
                    out_ema, _ = models_ema[mIdx](imgMap)
                    outs_aug_ema.append(out_ema)
            outs.append(torch.stack(outs_aug, dim=0))
            features.append(torch.stack(features_aug, dim=0))
            outs_ema.append(torch.stack(outs_aug_ema, dim=0))
        outs = torch.stack(outs, dim=0)
        features = torch.stack(features, dim=0)
        outs_ema = torch.stack(outs_ema, dim=0)
        # endregion

        # region 2.4 mean-teacher consistency constraint
        mtc_losses = [0. for mIdx in range(len(outs))]
        for mIdx in range(len(outs)):
            mtc_sum, mtc_count = 0., 0
            for aIdx in range(len(outs[mIdx])):
                loss, n = consistency_criterion(outs[mIdx, aIdx, :, -1], outs_ema[mIdx, aIdx, :, -1])
                mtc_sum += loss
                mtc_count += n
            mtc_losses[mIdx] = args.consWeight * ((mtc_sum / mtc_count) if mtc_count > 0 else mtc_sum)
            mtc_counters[mIdx].update(mtc_losses[mIdx].item(), mtc_count)
        # endregion

        # region 2.5 pose estimation constraint
        pec_losses = [0. for mIdx in range(len(outs))]
        for mIdx in range(len(outs)):
            pec_sum, pec_count = 0., 0
            for aIdx in range(len(outs[mIdx])):
                loss, n = pose_criterion(outs[mIdx, aIdx], augs_heatmaps[aIdx][0], augs_kpsGate[aIdx][0], augs_samplesWeight[0])
                pec_sum += loss
                pec_count += n
            pec_losses[mIdx] = args.poseWeight * ((pec_sum / pec_count) if pec_count > 0 else pec_sum)
            pec_counters[mIdx].update(pec_losses[mIdx].item(), pec_count)
        # endregion

        # region 2.6 ensenble prediction constraint
        if args.useEnsemblePseudo:
            # mean_outs_ema = torch.mean(outs_ema, dim=0)
            pec_n_pseudo, pec_n_pseudo_sel, pseudo_score = 0, 0, []
            # rateThrs1, rateThrs2 = [], []
            epc_losses = [0. for mIdx in range(len(outs))]
            for mIdx in range(len(outs)):
                epc_sum, epc_count = 0., 0
                for aIdx in range(len(outs[mIdx])):  # [modelNum, augNum, bs, k]
                    loss, n, n_sel, pseudo_scores, rateThr1, rateThr2 = pseudo_criterion2(outs[mIdx, aIdx], outs_ema.clone()[:, aIdx].detach(), nega_samplesWeight[0])
                    epc_sum += loss
                    epc_count += n
                    pec_n_pseudo += n
                    pec_n_pseudo_sel += n_sel
                    pseudo_score.append(pseudo_scores)
                    # rateThrs1.append(rateThr1.mean(0))
                    # rateThrs2.append(rateThr2.mean(0))
                epc_losses[mIdx] = args.ensemblePseudoWeight * ((epc_sum / epc_count) if epc_count > 0 else epc_sum)
                epc_counters[mIdx].update(epc_losses[mIdx].item(), epc_count)
            pseudo_score = torch.stack(pseudo_score, dim=0).mean(0)
            # rateThr1 = torch.stack(rateThrs1, dim=0).mean(0)
            # rateThr2 = torch.stack(rateThrs2, dim=0).mean(0)
            print("batch.{} (scoreThr:{}): {} ({}/{}), pseudo-score: [{}]".format(format(bat+1, "5d"), format(args.pseudoScoreThr, ".2f"),
                                                                                                format(pec_n_pseudo_sel/pec_n_pseudo, ".2f"), format(pec_n_pseudo_sel, "5d"),
                                                                                                format(pec_n_pseudo, "5d"), proj.setContent(pseudo_score, ".3f")))
        else:
            epc_losses = [0., 0.]
            for epc_counter in epc_counters: epc_counter.update(0., outs.shape[2])
        # endregion

        # region 2.7. multi-view features decorrelation loss
        if args.FDLWeight <= 0:
            fdc_loss = 0.
            fdc_counter.update(0., outs.shape[2])
        else:
            fdc_sum, fdc_count = 0., 0
            for aIdx in range(features.shape[1]):
                # only using labeled data
                v1, v2 = [], []
                for idx, sampleWeight in enumerate(augs_samplesWeight[0]):
                    if args.FDL_label == "labeled":
                        if sampleWeight > 0:
                            v1.append(features[0, aIdx, idx])
                            v2.append(features[1, aIdx, idx])
                    elif args.FDL_label == "unlabeled":
                        if sampleWeight == 0:
                            v1.append(features[0, aIdx, idx])
                            v2.append(features[1, aIdx, idx])
                    elif args.FDL_label == "all":
                        v1.append(features[0, aIdx, idx])
                        v2.append(features[1, aIdx, idx])
                v1, v2 = torch.stack(v1, dim=0), torch.stack(v2, dim=0)
                # endregion
                if args.FDL_type == "covariance":
                    c_cov, c_num = proc.features_cov(v1, v2)
                else:
                    c_cov, c_num = feature_distance_criterion(v1, v2)
                fdc_sum += c_cov
                fdc_count += c_num
            fdc_loss = args.FDLWeight * ((fdc_sum / fdc_count) if fdc_count > 0 else fdc_sum)
            fdc_counter.update(fdc_loss.item(), fdc_count)
        # endregion

        # region 2.8 calculate total loss & update model
        for mIdx in range(len(models)):
            total_loss = pec_losses[mIdx] + mtc_losses[mIdx] + epc_losses[mIdx] + fdc_loss
            total_loss.backward(retain_graph=True)
        for optim in optims: optim.step()
        for mIdx, model in enumerate(models): update_ema_variables(model, models_ema[mIdx], args)
        # endregion

        # region 2.9 clearing the GPU Cache
        del outs, outs_ema, pec_losses, mtc_losses, fdc_loss
        # endregion
    # endregion

    # region 3. records neaten
    pec_records = [pec_counter.avg for pec_counter in pec_counters]
    mtc_records = [mtc_counter.avg for mtc_counter in mtc_counters]
    epc_records = [epc_counter.avg for epc_counter in epc_counters]
    fdc_record = fdc_counter.avg
    # endregion
    return pec_records, mtc_records, epc_records, fdc_record


def validate(validLoader, models_ema, args):
    # region 1. Preparation
    accs_counters = [AvgCounters() for mIdx in range(len(args.best_epoch))]
    errs_counters = [AvgCounters() for mIdx in range(len(args.best_epoch))]
    predsArray = [[] for mIdx in range(len(args.best_epoch))]
    for model_ema in models_ema: model_ema.eval()
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
            outs_ema = []
            for model_ema in models_ema:
                out_ema, _ = model_ema(imgMap)
                outs_ema.append(out_ema)  # [b, n, k, 64, 64]
            outs_ema = torch.stack(outs_ema, dim=0)  # [#model, b, n, k, 64, 64]

            # predict
            preds_model = []
            pred_heatmaps = []
            for mIdx in range(len(models_ema)):
                pred_heatmaps.append(outs_ema[mIdx, :, -1].cpu().detach())
                preds, _ = proc.kps_fromHeatmap(outs_ema[mIdx, :, -1].cpu().detach(), meta["center"], meta["scale"], [args.outRes, args.outRes])
                preds_model.append(preds)
                predsArray[mIdx] += preds.clone().data.numpy().tolist()
            preds_mean = torch.mean(torch.stack(preds_model, dim=-1), dim=-1)
            preds_model.append(preds_mean)
            predsArray[-1] += preds_mean.clone().data.numpy().tolist()
            # endregion

            # region 2.3 calculate the error and accuracy
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
    parser.add_argument("--model", default="HG3", choices=["HG3", "HG2"])
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

    parser.add_argument("--FDL_type", default="covariance", choices=["covariance", "distance"])
    parser.add_argument("--FDL_label", default="labeled", choices=["all", "labeled", "unlabeled"])
    parser.add_argument("--FDLWeight_max", default=1.0, type=float)
    parser.add_argument("--FDLWeight_min", default=1.0, type=float)
    parser.add_argument("--FDLWeight_rampup", default=100, type=int)

    # Pseudo-Label
    parser.add_argument("--useEnsemblePseudo", default="True")
    parser.add_argument("--ensemblePseudoWeight", default=10.0, type=float)
    parser.add_argument("--pseudoWeight_max", default=1.0, type=float)
    parser.add_argument("--pseudoWeight_min", default=1.0, type=float)
    parser.add_argument("--pseudoWeight_rampup", default=100, type=int)
    parser.add_argument("--pseudoScoreThr", default=0.95, type=float)

    # mean-teacher
    parser.add_argument("--ema_decay", default=0.999, type=float, help="ema variable decay rate (default: 0.999)")

    # misc
    parser.add_argument("--debug", default="False")
    parser.add_argument("--program", default="SSL-Pose3_v1.0.20230810.1")
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()
