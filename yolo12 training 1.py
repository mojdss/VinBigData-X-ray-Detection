# File: train.py
# Relative Path: .
# Author: AUTHOR_NAME

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from threading import Thread
from warnings import warn

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Adapted YOLO12 imports (please verify exact module names in your YOLO12 installation)
import yolo12.testing as test                          # test.py equivalent in YOLO12
from yolo12.model import Model                          # core Model definition in YOLO12
from yolo12.utils.anchor import check_anchors           # anchor utilities in YOLO12
from yolo12.utils.datasets import create_dataloader     # dataset & dataloader creation
from yolo12.utils.general import (
    labels_to_class_weights,
    increment_path,
    labels_to_image_weights,
    init_seeds,
    fitness,
    strip_optimizer,
    get_latest_run,
    check_dataset,
    check_file,
    check_git_status,
    check_img_size,
    set_logging,
)
from yolo12.utils.loss import compute_loss             # loss computation in YOLO12
from yolo12.utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from yolo12.utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first

logger = logging.getLogger(__name__)

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f'Hyperparameters {hyp}')
    save_dir, epochs, batch_size, total_batch_size, weights, rank = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.total_batch_size,
        opt.weights,
        opt.global_rank,
    )

    # Create directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Plot flag
    plots = not opt.evolve
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)

    # Load data config (dataset paths, number of classes, names)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)

    train_path = data_dict['train']
    val_path = data_dict['val']
    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = (
        ['item']
        if opt.single_cls and len(data_dict['names']) != 1
        else data_dict['names']
    )
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {opt.data}'

    # Instantiate model (either from scratch or from a YOLO12 checkpoint)
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            # attempt to download if needed (YOLO12’s equivalent)
            # Here, `attempt_download` could be replaced by a YOLO12-specific loader if required
            from yolo12.utils.google_utils import attempt_download as y12_attempt_download
            y12_attempt_download(weights)
        ckpt = torch.load(weights, map_location=device)
        if hyp.get('anchors'):
            ckpt['model'].yaml['anchors'] = round(hyp['anchors'])
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc).to(device)
        exclude = ['anchor'] if opt.cfg or hyp.get('anchors') else []
        state_dict = ckpt['model'].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
        logger.info(
            f'Transferred {len(state_dict)}/{len(model.state_dict())} items from {weights}'
        )
    else:
        model = Model(opt.cfg, ch=3, nc=nc).to(device)

    # Freeze parameters if requested (empty list here means train all)
    freeze = []  # list of parameter-name substrings to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            v.requires_grad = False
            logger.info(f'Freezing parameter {k}')

    # Configure optimizer
    nbs = 64  # nominal batch size for normalization
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs

    pg0, pg1, pg2 = [], [], []
    for m in model.modules():
        if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
            pg2.append(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            pg0.append(m.weight)
        elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
            pg1.append(m.weight)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    logger.info(
        f'Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other'
    )
    del pg0, pg1, pg2

    # Cosine LR scheduler (OneCycle style)
    lf = (
        lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf'])
        + hyp['lrf']
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Initialize Weights & Biases if requested
    if wandb and wandb.run is None:
        opt.hyp = hyp
        wandb_run = wandb.init(
            config=opt,
            resume="allow",
            project='YOLO12' if opt.project == 'runs/train' else Path(opt.project).stem,
            name=save_dir.stem,
            id=ckpt.get('wandb_id') if pretrained else None,
        )
    else:
        wandb_run = None

    # Resume logic if loading a checkpoint
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as f:
                f.write(ckpt['training_results'])
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, f'{weights} has finished training, nothing to resume.'
        if epochs < start_epoch:
            logger.info(
                f'{weights} already trained for {ckpt["epoch"]} epochs. Fine‐tuning for {epochs} more.'
            )
            epochs += ckpt['epoch']
        del ckpt, state_dict

    # Verify and adjust image sizes
    gs = int(max(model.stride))  # max stride from YOLO12’s model
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # DataParallel if multiple GPUs (single‐process)
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm if in DDP mode
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Exponential Moving Average
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # DistributedDataParallel
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # Create dataloaders
    dataloader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size,
        gs,
        opt,
        hyp=hyp,
        augment=True,
        cache=opt.cache_images,
        rect=opt.rect,
        rank=rank,
        world_size=opt.world_size,
        workers=opt.workers,
        image_weights=opt.image_weights,
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    nb = len(dataloader)
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {opt.data}'

    # Prepare validation dataloader on rank 0 or single‐GPU
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate
        testloader = create_dataloader(
            val_path,
            imgsz_test,
            total_batch_size,
            gs,
            opt,
            hyp=hyp,
            cache=opt.cache_images and not opt.notest,
            rect=True,
            rank=-1,
            world_size=opt.world_size,
            workers=opt.workers,
            pad=0.5,
        )[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])
            if plots:
                plot_labels(labels, save_dir, {'wandb': wandb})
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Attach remaining model attributes
    hyp['cls'] *= nc / 80.0
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # Start training loop
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)

    logger.info(
        f'Image sizes {imgsz} train, {imgsz_test} test\n'
        f'Using {dataloader.num_workers} dataloader workers\n'
        f'Logging results to {save_dir}\n'
        f'Starting training for {epochs} epochs...'
    )

    for epoch in range(start_epoch, epochs):
        model.train()

        # Optionally update image weights for class imbalance
        if opt.image_weights:
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
            if rank != -1:
                indices = (
                    torch.tensor(dataset.indices)
                    if rank == 0
                    else torch.zeros(dataset.n)
                ).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)

        if rank != -1:
            dataloader.sampler.set_epoch(epoch)

        pbar = enumerate(dataloader)
        logger.info(
            ('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size')
        )
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)

        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warmup scheduling
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(
                        ni,
                        xi,
                        [
                            hyp['warmup_bias_lr'] if j == 2 else 0.0,
                            x['initial_lr'] * lf(epoch),
                        ],
                    )
                    if 'momentum' in x:
                        x['momentum'] = np.interp(
                            ni, xi, [hyp['warmup_momentum'], hyp['momentum']]
                        )

            # Multi‐scale training
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward + Compute loss
            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets.to(device), model)
                if rank != -1:
                    loss *= opt.world_size

            # Backward
            scaler.scale(loss).backward()

            # Step optimizer
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Logging
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = f"{torch.cuda.memory_reserved()/1e9:.3g}G" if torch.cuda.is_available() else "0G"
                s = (
                    '%10s' * 2
                    + '%10.4g' * 6
                ) % (
                    f'{epoch}/{epochs - 1}',
                    mem,
                    *mloss,
                    targets.shape[0],
                    imgs.shape[-1],
                )
                pbar.set_description(s)

                # Plot first few batches
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'
                    Thread(
                        target=plot_images, args=(imgs, targets, paths, f), daemon=True
                    ).start()
                elif plots and ni == 3 and wandb:
                    wandb.log(
                        {
                            "Mosaics": [
                                wandb.Image(str(x), caption=x.name)
                                for x in save_dir.glob('train*.jpg')
                            ]
                        }
                    )

        # End of batch loop

        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # DDP or single‐GPU validation and checkpointing
        if rank in [-1, 0]:
            if ema:
                ema.update_attr(
                    model,
                    include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'],
                )
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:
                results, maps, times = test.test(
                    opt.data,
                    batch_size=total_batch_size,
                    imgsz=imgsz_test,
                    model=ema.ema,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    save_dir=save_dir,
                    plots=plots and final_epoch,
                    log_imgs=opt.log_imgs if wandb else 0,
                )

            # Append results to results_file
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')

            # Optionally upload to remote bucket
            if len(opt.name) and opt.bucket:
                os.system(f'gsutil cp {results_file} gs://{opt.bucket}/results/results{opt.name}.txt')

            # Log to TensorBoard / W&B
            tags = [
                'train/box_loss',
                'train/obj_loss',
                'train/cls_loss',
                'metrics/precision',
                'metrics/recall',
                'metrics/mAP_0.5',
                'metrics/mAP_0.5:0.95',
                'val/box_loss',
                'val/obj_loss',
                'val/cls_loss',
                'x/lr0',
                'x/lr1',
                'x/lr2',
            ]
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)
                if wandb:
                    wandb.log({tag: x})

            # Update best fitness and save weights
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi

            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:
                    ckpt_data = {
                        'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict(),
                        'wandb_id': wandb_run.id if wandb_run else None,
                    }

                torch.save(ckpt_data, last)
                if best_fitness == fi:
                    torch.save(ckpt_data, best)
                del ckpt_data

        # End of per‐epoch validation

    # End of all epochs

    if rank in [-1, 0]:
        final_model = best if best.exists() else last
        for f in [last, best]:
            if f.exists():
                strip_optimizer(f)
        if opt.bucket:
            os.system(f'gsutil cp {final_model} gs://{opt.bucket}/weights')

        if plots:
            plot_results(save_dir=save_dir)
            if wandb:
                files = ['results.png', 'precision_recall_curve.png', 'confusion_matrix.png']
                wandb.log(
                    {
                        "Results": [
                            wandb.Image(str(save_dir / f), caption=f)
                            for f in files
                            if (save_dir / f).exists()
                        ]
                    }
                )
                if opt.log_artifacts:
                    wandb.log_artifact(artifact_or_path=str(final_model), type='model', name=save_dir.stem)

        logger.info(
            f'{epochs - start_epoch} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n'
        )

        # If training on COCO with 80 classes, optionally run speed/mAP tests
        if opt.data.endswith('coco.yaml') and nc == 80:
            for conf, iou, save_json in ([0.25, 0.45, False], [0.001, 0.65, True]):
                results, _, _ = test.test(
                    opt.data,
                    batch_size=total_batch_size,
                    imgsz=imgsz_test,
                    conf_thres=conf,
                    iou_thres=iou,
                    model=Model.load_checkpoint(final_model, device).half(),
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    save_dir=save_dir,
                    save_json=save_json,
                    plots=False,
                )
    else:
        dist.destroy_process_group()

    if wandb_run:
        wandb_run.finish()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        type=str,
        default='yolov12s.pt',
        help='initial weights path or checkpoint for YOLO12',
    )
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path for YOLO12 architecture')
    parser.add_argument(
        '--data',
        type=str,
        default='configs/datasets/coco.yaml',
        help='data.yaml path (dataset configuration) for YOLO12',
    )
    parser.add_argument(
        '--hyp',
        type=str,
        default='configs/hypers/hyp.scratch.yaml',
        help='hyperparameters yaml path',
    )
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[640, 640],
        help='[train, test] image sizes',
    )
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent YOLO12 training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, e.g., 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size ±50%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm (only in DDP mode)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter (do not modify)')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging (max 100)')
    parser.add_argument('--log-artifacts', action='store_true', help='log artifacts (final model)')
    parser.add_argument('--workers', type=int, default=8, help='max number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    # Distributed training variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ.get('WORLD_SIZE', 1))
    opt.global_rank = int(os.environ.get('RANK', -1))
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_git_status()

    # Resume logic: if --resume is provided, load checkpoint arguments
    if opt.resume:
        ckpt_path = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt_path), 'ERROR: --resume checkpoint does not exist'
        apriori = (opt.global_rank, opt.local_rank)
        with open(Path(ckpt_path).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        opt.cfg, opt.weights, opt.resume, opt.global_rank, opt.local_rank = (
            '',
            ckpt_path,
            True,
            *apriori,
        )
        logger.info(f'Resuming training from {ckpt_path}')
    else:
        opt.data, opt.cfg, opt.hyp = (
            check_file(opt.data),
            check_file(opt.cfg),
            check_file(opt.hyp),
        )
        assert len(opt.cfg) or len(opt.weights), 'Either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)

    # Select device (CPU or GPU) and set up DDP if needed
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be divisible by number of GPUs'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Load hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
        if 'box' not in hyp:
            warn(f'Compatibility: {opt.hyp} missing "box" (renamed from "giou")')
            hyp['box'] = hyp.pop('giou')

    # Begin training or hyperparameter evolution
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start TensorBoard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)
        train(hyp, opt, device, tb_writer, wandb)
    else:
        # Hyperparameter evolution loop (identical structure to YOLOv5, adjusted for YOLO12)
        meta = {
            'lr0': (1, 1e-5, 1e-1),
            'lrf': (1, 0.01, 1.0),
            'momentum': (0.3, 0.6, 0.98),
            'weight_decay': (1, 0.0, 0.001),
            'warmup_epochs': (1, 0.0, 5.0),
            'warmup_momentum': (1, 0.0, 0.95),
            'warmup_bias_lr': (1, 0.0, 0.2),
            'box': (1, 0.02, 0.2),
            'cls': (1, 0.2, 4.0),
            'cls_pw': (1, 0.5, 2.0),
            'obj': (1, 0.2, 4.0),
            'obj_pw': (1, 0.5, 2.0),
            'iou_t': (0, 0.1, 0.7),
            'anchor_t': (1, 2.0, 8.0),
            'anchors': (2, 2.0, 10.0),
            'fl_gamma': (0, 0.0, 2.0),
            'hsv_h': (1, 0.0, 0.1),
            'hsv_s': (1, 0.0, 0.9),
            'hsv_v': (1, 0.0, 0.9),
            'degrees': (1, 0.0, 45.0),
            'translate': (1, 0.0, 0.9),
            'scale': (1, 0.0, 0.9),
            'shear': (1, 0.0, 10.0),
            'perspective': (0, 0.0, 0.001),
            'flipud': (1, 0.0, 1.0),
            'fliplr': (0, 0.0, 1.0),
            'mosaic': (1, 0.0, 1.0),
            'mixup': (1, 0.0, 1.0),
        }

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.txt .')

        for _ in range(300):
            if Path('evolve.txt').exists():
                parent = 'single'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))
                x = x[np.argsort(-fitness(x))][:n]
                w = fitness(x) - fitness(x).min()
                if parent == 'single' or len(x) == 1:
                    x = x[random.choices(range(n), weights=w)[0]]
                else:
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()

                mp, s = 0.8, 0.2
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):
                    hyp[k] = float(x[i + 7] * v[i])

            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])
                hyp[k] = min(hyp[k], v[2])
                hyp[k] = round(hyp[k], 5)

            results = train(hyp.copy(), opt, device, wandb=wandb)
            from yolo12.utils.general import print_mutation

            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        plot_evolution(yaml_file)
        print(
            f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
            f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}'
        )
