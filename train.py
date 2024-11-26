import warnings
warnings.filterwarnings("ignore")
import io
import os
import sys
import argparse
import random
import datetime
import time
from importlib import import_module
import torch
import torch.distributed as dist
from torch.distributed import get_world_size, get_rank
torch.backends.cudnn.benchmark = True
torch.manual_seed(123456)
random.seed(123456)

parser = argparse.ArgumentParser(description='Multi-Modal Training')
parser.add_argument('config', type=str, help='path to config file')
parser.add_argument('--local_rank', type=int)
parser.add_argument('--nproc_per_node', type=int)
args = parser.parse_args()

from utils.logging import MultiModalLogging
from utils.oss_op import save_model_to_oss, OssProxy
from utils.logging import AverageMeter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train_epoch(ddp_model, optimizer, train_loader, epoch, dist_info, logger, amp_scaler, config, node_group, use_amp, iter_scheduler):

    data_time_metric = AverageMeter('Data Time')
    forward_time_metric = AverageMeter('Forward Time')
    backward_time_metric = AverageMeter('Backward Time')

    torch.cuda.synchronize()
    t1 = time.time()

    if hasattr(config, 'if_clip_grad') and config.if_clip_grad:
        if_clip_grad = True
    else:
        if_clip_grad = False
    
    if hasattr(config, 'use_bf16') and config.use_bf16:
        use_bf16 = True
    else:
        use_bf16 = False

    for batch_idx, batch_data in enumerate(train_loader):

        torch.cuda.synchronize()
        data_time = time.time() - t1
        t1 = time.time()

        log_info = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'all_batch_cnt': len(train_loader)
        }
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if use_bf16 else torch.float16):
            losses = ddp_model.forward(batch_data, dist_info, batch_idx % 10 == 0, log_info, phase='train', node_group=node_group)

        torch.cuda.synchronize()
        forward_time = time.time() - t1
        t1 = time.time()

        if isinstance(losses, list) or isinstance(losses, tuple):
            raise NotImplementedError
        else:
            amp_scaler.scale(losses).backward()
        
        if if_clip_grad:
            amp_scaler.unscale_(optimizer)
            if hasattr(config, 'grad_norm'):
                total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=config.grad_norm)
            else:
                total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
        else:
            total_norm = -1

        amp_scaler.step(optimizer)
        amp_scaler.update()

        torch.cuda.synchronize()
        backward_time = time.time() - t1

        data_time_metric.update(data_time)
        forward_time_metric.update(forward_time)
        backward_time_metric.update(backward_time)

        if batch_idx % 10 == 0:
            if iter_scheduler is not None:
                logger.info('Data Time: {:.3f}, Forward Time: {:.3f}, Backward Time: {:.3f}, amp: {:.5f}, grad_norm: {:.5f}, lr: {:.10f}'.format(
                    data_time_metric.avg, forward_time_metric.avg, backward_time_metric.avg, amp_scaler.get_scale(), total_norm, iter_scheduler.get_lr()[0]))
            else:
                logger.info('Data Time: {:.3f}, Forward Time: {:.3f}, Backward Time: {:.3f}, amp: {:.5f}, grad_norm: {:.5f}'.format(
                    data_time_metric.avg, forward_time_metric.avg, backward_time_metric.avg, amp_scaler.get_scale(), total_norm))

        if batch_idx % 1000 == 0 and batch_idx != 0:
            global_rank = dist_info['global_rank']
            if global_rank == 0:
                logger.info('saving models to oss')
                save_model_to_oss('{}/epoch{}_{}_params.pth'.format(config.exp_dir, epoch, batch_idx), ddp_model)
                save_model_to_oss('{}/epoch{}_{}_scaler.pth'.format(config.exp_dir, epoch, batch_idx), amp_scaler)
                save_model_to_oss('{}/epoch{}_{}_opt.pth'.format(config.exp_dir, epoch, batch_idx), optimizer)
                if iter_scheduler is not None:
                    save_model_to_oss('{}/epoch{}_{}_scheduler.pth'.format(config.exp_dir, epoch, batch_idx), iter_scheduler)
                else:
                    logger.error('skip saving scheduler, because iter_scheduler is None')

        if iter_scheduler is not None:
            iter_scheduler.step()

        t1 = time.time()

def worker_th_launch(local_rank, dist_world_size, global_rank):
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600), world_size=dist_world_size, rank=global_rank)

    config_dir = os.path.dirname(args.config)
    config_name = os.path.basename(args.config).rsplit('.', 1)[0]
    sys.path.insert(0, config_dir)
    config = import_module(config_name)

    logging = MultiModalLogging()
    logging.add_std()
    logging.add_oss(config.exp_dir)
    logger = logging.get()
    logger.info('exp_dir: {}'.format(config.exp_dir))
    logger.info('GPU info: {}'.format(torch.cuda.get_device_name(0)))

    logger.info('local_rank: {}, global_rank: {}, get_rank(): {}, dist_world_size: {}, get_world_size(): {}'.format(
        local_rank, global_rank, get_rank(), dist_world_size, get_world_size()))

    if hasattr(config, 'use_node_group') and config.use_node_group:
        raise NotImplementedError()
    else:
        node_group = None
 
    if hasattr(config, 'use_amp') and not config.use_amp:
        use_amp = False
    else:
        use_amp = True

    assert global_rank == get_rank()
    assert dist_world_size == get_world_size()
    dist_info = {
        'local_rank': local_rank,
        'global_rank': global_rank,
        'dist_world_size': dist_world_size
    }

    torch.cuda.set_device(local_rank)

    device = torch.device("cuda:{}".format(local_rank))
    # init model here
    model = config.model
    logger.info('model nparams: {}'.format(sum(p.numel() for p in model.parameters())))
    model.to(device)

    # init tokenizer
    text_tokenizer = config.text_tokenizer

    if hasattr(config, 'find_unused_parameters') and config.find_unused_parameters:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    if hasattr(config, 'use_static_graph') and config.use_static_graph:
        ddp_model._set_static_graph()
    ddp_model.train()

    optimizer = config.get_optimizer(ddp_model, logger)
    if hasattr(config, 'use_iter_scheduler') and config.use_iter_scheduler:
        iter_scheduler = config.get_scheduler(optimizer)
        epoch_scheduler = None
    else:
        iter_scheduler = None
        epoch_scheduler = config.get_scheduler(optimizer)

    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if hasattr(config, 'resume'):
        oss_proxy = OssProxy()
        resume_params = torch.load(io.BytesIO(oss_proxy.download('{}_params.pth'.format(config.resume))), 'cpu')
        ddp_model.load_state_dict(resume_params)
        logger.info('load params from {}'.format('{}_params.pth'.format(config.resume)))
        resume_opts = torch.load(io.BytesIO(oss_proxy.download('{}_opt.pth'.format(config.resume))), 'cpu')
        optimizer.load_state_dict(resume_opts)
        logger.info('load opts from {}'.format('{}_opt.pth'.format(config.resume)))

        if use_amp:
            resume_scaler = torch.load(io.BytesIO(oss_proxy.download('{}_scaler.pth'.format(config.resume))), 'cpu')
            amp_scaler.load_state_dict(resume_scaler)
            logger.info('load scaler from {}'.format('{}_scaler.pth'.format(config.resume)))
        
        resume_scheduler = torch.load(io.BytesIO(oss_proxy.download('{}_scheduler.pth'.format(config.resume))), 'cpu')
        if iter_scheduler is not None:
            iter_scheduler.load_state_dict(resume_scheduler)
        elif epoch_scheduler is not None:
            epoch_scheduler.load_state_dict(resume_scheduler)
        else:
            raise ValueError('wrong scheduler')

        logger.info('load scheduler from {}'.format('{}_scheduler.pth'.format(config.resume)))

        resume_prefix = config.resume.split('/')[-1]
        if '_' in resume_prefix:
            resume_prefix = resume_prefix.split('_')[0]
        config.start_epoch = int(resume_prefix.replace('epoch', '')) + 1

    if hasattr(config, 'EPOCH'):
        EPOCH = config.EPOCH
    else:
        EPOCH = 30
    for epoch in range(config.start_epoch, EPOCH):
        
        all_train_loaders = config.get_train_dataloader(epoch)

        if epoch_scheduler is not None:
            logger.info('epoch {} training starts, lr: {}'.format(epoch, epoch_scheduler.get_last_lr()))
        ddp_model.train()

        for train_loader, train_name in all_train_loaders:
            if hasattr(config, 'use_sam') and config.use_sam:
                train_epoch_sam(ddp_model, optimizer, train_loader, epoch, dist_info, logger, amp_scaler, config, node_group, use_amp, iter_scheduler)
            else:
                train_epoch(ddp_model, optimizer, train_loader, epoch, dist_info, logger, amp_scaler, config, node_group, use_amp, iter_scheduler)

        if global_rank == 0:
            logger.info('saving models to oss')
            save_model_to_oss('{}/epoch{}_params.pth'.format(config.exp_dir, epoch), ddp_model)
            save_model_to_oss('{}/epoch{}_scaler.pth'.format(config.exp_dir, epoch), amp_scaler)
            save_model_to_oss('{}/epoch{}_opt.pth'.format(config.exp_dir, epoch), optimizer)
            if iter_scheduler is not None:
                save_model_to_oss('{}/epoch{}_scheduler.pth'.format(config.exp_dir, epoch), iter_scheduler)
            elif epoch_scheduler is not None:
                save_model_to_oss('{}/epoch{}_scheduler.pth'.format(config.exp_dir, epoch), epoch_scheduler)
            else:
                logger.error('skip saving scheduler, because both iter_scheduler and epoch_scheduler are None')

        if epoch_scheduler is not None:
            epoch_scheduler.step()

def worker(local_rank, ngpu, dist_world_size, node_rank):
    global_rank = local_rank + node_rank * ngpu
    worker_th_launch(local_rank, dist_world_size, global_rank)

if __name__ == '__main__':
    print(os.environ["WORLD_SIZE"], os.environ["RANK"])
    worker_th_launch(
        local_rank=int(os.environ['LOCAL_RANK']),
        dist_world_size=int(os.environ["WORLD_SIZE"]),
        global_rank=int(os.environ["RANK"])
    )
