exp_dir = 'snapshots/clip_b32_yfcc15m_lamb'
eval_only = False
start_epoch = 0
use_iter_scheduler = True
EPOCH = 25
WARMUP_EPOCH = 4

#####################################################################
############################# model #############################
#####################################################################
import torch
import torch.nn as nn
from models.backbone.bert import BertWrapper
from models.backbone.vit_large import VITLargeWrapper
from models.multi_modal_contrast import MultiModalContrast
text_model = BertWrapper(use_cls_token=True, use_gradient_ckpt=False)
text_model.bert.cls = None
IMG_SIZE=224
# random initialize vit
image_model = VITLargeWrapper(img_size=IMG_SIZE, use_gradient_checkpoint=False, model_type='base_p32', pretrained=False)
model = MultiModalContrast(
    text_model=text_model,
    image_model=image_model,
    pretrained=None)

#####################################################################
############################# optimizer #############################
#####################################################################
import apex.optimizers.fused_lamb as fused_lamb
learning_rate_default = 2.5e-3
betas = [0.9, 0.98]
weight_decay = 0.2
adamw_eps = 1e-6
def get_params_group(model, weight_decay, skip_list=(), logger=None):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) < 2 or name.endswith(".bias") or 'logit_scale' in name or name in skip_list:
            no_decay.append(param)
            if logger is not None:
                logger.info('{} has NO weight decay'.format(name))
        else:
            decay.append(param)
            if logger is not None:
                logger.info('{} has weight decay'.format(name))
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
def get_optimizer(model, logger=None):
    return fused_lamb.FusedLAMB(get_params_group(model, weight_decay=weight_decay, logger=logger), lr=learning_rate_default, betas=betas, eps=adamw_eps, weight_decay=weight_decay)

#####################################################################
############################# scheduler #############################
#####################################################################
# 按照iteration来设置lr
import torch
import math
def lr_lmbda(iter_cnt):
    warmup_iters = 3606 * WARMUP_EPOCH
    epoch_iters = 3606  # 14772582 / 2048
    if iter_cnt < warmup_iters:
        return (iter_cnt + 1) / warmup_iters
    else:
        iter_after_warm = iter_cnt - warmup_iters
        lr_min = 1e-5
        lr_max = 1.0
        iter_max = epoch_iters * (EPOCH - WARMUP_EPOCH)
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(iter_after_warm / iter_max * math.pi))
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lmbda)

#####################################################################
############################# dataset ###############################
#####################################################################
import pickle
import io
import tempfile
from tokenizers import BertWordPieceTokenizer
from utils.oss_op import get_bucket
from datasets.utils import collate_fn77_cn
from datasets.multi_dataset import get_multi_dataloaders
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 480
bucket = get_bucket()
vocab_fp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=True)
bucket.get_object_to_file('pretrained/bert-base-uncased-vocab.txt', vocab_fp.name)
text_tokenizer = BertWordPieceTokenizer(vocab_fp.name, lowercase=True)
text_tokenizer.enable_truncation(max_length=77)
vocab_fp.close()

def get_train_dataloader(epoch):
    dataname2epoch_size = {
        'yfcc15m_llava': 923286-1,
    }
    return get_multi_dataloaders(TRAIN_BATCH_SIZE, epoch, dataname2epoch_size, text_tokenizer, max_words=77, img_reso=IMG_SIZE, norm_type=2,
        buf_num=256, num_workers=24, collate_fn=collate_fn77_cn, new_da=True, multi_text=True)
