import time
import torch
import torch.nn.functional as F
import numpy as np
from utils.eval_utils import evaluation_imagenet_zs, padding_tokens77
from datasets.imagenet import get_imagenet_cname_retrieve_dataloaders

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
    pretrained='snapshots/fuse_teacher_b32_yfcc15m_lamb/epoch24_params.pth')
NORM_TYPE = 2
MAX_WORDS = 77
LOWERCASE = True
vocab_file = 'pretrained/bert-base-uncased-vocab.txt'
padding_func = padding_tokens77

for get_dataloader_func in [ get_imagenet_cname_retrieve_dataloaders, ]:
    #####################################################################
    ############################# dataloader ############################
    #####################################################################
    re_val_dataloader_list = get_dataloader_func(batch_size=64, img_reso=IMG_SIZE, norm_type=NORM_TYPE)


    #####################################################################
    ############################# tokenizer ############################
    #####################################################################
    import tempfile
    from tokenizers import BertWordPieceTokenizer
    from utils.oss_op import get_bucket
    bucket = get_bucket()
    vocab_fp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=True)
    bucket.get_object_to_file(vocab_file, vocab_fp.name)
    text_tokenizer = BertWordPieceTokenizer(vocab_fp.name, lowercase=LOWERCASE)
    text_tokenizer.enable_truncation(max_length=MAX_WORDS)
    vocab_fp.close()

    for re_val_dataloader, _ in re_val_dataloader_list:
        evaluation_imagenet_zs(model, re_val_dataloader, text_tokenizer, device=torch.device('cuda'), padding_tokens=padding_func)
    print(get_dataloader_func)
