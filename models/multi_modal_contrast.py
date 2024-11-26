from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist

from models.base_model import BaseModel
from models.utils import GatherLayer
from utils.logging import AverageMeter
from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()


class MultiModalContrast(BaseModel):
    def __init__(self, text_model, image_model, pretrained, mask_noise=False):
        super(MultiModalContrast, self).__init__()
        self.text_model = text_model
        self.image_model = image_model
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.66)  # ln(1/0.07)=2.66

        self.loss_metric = AverageMeter('Loss')

        self.vis_tensor_projector = nn.Identity()

        if pretrained is not None:
            logger.info('loading from {}'.format(pretrained))
            self.load_weight_from_file(pretrained)
        
        self.mask_noise = mask_noise

        self.test_gt_list = []
        self.test_score_list = []

    def get_feature(self, text_data=None, text_mask=None, img_tensor=None, return_before_l2=False):
        if text_data is not None and text_mask is not None:
            text_feature, text_tensor = self.text_model(text_data, text_mask, return_tensor=True)
            text_feature = F.normalize(text_feature, p=2.0, dim=1)
        else:
            text_feature, text_tensor = None, None
        
        if img_tensor is not None:
            image_feature, image_tensor = self.image_model(img_tensor.type(next(self.image_model.parameters()).dtype), return_tensor=True)
            if return_before_l2:
                pass
            else:
                image_feature = F.normalize(image_feature, p=2.0, dim=1)
        else:
            image_feature, image_tensor = None, None
        return text_feature, text_tensor, image_feature, image_tensor
    
    def forward(self, *args, **kwargs):
        if kwargs['phase'] == 'eval':
            batch_data, dist_info = args
            return self.forward_test(batch_data, dist_info)
        elif kwargs['phase'] == 'train':
            batch_data, dist_info, print_log, log_info = args
            return self.forward_train(batch_data, dist_info, print_log, log_info)
        else:
            raise ValueError
 
    def forward_test(self, batch_data, dist_info):
        with torch.no_grad():
            text_data = batch_data.text
            text_mask = batch_data.text_mask
            img_tensor = batch_data.img

            device = torch.device('cuda:{}'.format(dist_info['local_rank']))
            text_data = text_data.to(device, non_blocking=True)
            text_mask = text_mask.to(device, non_blocking=True)
            img_tensor = img_tensor.to(device, non_blocking=True)

            text_feature, _, image_feature, _ = self.get_feature(text_data, text_mask, img_tensor)

            match_score_mat = torch.mm(text_feature, image_feature.t())
            score = match_score_mat.data.cpu().numpy().flatten()
            assert match_score_mat.shape[0] == match_score_mat.shape[1]
            gt = torch.eye(match_score_mat.shape[0]).long().numpy().flatten()
            self.test_gt_list.extend(gt)
            self.test_score_list.extend(score)
            logger.info('AP: {}'.format(average_precision_score(self.test_gt_list, self.test_score_list)))
            return match_score_mat

    def contrastive_loss(self, logits, dim):
        neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
        return -neg_ce.mean()

    def clip_loss(self, text_image_similarity, image_text_similarity, img_idx=None, all_img_idx=None):
        if img_idx is not None and all_img_idx is not None:
            with torch.no_grad():
                false_neg_indicator = (img_idx[:, None] == all_img_idx[None, :])
                false_neg_indicator.fill_diagonal_(False)
            text_image_similarity.masked_fill_(false_neg_indicator, float('-inf'))
            image_text_similarity.masked_fill_(false_neg_indicator, float('-inf'))
            caption_loss = self.contrastive_loss(text_image_similarity, dim=1)
            image_loss = self.contrastive_loss(image_text_similarity, dim=1)
        else:
            caption_loss = self.contrastive_loss(text_image_similarity, dim=1)
            image_loss = self.contrastive_loss(image_text_similarity, dim=1)
        return (caption_loss + image_loss) / 2.0

    def re_arrange_tensor(self, gathered_list, world_size, global_rank):
        assert len(gathered_list) == world_size
        return [gathered_list[global_rank], ] + [gathered_list[idx] for idx in range(world_size) if idx != global_rank]

    def forward_train(self, batch_data, dist_info, print_log, log_info):
        if len(self.test_gt_list) > 0:
            self.test_gt_list = []
        if len(self.test_score_list) > 0:
            self.test_score_list = []
            
        text_data = batch_data.text
        text_mask = batch_data.text_mask
        img_tensor = batch_data.img

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        device = torch.device('cuda:{}'.format(dist_info['local_rank']))
        text_data = text_data.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        img_tensor = img_tensor.to(device, non_blocking=True)

        text_feature, _, image_feature, _ = self.get_feature(text_data, text_mask, img_tensor)

        if self.mask_noise:
            batch_img_idx = batch_data.misc_list
            batch_img_idx = torch.tensor(batch_img_idx, dtype=torch.long, device=text_feature.device)
            gather_layer0 = GatherLayer.apply
            world_img_idx = gather_layer0(batch_img_idx)
            world_img_idx = torch.cat(self.re_arrange_tensor(world_img_idx, dist.get_world_size(), dist.get_rank()), dim=0)

        gather_layer1 = GatherLayer.apply
        tv_feature = torch.cat((text_feature, image_feature), dim=1)
        world_feature = gather_layer1(tv_feature)
        world_text_feature, world_image_feature = torch.split(
            torch.cat(self.re_arrange_tensor(world_feature, dist.get_world_size(), dist.get_rank()), dim=0),
            [text_feature.shape[1], image_feature.shape[1]],
            dim=1
        )

        ###########################################################
        # contrast learning
        ###########################################################
        global_rank = dist_info['global_rank']
        text_img_sco = torch.matmul(text_feature, world_image_feature.t()) * logit_scale
        img_text_sco = torch.matmul(image_feature, world_text_feature.t()) * logit_scale

        loss = self.clip_loss(
            text_img_sco,
            img_text_sco,
            img_idx=batch_img_idx if self.mask_noise else None,
            all_img_idx=world_img_idx if self.mask_noise else None)
        self.loss_metric.update(loss.item())

        if print_log:
            info_str = 'rank={}, epoch={}, batch={}/{}, batch_size={}, loss={:.4f}, logit_scale={:.4f}'.format(
                    global_rank, log_info['epoch'], log_info['batch_idx'], log_info['all_batch_cnt'], tv_feature.shape[0], self.loss_metric.avg, logit_scale.item())
            logger.info(info_str)

        return loss
