import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torch.distributed as dist
from torch.nn.utils import weight_norm

from models.base_model import BaseModel
from models.utils import GatherLayer
from utils.logging import AverageMeter
from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()
        self.last_layer = weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.last_layer(x)
        return x

class PreCrossModel(nn.Module):
    def __init__(self, feat_dim=768, mlp_ratio=2, use_dropout=False, use_mlp=True, use_self_attn=True):
        super(PreCrossModel, self).__init__()
        self.use_dropout = use_dropout
        self.use_mlp = use_mlp
        self.use_self_attn = use_self_attn

        if self.use_self_attn:
            self.norm1 = nn.LayerNorm(feat_dim)
        self.norm2 = nn.LayerNorm(feat_dim)
        if self.use_mlp:
            self.norm3 = nn.LayerNorm(feat_dim)
        
        if self.use_self_attn:
            self.self_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=feat_dim // 64)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=feat_dim // 64)
        if self.use_mlp:
            self.ffn = Mlp(in_features=feat_dim, hidden_features=feat_dim*mlp_ratio, drop=0.1 if self.use_dropout else 0.0)

        if self.use_dropout:
            if self.use_self_attn:
                self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)
            if self.use_mlp:
                self.dropout3 = nn.Dropout(0.1)
        else:
            if self.use_self_attn:
                self.dropout1 = nn.Identity()
            self.dropout2 = nn.Identity()
            if self.use_mlp:
                self.dropout3 = nn.Identity()
    
    def forward(self, image_tensors, text_tensors, text_masks):
        if self.use_self_attn:
            image_tensors_res = self.norm1(image_tensors)
            image_tensors_res = self.self_attn(
                image_tensors_res.permute(1, 0, 2),
                image_tensors_res.permute(1, 0, 2),
                image_tensors_res.permute(1, 0, 2),
            )[0].permute(1, 0, 2)
            image_tensors = image_tensors + self.dropout1(image_tensors_res)

        if (text_tensors is not None) and (text_masks is not None):
            image_tensors_res = self.norm2(image_tensors)
            image_tensors_res = self.cross_attn(
                image_tensors_res.permute(1, 0, 2),
                text_tensors.permute(1, 0, 2),
                text_tensors.permute(1, 0, 2),
                key_padding_mask=(text_masks==0)
            )[0].permute(1, 0, 2)
            image_tensors = image_tensors + self.dropout2(image_tensors_res)
        
        if self.use_mlp:
            image_tensors_res = self.norm3(image_tensors)
            image_tensors = image_tensors + self.dropout3(self.ffn(image_tensors_res))

        return image_tensors


class MultiModalFuseTeacher(BaseModel):
    def __init__(self, text_model, image_model, pretrained, mask_noise=False, distill_loss_w=1.0, num_centers=4096, distill_sim=False,
            mlp_ratio=2, fusion_norm=False, hidden_dim=768, feat_dim=512, use_gc=False, num_fusion_blocks=2):
        super(MultiModalFuseTeacher, self).__init__()
        self.text_model = text_model
        self.image_model = image_model

        self.use_gc = use_gc
        self.num_fusion_blocks = num_fusion_blocks
        self.fusion_blocks = nn.ModuleList(
            [PreCrossModel(
                feat_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                use_dropout=True,
                use_mlp=True,
                use_self_attn=True) for _ in range(self.num_fusion_blocks)]
        )
        if fusion_norm:
            self.fusion_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        else:
            self.fusion_norm = nn.Identity()
        self.fusion_proj = nn.Linear(hidden_dim, feat_dim, bias=False)

        if distill_loss_w > 0:
            out_dim = num_centers
            self.dino_head = DINOHead(in_dim=feat_dim, out_dim=out_dim)

        self.logit_scale_fusion1 = nn.Parameter(torch.ones([]) * 2.66)  # ln(1/0.07)=2.66
        self.logit_scale_fusion2 = nn.Parameter(torch.ones([]) * 2.66)  # ln(1/0.07)=2.66
        if distill_loss_w > 0:
            self.logit_scale_distill = nn.Parameter(torch.ones([]) * 2.66)  # ln(1/0.07)=2.66

        self.loss_clip_metric = AverageMeter('Loss1')
        self.loss_fusion_metric = AverageMeter('Loss2')
        if distill_loss_w > 0:
            self.loss_distill_center_metric = AverageMeter('Loss3')
            self.teacher_entropy_metric = AverageMeter('TeacherMetric')
            self.rnd_teacher_entropy_metric = AverageMeter('RndTeacherMetric')

        self.distill_loss_w = distill_loss_w

        self.distill_sim = distill_sim
        if self.distill_sim:
            self.loss_distill_sim_metric = AverageMeter('Loss3')

        if pretrained is not None:
            logger.info('loading from {}'.format(pretrained))
            self.load_weight_from_file(pretrained)
        
        self.mask_noise = mask_noise

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

    def kl_loss(self, logits, target_logits, dim):
        loss = -torch.sum(
            F.softmax(target_logits.detach(), dim=1) * \
                F.log_softmax(logits, dim=dim),
            dim=1)
        loss = loss.mean()
        return loss

    def distill_sim_loss(self,
            student_text_image_similarity, student_image_text_similarity,
            teacher_text_image_similarity, teacher_image_text_similarity):
        distill_caption_loss = self.kl_loss(student_text_image_similarity, teacher_text_image_similarity, dim=1)
        distill_image_loss = self.kl_loss(student_image_text_similarity, teacher_image_text_similarity, dim=1)
        return (distill_caption_loss + distill_image_loss) / 2.0

    def re_arrange_tensor(self, gathered_list, world_size, global_rank):
        assert len(gathered_list) == world_size
        return [gathered_list[global_rank], ] + [gathered_list[idx] for idx in range(world_size) if idx != global_rank]

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_logits, n_iterations=3):
        teacher_logits = teacher_logits.float()
        world_size = dist.get_world_size()
        Q = torch.exp(teacher_logits).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward_train(self, batch_data, dist_info, print_log, log_info):
        text_data = batch_data.text
        text_mask = batch_data.text_mask
        img_tensor = batch_data.img

        text_data2 = batch_data.mlm_text
        text_mask2 = batch_data.mlm_text_mask

        logit_scale_fusion1 = self.logit_scale_fusion1.exp().clamp(max=100.0)
        logit_scale_fusion2 = self.logit_scale_fusion2.exp().clamp(max=100.0)
        if self.distill_loss_w > 0:
            logit_scale_distill = self.logit_scale_distill.exp().clamp(max=100.0)

        device = torch.device('cuda:{}'.format(dist_info['local_rank']))
        text_data = text_data.to(device, non_blocking=True)
        text_mask = text_mask.to(device, non_blocking=True)
        img_tensor = img_tensor.to(device, non_blocking=True)

        text_data2 = text_data2.to(device, non_blocking=True)
        text_mask2 = text_mask2.to(device, non_blocking=True)

        text_feature, _, image_feature, image_tensor = self.get_feature(text_data, text_mask, img_tensor)
        _, text_tensor2, _, _ = self.get_feature(text_data2, text_mask2)

        # get fusion feature
        fusion_tensor = image_tensor
        for each_fusion_block in self.fusion_blocks:
            if self.use_gc:
                fusion_tensor = checkpoint.checkpoint(each_fusion_block, fusion_tensor, text_tensor2, text_mask2)
            else:
                fusion_tensor = each_fusion_block(fusion_tensor, text_tensor2, text_mask2)
        fusion_tensor = self.fusion_norm(fusion_tensor)
        # NOTE: index=0 is not always cls token
        fusion_feature = fusion_tensor[:, 0, :]
        fusion_feature = self.fusion_proj(fusion_feature)
        fusion_feature = F.normalize(fusion_feature, p=2.0, dim=1)
        
        if self.mask_noise:
            batch_img_idx = batch_data.misc_list
            batch_img_idx = torch.tensor(batch_img_idx, dtype=torch.long, device=text_feature.device)
            gather_layer0 = GatherLayer.apply
            world_img_idx = gather_layer0(batch_img_idx)
            world_img_idx = torch.cat(self.re_arrange_tensor(world_img_idx, dist.get_world_size(), dist.get_rank()), dim=0)

        gather_layer1 = GatherLayer.apply
        tv_feature = torch.cat((text_feature, image_feature, fusion_feature), dim=1)
        world_feature = gather_layer1(tv_feature)
        world_text_feature, world_image_feature, world_fusion_feature = torch.split(
            torch.cat(self.re_arrange_tensor(world_feature, dist.get_world_size(), dist.get_rank()), dim=0),
            [text_feature.shape[1], image_feature.shape[1], fusion_feature.shape[1]],
            dim=1
        )

        ###########################################################
        # contrast learning
        ###########################################################
        global_rank = dist_info['global_rank']
        text_img_sco = torch.matmul(text_feature, world_image_feature.t()) * logit_scale_fusion1
        img_text_sco = torch.matmul(image_feature, world_text_feature.t()) * logit_scale_fusion1

        loss_clip = self.clip_loss(
            text_img_sco,
            img_text_sco,
            img_idx=batch_img_idx if self.mask_noise else None,
            all_img_idx=world_img_idx if self.mask_noise else None)
        self.loss_clip_metric.update(loss_clip.item())

        text_fusion_sco = torch.matmul(text_feature, world_fusion_feature.t()) * logit_scale_fusion2
        fusion_text_sco = torch.matmul(fusion_feature, world_text_feature.t()) * logit_scale_fusion2

        loss_fusion = self.clip_loss(
            text_fusion_sco,
            fusion_text_sco,
            img_idx=batch_img_idx if self.mask_noise else None,
            all_img_idx=world_img_idx if self.mask_noise else None)
        self.loss_fusion_metric.update(loss_fusion.item())

        ###########################################################
        # distill learning
        ###########################################################
        if self.distill_loss_w > 0:
            if self.distill_sim:
                loss_distill_sim = self.distill_sim_loss(
                    text_img_sco, img_text_sco,
                    text_fusion_sco, fusion_text_sco
                )
                loss_distill_sim = loss_distill_sim * self.distill_loss_w
                self.loss_distill_sim_metric.update(loss_distill_sim.item())
            else:
                loss_distill_sim = 0
                self.loss_distill_sim_metric.update(0)


            teacher_logit = self.dino_head(fusion_feature.detach())

            with torch.no_grad():
                teacher_logit = teacher_logit * logit_scale_fusion2
                teacher_prob = self.sinkhorn_knopp_teacher(teacher_logit, 3)
                teacher_entropy = -torch.mean(
                    torch.sum(teacher_prob * torch.log(torch.clamp(teacher_prob, min=1e-8)), dim=1)
                ).item()
                self.teacher_entropy_metric.update(teacher_entropy)

                rnd_teacher_entropy = -torch.mean(
                    teacher_prob @ torch.log(torch.clamp(teacher_prob, min=1e-8)).t()
                ).item()
                self.rnd_teacher_entropy_metric.update(rnd_teacher_entropy)

            student_logit = self.dino_head(image_feature) * logit_scale_distill

            loss_distill_center = -torch.sum(
                    teacher_prob.detach() * F.log_softmax(student_logit, dim=1),
                dim=1)
            loss_distill_center = loss_distill_center.mean()

            loss_distill_center = loss_distill_center * self.distill_loss_w
            self.loss_distill_center_metric.update(loss_distill_center.item())

            if print_log:
                info_str = 'rank={}, epoch={}, batch={}/{}, batch_size={}, loss_clip={:.4f}, loss_fusion={:.4f}, loss_distill_center={:.4f}, loss_distill_sim={:.4f}, logit_scale_fusion1={:.4f}, logit_scale_fusion2={:.4f}, logit_scale_distill={:.4f}, teacher_entropy_metric={:.4f}, rnd_teacher_entropy_metric={:.4f}'.format(
                        global_rank, log_info['epoch'], log_info['batch_idx'], log_info['all_batch_cnt'], tv_feature.shape[0], self.loss_clip_metric.avg,
                        self.loss_fusion_metric.avg, self.loss_distill_center_metric.avg, self.loss_distill_sim_metric.avg, logit_scale_fusion1.item(), logit_scale_fusion2.item(), logit_scale_distill.item(),
                        self.teacher_entropy_metric.avg, self.rnd_teacher_entropy_metric.avg)
                logger.info(info_str)

            return loss_clip + loss_fusion + loss_distill_center + loss_distill_sim

        else:
            if print_log:
                info_str = 'rank={}, epoch={}, batch={}/{}, batch_size={}, loss_clip={:.4f}, loss_fusion={:.4f}, logit_scale_fusion1={:.4f}, logit_scale_fusion2={:.4f}'.format(
                        global_rank, log_info['epoch'], log_info['batch_idx'], log_info['all_batch_cnt'], tv_feature.shape[0], self.loss_clip_metric.avg,
                        self.loss_fusion_metric.avg, logit_scale_fusion1.item(), logit_scale_fusion2.item())
                logger.info(info_str)

            return loss_clip + loss_fusion
