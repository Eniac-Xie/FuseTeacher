import json
import io
import os
import random
import tempfile
import re
import time
import tarfile
import copy

import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, \
        RandomHorizontalFlip, RandomResizedCrop, RandomApply, RandomGrayscale, ColorJitter
from transformers import BertTokenizerFast, DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer

try:
    import sng_parser
except:
    print('sng_parser not installed')

import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile
from PIL import ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from datasets.utils import MMData, openai_text_processor
from utils.oss_op import get_bucket
from models.tokenizer import CustomTokenizerWrapper
from utils.logging import MultiModalLogging

logging = MultiModalLogging()
logger = logging.get()


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TarDataset(Dataset):
    def __init__(self,
                dataset_name,
                epoch,
                epoch_size,
                min_tar_id,
                max_tar_id,
                name_format,
                tar_root,
                sample_id_base,
                tokenizer,
                max_words=30,
                img_reso=224,
                norm_type=0,
                new_da=False,
                text_processor=None,
                tar_size=16,
                return_str=False,
                return_sample_name=False,
                sample_tag=False,
                multi_text=False,
                multi_image=False,
                slip_image=False):
        sid = range(min_tar_id, max_tar_id+1)[(epoch * epoch_size) % (max_tar_id+1 - min_tar_id)]
        eid = range(min_tar_id, max_tar_id+1)[((epoch + 1) * epoch_size) % (max_tar_id+1 - min_tar_id)]

        if eid > sid:
            epoch_tar_id_list = list(range(sid, eid))
            logger.info('{} ({}-{}) use {}-{}'.format(dataset_name, min_tar_id, max_tar_id, sid, eid))
        elif eid < sid:
            epoch_tar_id_list = list(range(sid, max_tar_id + 1)) + list(range(min_tar_id, eid))
            logger.info('{} ({}-{}) use {}-{} and {}-{}'.format(dataset_name, min_tar_id, max_tar_id, sid, max_tar_id, min_tar_id, eid))
        else:
            raise ValueError
        self.epoch_tar_id_list = epoch_tar_id_list
        assert len(self.epoch_tar_id_list) == epoch_size

        self.name_format = name_format
        self.tar_root = tar_root
        self.sample_id_base = sample_id_base
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.bucket = get_bucket(name='multimodal')
        self.bucket_name = 'multimodal'

        if norm_type == 0:
            norm_op = Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711))
        elif norm_type == 1:
            # for vit-huge model
            norm_op = Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
        elif norm_type == 2:
            # for mae model
            from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
            norm_op = Normalize(
                IMAGENET_DEFAULT_MEAN,
                IMAGENET_DEFAULT_STD)
        else:
            raise ValueError

        if new_da:
            from datasets.randaugment import RandomAugment
            self.transform = [
                    Compose([                        
                        RandomResizedCrop(img_reso, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                        RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        RandomGrayscale(p=0.2),
                        RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                        RandomHorizontalFlip(),
                        RandomAugment(2, 7, isPIL=True, augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                            'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                        ToTensor(),
                        norm_op
                    ])
            ]
            if slip_image:
                self.weak_transform = [
                        Compose([                        
                            RandomResizedCrop(img_reso, scale=(0.64, 1.0), interpolation=Image.BICUBIC),
                            RandomHorizontalFlip(),  
                            ToTensor(),
                            norm_op
                        ])
                ]
        else:
            self.transform = [
                Compose([                        
                        RandomResizedCrop(img_reso, scale=(0.64, 1.0), interpolation=Image.BICUBIC),
                        RandomHorizontalFlip(),
                        ToTensor(),
                        norm_op
                ])
            ]
        
        self.dataset_name = dataset_name
        self.text_processor = text_processor
        self.tar_size = tar_size
        self.return_str = return_str
        self.return_sample_name = return_sample_name
        self.sample_tag = sample_tag
        self.multi_text = multi_text
        self.multi_image = multi_image
        self.slip_image = slip_image

        if self.sample_tag:
            self.tag_set = open('yfcc15m_tag_wo_prompt.txt').read().splitlines()
            self.tag_set = {tmp: None for tmp in self.tag_set}

    def __len__(self):
        return len(self.epoch_tar_id_list)

    def prepare_mmdata_worker(self, text_raw, image_raw, sample_id):
        if self.text_processor is not None:
            text_raw = self.text_processor(text_raw, self.max_words)

        if isinstance(self.tokenizer, BertTokenizerFast):
            text = self.tokenizer(text_raw, return_tensors='pt', max_length=77, padding=True, truncation=True)
            token_ids, token_attention_mask = text.input_ids[0], text.attention_mask[0]
        elif isinstance(self.tokenizer, DistilBertTokenizer):
            text = self.tokenizer(text_raw, return_tensors='pt', max_length=self.max_words, padding=True, truncation=True)
            token_ids, token_attention_mask = text.input_ids.tolist()[0], text.attention_mask.tolist()[0]
        elif isinstance(self.tokenizer, BertWordPieceTokenizer) or isinstance(self.tokenizer, CustomTokenizerWrapper):
            text = self.tokenizer.encode(text_raw)
            token_ids, token_attention_mask = text.ids, text.attention_mask
        else:                
            text = self.tokenizer(text_raw, truncation=True, max_length=self.max_words)
            token_ids, token_attention_mask = text.input_ids, text.attention_mask
        
        assert isinstance(token_ids, list) and isinstance(token_attention_mask, list)

        try:
            image_pil = Image.open(io.BytesIO(image_raw)).convert('RGB')
        except Exception as err:
            print(err, flush=True)
            image_stream = io.BytesIO()
            image_stream.write(image_raw)
            image_stream.seek(0)
            cv2_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            image_pil = cv2.imdecode(cv2_bytes, cv2.IMREAD_COLOR)
            image_pil = image_pil[:, :, ::-1].copy()
            image_pil = Image.fromarray(image_pil)
            
        image = random.choice(self.transform)(image_pil)
        if self.multi_image:
            if self.slip_image:
                aug_image1 = random.choice(self.transform)(image_pil)
                aug_image2 = random.choice(self.weak_transform)(image_pil)
                aug_image = torch.cat(
                    [aug_image1, aug_image2],
                    dim=0
                )
            else:
                aug_image = random.choice(self.transform)(image_pil)
        else:
            aug_image = None

        if self.return_sample_name or self.return_str:
            misc_info = []
            if self.return_sample_name:
                assert len(sample_id.split(':')) == 2
                sample_name = sample_id.split(':')[0]
                misc_info.append(sample_name)
            if self.return_str:
                misc_info.append(text_raw)
            misc_info = '_xcw_fd_'.join(misc_info)
        else:
            misc_info = int(sample_id.rsplit(':', 1)[1])
        
        data = MMData(
            torch.tensor(token_ids),
            torch.tensor(token_attention_mask),
            image,
            misc=misc_info,
            aug_img=aug_image
        )
        return data

    def prepare_mmdata(self, text_raw, image_raw, sample_id):
        if isinstance(text_raw, list):
            assert len(text_raw) == 2
            data1 = self.prepare_mmdata_worker(text_raw[0], image_raw, sample_id)
            data2 = self.prepare_mmdata_worker(text_raw[1], image_raw, sample_id)
            data1.mlm_text = data2.text
            data1.mlm_text_mask = data2.text_mask
            return data1
        else:
            return self.prepare_mmdata_worker(text_raw, image_raw, sample_id)

    def get_by_index(self, index):
        tar_name = self.name_format.format(index)
        tar_oss_path = '{}/{}'.format(self.tar_root, tar_name)

        if os.path.exists('oss/{}'.format(tar_oss_path)):
            tmp_fp = open('oss/{}'.format(tar_oss_path), 'rb')
        else:
            if not self.bucket.object_exists(tar_oss_path):
                raise FileNotFoundError

            tmp_fp = tempfile.NamedTemporaryFile(suffix='.tar', delete=True)
            self.bucket.get_object_to_file(tar_oss_path, tmp_fp.name)
            tmp_fp.flush()
        tar_fp = tarfile.open(tmp_fp.name, 'r')
        member_list = tar_fp.getmembers()
        prefix2content = {}
        for member_idx, member in enumerate(member_list):
            prefix_name, suffix_name = member.name.rsplit('.', 1)
            if 'jpg' != suffix_name:
                continue

            jpg_content = tar_fp.extractfile(member).read()
            try:
                txt_content = tar_fp.extractfile('{}.txt'.format(prefix_name)).read().decode('UTF-8')
            except:
                txt_content = json.loads(tar_fp.extractfile('{}.json'.format(prefix_name)).read())
                if isinstance(txt_content, dict):
                    txt_content = txt_content['caption']
                if self.multi_text:
                    assert isinstance(txt_content, list) and len(txt_content) >= 2
                    txt_content = random.sample(txt_content, 2)
                else:
                    if isinstance(txt_content, list):
                        txt_content = txt_content[0]

            sample_name = '{}_{}_{}'.format(self.bucket_name, tar_oss_path, member_idx)
            sample_id = '{}:{}'.format(sample_name, self.sample_id_base + index * self.tar_size + member_idx)
            prefix2content[prefix_name] = (jpg_content, txt_content, sample_id)

        tar_fp.close()
        tmp_fp.close()

        return prefix2content
    
    def __getitem__(self, index):
        for try_idx in range(10):
            try:
                tar_id = self.epoch_tar_id_list[index]
                prefix2content = self.get_by_index(tar_id)
                if len(prefix2content) != self.tar_size:
                    raise ValueError('{}/{}'.format(self.tar_root, self.name_format.format(tar_id)))

                text_list, img_list, sample_id_list = [], [], []
                for jpg_content, txt_content, sample_id in prefix2content.values():
                    text_list.append((txt_content))
                    img_list.append(jpg_content)
                    sample_id_list.append(sample_id)

            except Exception as err:
                img_list = None
                text_list = None
            
            if img_list is not None and text_list is not None:
                break
            else:
                index = random.choice(range(len(self.epoch_tar_id_list)))
        
        assert img_list is not None
        if self.sample_tag:
            assert not self.multi_text
            new_text_list = []
            for each_text in text_list:
                each_text_graph = sng_parser.parse(each_text)
                current_tag_list = []
                for tag_info in each_text_graph['entities']:
                    tag = tag_info['lemma_head']
                    if tag in self.tag_set:
                        current_tag_list.append(
                            'a photo of {}.'.format(tag)
                        )
                if len(current_tag_list) > 0:
                    if random.random() > 0.9:
                        new_text_list.append(
                            random.choice(current_tag_list)
                        )
                    else:
                        new_text_list.append(each_text)
                else:
                    new_text_list.append(each_text)
            text_list = new_text_list
        rawdata_list = [[text_raw, img_raw, sample_id] for text_raw, img_raw, sample_id in zip(text_list, img_list, sample_id_list)]
        return rawdata_list

def get_yfcc15m_llava_trainset(epoch, epoch_size, sample_id_base, text_tokenizer, max_words, img_reso, norm_type, new_da=False, return_str=False, return_sample_name=False, sample_tag=False, multi_text=False, multi_image=False, slip_image=False):
    dataset_name = 'yfcc15m_llava'
    min_tar_id = 1
    max_tar_id = 923286
    name_format = '{:07d}.tar'
    tar_root = 'datasets/yfcc15m_llava/tars'

    train_dataset = TarDataset(
            dataset_name,
            epoch,
            epoch_size,
            min_tar_id,
            max_tar_id,
            name_format,
            tar_root,
            sample_id_base,
            text_tokenizer,
            max_words,
            img_reso,
            norm_type,
            new_da=new_da,
            text_processor=openai_text_processor,
            tar_size=16,
            return_str=return_str,
            return_sample_name=return_sample_name,
            sample_tag=sample_tag,
            multi_text=multi_text,
            multi_image=multi_image,
            slip_image=slip_image)

    return train_dataset
