import tempfile
import ftfy
import re
import html
import random
from utils.oss_op import OssProxy

class MMData(object):
    def __init__(self, text, text_mask, img, misc, mlm_text=None, mlm_target=None, mlm_text_mask=None, aug_img=None, neg_text=None, neg_text_mask=None, neg_misc=None):
        self.text = text
        self.text_mask = text_mask
        self.img = img
        self.misc = misc

        self.mlm_text = mlm_text
        self.mlm_target = mlm_target
        self.mlm_text_mask = mlm_text_mask
        self.aug_img = aug_img
        
        self.neg_text = neg_text
        self.neg_text_mask = neg_text_mask
        self.neg_misc = neg_misc

    def to(self, device):
        self.text = self.text.to(device)
        self.text_mask = self.text_mask.to(device)
        self.img = self.img.to(device)

        if self.mlm_text is not None:
            self.mlm_text = self.mlm_text.to(device)
        if self.mlm_target is not None:
            self.mlm_target = self.mlm_target.to(device)
        if self.mlm_text_mask is not None:
            self.mlm_text_mask = self.mlm_text_mask.to(device)
        if self.aug_img is not None:
            self.aug_img = self.aug_img.to(device)
        
        if self.neg_misc is not None:
            self.neg_misc = self.neg_misc.to(device)

        return self

class MMBatch(object):
    def __init__(self, text, text_mask, img, misc_list, mlm_text=None, mlm_target=None, mlm_text_mask=None, aug_img=None, neg_text=None, neg_text_mask=None, neg_misc=None):
        self.text = text
        self.text_mask = text_mask
        self.img = img
        self.misc_list = misc_list

        self.mlm_text = mlm_text
        self.mlm_target = mlm_target
        self.mlm_text_mask = mlm_text_mask
        self.aug_img = aug_img

        self.neg_text = neg_text
        self.neg_text_mask = neg_text_mask
        self.neg_misc = neg_misc

    def to(self, device):
        self.text = self.text.to(device)
        self.text_mask = self.text_mask.to(device)
        self.img = self.img.to(device)

        if self.mlm_text is not None:
            self.mlm_text = self.mlm_text.to(device)
        if self.mlm_target is not None:
            self.mlm_target = self.mlm_target.to(device)
        if self.mlm_text_mask is not None:
            self.mlm_text_mask = self.mlm_text_mask.to(device)
        if self.aug_img is not None:
            self.aug_img = self.aug_img.to(device)

        if self.neg_text is not None:
            self.neg_text = self.neg_text.to(device)
        if self.neg_text_mask is not None:
            self.neg_text_mask = self.neg_text_mask.to(device)
        if self.neg_misc is not None:
            self.neg_misc = self.neg_misc.to(device)
        return self

def collate_fn(input_list):
    raise NotImplementedError
    
def collate_fn40_worker(input_list, max_text_len):
    for item in input_list:
        assert len(item.text) <= max_text_len, 'len(item.text): {}'.format(len(item.text))
    if input_list[0].neg_text is not None:
        for item in input_list:
            for each_neg_text_str in item.neg_text:
                assert len(each_neg_text_str) <= max_text_len, 'len(item.neg_text): {}'.format(len(item.neg_text))

    text_code_tensor = input_list[0].text.new_full((len(input_list), max_text_len), 0)
    text_mask_tensor = input_list[0].text_mask.new_full((len(input_list), max_text_len), 0)
    img_tensor = input_list[0].img.new_full((len(input_list), ) + input_list[0].img.shape, 0)
    misc_list = []
    if input_list[0].mlm_text is not None:
        mlm_text_code_tensor = input_list[0].mlm_text.new_full((len(input_list), max_text_len), 0)
    else:
        mlm_text_code_tensor = None
    if input_list[0].mlm_target is not None:
        mlm_target_tensor = input_list[0].mlm_target.new_full((len(input_list), max_text_len), -100)
    else:
        mlm_target_tensor = None
    if input_list[0].mlm_text_mask is not None:
        mlm_text_mask_tensor = input_list[0].mlm_text_mask.new_full((len(input_list), max_text_len), 0)
    else:
        mlm_text_mask_tensor = None
    if input_list[0].aug_img is not None:
        aug_img_tensor = input_list[0].aug_img.new_full((len(input_list), ) + input_list[0].aug_img.shape, 0)
    else:
        aug_img_tensor = None
    if input_list[0].neg_text is not None:
        neg_text_code_tensor = input_list[0].neg_text[0].new_full((len(input_list), len(input_list[0].neg_text), max_text_len), 0)
    else:
        neg_text_code_tensor = None
    if input_list[0].neg_text_mask is not None:
        neg_text_mask_tensor = input_list[0].neg_text_mask[0].new_full((len(input_list), len(input_list[0].neg_text_mask), max_text_len), 0)
    else:
        neg_text_mask_tensor = None
    if input_list[0].neg_misc is not None:
        neg_misc_tensor = input_list[0].neg_misc.new_full((len(input_list), input_list[0].neg_misc.shape[0]), -1)
    else:
        neg_misc_tensor = None

    for idx, item in enumerate(input_list):
        text, text_mask, img, misc, mlm_text, mlm_target, mlm_text_mask, aug_img, neg_text, neg_text_mask, neg_misc = item.text, item.text_mask, \
            item.img, item.misc, item.mlm_text, item.mlm_target, item.mlm_text_mask, item.aug_img, item.neg_text, item.neg_text_mask, item.neg_misc
        text_code_tensor[idx, 0:len(text)] = text
        text_mask_tensor[idx, 0:len(text_mask)] = text_mask
        img_tensor[idx] = img
        misc_list.append(misc)

        if mlm_text_code_tensor is not None:
            mlm_text_code_tensor[idx, 0:len(mlm_text)] = mlm_text
        if mlm_target_tensor is not None:
            mlm_target_tensor[idx, 0:len(mlm_target)] = mlm_target
        if mlm_text_mask_tensor is not None:
            mlm_text_mask_tensor[idx, 0:len(mlm_text_mask)] = mlm_text_mask
        if aug_img_tensor is not None:
            aug_img_tensor[idx] = aug_img

        if neg_text_code_tensor is not None:
            for neg_idx, each_text in enumerate(neg_text):
                neg_text_code_tensor[idx, neg_idx, 0:len(each_text)] = each_text
        if neg_text_mask_tensor is not None:
            for neg_idx, each_text_mask in enumerate(neg_text_mask):
                neg_text_mask_tensor[idx, neg_idx, 0:len(each_text_mask)] = each_text_mask
        if neg_misc is not None:
            neg_misc_tensor[idx, :] = neg_misc

    return MMBatch(text_code_tensor, text_mask_tensor, img_tensor, misc_list, mlm_text_code_tensor, mlm_target_tensor, 
                mlm_text_mask_tensor, aug_img_tensor, neg_text_code_tensor, neg_text_mask_tensor, neg_misc_tensor)

def collate_fn77(input_list):
    return collate_fn40_worker(input_list, 77)

def collate_fn77_cn(input_batch_list):
    input_list = []
    for input_batch in input_batch_list:
        input_list.extend(input_batch)
    return collate_fn77(input_list)

def openai_text_processor(text, max_words):
    # max_words is not used

    # basic_clean
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()

    # whitespace_clean
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
