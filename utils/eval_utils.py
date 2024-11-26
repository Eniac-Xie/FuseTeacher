import time
import torch
import torch.nn.functional as F
import numpy as np

class CustomToken():
    def __init__(self, ids, attention_mask):
        self.ids = ids
        self.attention_mask = attention_mask

def padding_tokens(tokens_list):
    max_text_len = 0
    for tokens in tokens_list:
        if len(tokens.ids) > max_text_len:
            max_text_len = len(tokens.ids)
    if max_text_len <= 50:
        max_text_len = 50
    else:
        raise ValueError

    text_code_tensor = torch.zeros((len(tokens_list), max_text_len)).long()
    text_mask_tensor = torch.zeros((len(tokens_list), max_text_len))

    for idx, tokens in enumerate(tokens_list):
        text, text_mask = tokens.ids, tokens.attention_mask
        text_code_tensor[idx, 0:len(text)] = torch.tensor(text)
        text_mask_tensor[idx, 0:len(text_mask)] = torch.tensor(text_mask)

    return text_code_tensor, text_mask_tensor

def padding_tokens77(tokens_list):
    max_text_len = 0
    for tokens in tokens_list:
        if len(tokens.ids) > max_text_len:
            max_text_len = len(tokens.ids)
    if max_text_len <= 77:
        max_text_len = 77
    else:
        raise ValueError

    text_code_tensor = torch.zeros((len(tokens_list), max_text_len)).long()
    text_mask_tensor = torch.zeros((len(tokens_list), max_text_len))

    for idx, tokens in enumerate(tokens_list):
        text, text_mask = tokens.ids, tokens.attention_mask
        text_code_tensor[idx, 0:len(text)] = torch.tensor(text)
        text_mask_tensor[idx, 0:len(text_mask)] = torch.tensor(text_mask)

    return text_code_tensor, text_mask_tensor
    
def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()  # 5*N
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 5*N
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def evaluation_imagenet_zs(model, data_loader, tokenizer, device, two_stage=False, padding_tokens=padding_tokens, eval_top5=True):
    # test
    model.to(device)
    model.eval()  
    
    print('Computing features for evaluation...')

    cid2caption_list = data_loader.dataset.cid2caption_list
    text_embeds = []
    with torch.no_grad():
        for cid in range(len(cid2caption_list)):
            if cid % 10 == 0:
                print('extracting {}/{} cid feature'.format(cid, len(cid2caption_list)))
            tokens_list = []
            caption_list = cid2caption_list[cid]
            for caption in caption_list:
                tokens = tokenizer.encode(caption)
                tokens_list.append(tokens)

            text_ids, text_mask = padding_tokens(tokens_list)
            text_embed = model.text_model(text_ids.to(device), attention_mask=text_mask.to(device), return_tensor=False)
            text_embed = F.normalize(text_embed, p=2.0, dim=1)
            text_embed = torch.mean(text_embed, dim=0, keepdim=True)
            text_embed = F.normalize(text_embed, p=2.0, dim=1)
            text_embeds.append(text_embed)
        text_embeds = torch.cat(text_embeds, dim=0)

    cnt, top1, top5 = 0, 0, 0
    pred_list, gt_list = [], []
    with torch.no_grad():
        for idx, (image, cid) in enumerate(data_loader): 
            image = image.to(device) 
            _, _, image_embed, _ = model.get_feature(text_data=None, text_mask=None, img_tensor=image)

            batch_i2t = image_embed @ text_embeds.t()

            pred_list.extend(
                torch.max(batch_i2t, dim=1)[1].cpu().tolist()
            )
            gt_list.extend(
                cid.tolist()
            )

            if eval_top5:
                tp1_cnt, tp5_cnt = accuracy(batch_i2t.cpu(), cid, topk=(1, 5))
            else:
                tp1_cnt,  = accuracy(batch_i2t.cpu(), cid, topk=(1, ))
            cnt += image_embed.shape[0]
            top1 += tp1_cnt
            if eval_top5:
                top5 += tp5_cnt

            if eval_top5:
                print('extracting {}/{} image embedding, top1: {:.5f}, top5: {:.5f}'.format(idx, len(data_loader), top1 / cnt, top5 / cnt))
            else:
                print('extracting {}/{} image embedding, top1: {:.5f}'.format(idx, len(data_loader), top1 / cnt))
