import tempfile
import os
from transformers import BertForMaskedLM, BertModel, BertConfig
import torch
import torch.nn as nn

from utils.oss_op import get_bucket, OssProxy


class Pooling(nn.Module):
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        if pooling_mode is not None:        #Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)

class BertWrapper(nn.Module):
    def __init__(self, projector='linear', use_gradient_ckpt=False, use_cls_token=False, language='en', feat_dim=512, token_dim=-1):
        super(BertWrapper, self).__init__()
        if language == 'en':
            if os.path.exists('pretrained/bert_base_uncased') and (not use_gradient_ckpt):
                self.bert = BertForMaskedLM.from_pretrained('pretrained/bert_base_uncased').bert
            else:
                raise FileNotFoundError
        else:
            raise ValueError()

        self.pooling = Pooling(768)
        self.use_cls_token = use_cls_token
        
        if projector == 'linear':
            self.projector = nn.Linear(768, feat_dim, bias=False)
        elif projector == 'linear_bias':
            self.projector = nn.Linear(768, feat_dim)
        elif projector == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(768, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, feat_dim, bias=False))
        elif projector == 'iden':
            self.projector = nn.Identity()
        else:
            raise ValueError
        
        if token_dim > 0:
            self.projector_token_embeds = nn.Linear(768, token_dim)
        else:
            self.projector_token_embeds = nn.Identity()


    def forward(self, input_ids, attention_mask, output_hidden_states=False, use_linear=True, return_tensor=False):
        trans_features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        output_states = self.bert(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        cls_tokens = output_tokens[:, 0, :]  # CLS token is first token
        features.update({
            'token_embeddings': output_tokens,
            'cls_token_embeddings': cls_tokens,
            'attention_mask': features['attention_mask']
        })

        if output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3:  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        features = self.pooling(features)
        
        token_name = 'cls_token_embeddings' if self.use_cls_token else 'sentence_embedding'
        if use_linear:
            if return_tensor:
                return self.projector(features[token_name]), self.projector_token_embeds(features['token_embeddings'])
            else:
                return self.projector(features[token_name])
        else:
            if return_tensor:
                return features[token_name], self.projector_token_embeds(features['token_embeddings'])
            else:
                return features[token_name]
