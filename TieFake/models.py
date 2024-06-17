# import torch
# from torch import nn
# from torch.nn import Module
# import torch.nn.functional as F
# from transformers import BertForSequenceClassification
# import numpy as np
# from attention import MultiHeadAttention

# class FusionModel(Module):

#     def __init__(self, resnest_model, bert_model):
#         super(FusionModel, self).__init__()

#         assert isinstance(bert_model, BertForSequenceClassification)

#         self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         resnest_feature_size = resnest_model.fc.in_features
#         self._resnest = resnest_model
#         self._resnest.fc = nn.Linear(resnest_feature_size, 30)
#         for param in self._resnest.parameters():
#             param.requires_grad = False

#         self._bert = bert_model.bert
#         self._bert.eval()
#         for param in self._bert.parameters():
#             param.requires_grad = False
#         self._attention_T_to_Ti = MultiHeadAttention(768,768,32,1)
#         self._attention_Ti_to_T = MultiHeadAttention(768,768,32,1)
#         self.linear = nn.Linear(32+30+38+32+32, 1)

#     def forward(self, batch_in:dict):

#         batch_in = {x: batch_in[x].to(next(self.parameters()).device) for x in batch_in}
#         bert_output = self._bert(batch_in['bert_input_id_title'].squeeze(1), attention_mask=batch_in['bert_attention_mask_title'].squeeze(1))
#         title_vector = bert_output.pooler_output
#         bert_output = self._bert(batch_in['bert_input_id_title'].squeeze(1), attention_mask=batch_in['bert_attention_mask_title'].squeeze(1))
#         text_vector = bert_output.pooler_output
#         emo_feature = np.float32(batch_in['bert_input_emo'].cpu())
#         emo_feature = torch.from_numpy(emo_feature).squeeze(1)
#         emo_feature=emo_feature.to(self.device)
#         resnest_feature = self._resnest(batch_in['image'])
#         # print(text_vector.shape)
#         # print(title_vector.shape)
#         # 改变矩阵形状
#         # text_vector = text_vector.reshape(16, 24, 32)
#         # 假设 text_vector 和 title_vector 的形状为 (batch_size, 768)
#         # text_vector = text_vector.view(-1, 768)  # 修改形状为 (batch_size, 768)
#         # title_vector = title_vector.view(-1, 768)  # 修改形状为 (batch_size, 768)
#         att_T_Ti,_=self._attention_T_to_Ti(text_vector,title_vector)
#         att_Ti_T,_=self._attention_Ti_to_T(title_vector,text_vector)
#         return self.linear(torch.cat((text_vector,emo_feature,resnest_feature,att_T_Ti,att_Ti_T), dim=1))

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from transformers import BertForSequenceClassification
import numpy as np
from attention import MultiHeadAttention

class FusionModel(Module):

    def __init__(self, resnest_model, bert_model, device):
        super(FusionModel, self).__init__()

        assert isinstance(bert_model, BertForSequenceClassification)

        self.device = device

        resnest_feature_size = resnest_model.fc.in_features
        self._resnest = resnest_model
        self._resnest.fc = nn.Linear(resnest_feature_size, 30)
        for param in self._resnest.parameters():
            param.requires_grad = False

        self._bert = bert_model.bert
        self._bert.eval()
        for param in self._bert.parameters():
            param.requires_grad = False
        self._attention_T_to_Ti = MultiHeadAttention(768, 768, 32, 1)
        self._attention_Ti_to_T = MultiHeadAttention(768, 768, 32, 1)
        # self.linear = nn.Linear(32+30+38+32+32, 1)
        self.linear = nn.Linear(900, 1)

    def forward(self, batch_in: dict):
        batch_in = {x: batch_in[x].to(next(self.parameters()).device) for x in batch_in}
        bert_output = self._bert(batch_in['bert_input_id_title'].squeeze(1), attention_mask=batch_in['bert_attention_mask_title'].squeeze(1))
        title_vector = bert_output.pooler_output
        bert_output = self._bert(batch_in['bert_input_id_title'].squeeze(1), attention_mask=batch_in['bert_attention_mask_title'].squeeze(1))
        text_vector = bert_output.pooler_output
        emo_feature = np.float32(batch_in['bert_input_emo'].cpu())
        emo_feature = torch.from_numpy(emo_feature).squeeze(1)
        emo_feature = emo_feature.to(self.device)
        resnest_feature = self._resnest(batch_in['image'])
        
        # 打印各个张量的形状
        # print(f"text_vector shape: {text_vector.shape}")
        # print(f"title_vector shape: {title_vector.shape}")
        # print(f"emo_feature shape: {emo_feature.shape}")
        # print(f"resnest_feature shape: {resnest_feature.shape}")
        
        att_T_Ti, _ = self._attention_T_to_Ti(text_vector, title_vector)
        att_Ti_T, _ = self._attention_Ti_to_T(title_vector, text_vector)
        
        # 打印注意力输出的形状
        # print(f"att_T_Ti shape: {att_T_Ti.shape}")
        # print(f"att_Ti_T shape: {att_Ti_T.shape}")
        
        # 调整注意力输出的形状
        att_T_Ti = att_T_Ti.squeeze(1)
        att_Ti_T = att_Ti_T.squeeze(1)
        
        # 打印调整后的注意力输出的形状
        # print(f"adjusted att_T_Ti shape: {att_T_Ti.shape}")
        # print(f"adjusted att_Ti_T shape: {att_Ti_T.shape}")
        
        return self.linear(torch.cat((text_vector, emo_feature, resnest_feature, att_T_Ti, att_Ti_T), dim=1))

