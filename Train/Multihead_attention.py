from math import sqrt
import torch
import torch.nn as nn


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e4)
        attention_weights = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        final_attention = torch.matmul(attention_weights, V)
        return final_attention


class Multi_CrossAttention(nn.Module):

    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.calculate_attention = CalculateAttention()
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, y, attention_mask):

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention_mask = attention_mask.eq(0).unsqueeze(1).unsqueeze(1)

        attention = self.calculate_attention(q_s, k_s, v_s, attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output


class Multi_SelfAttention(nn.Module):

    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.calculate_attention = CalculateAttention()
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, attention_mask):
        batch_size = x.size(0)

        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        k_s = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        v_s = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention_mask = attention_mask.eq(0).unsqueeze(1).unsqueeze(1)

        attention = self.calculate_attention(q_s, k_s, v_s, attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output


class Multi_Symmetric_CrossAttention(nn.Module):

    def __init__(self, hidden_size, all_head_size, head_num, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        assert all_head_size % head_num == 0
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.calculate_attention = CalculateAttention()
        self.linear_output = nn.Linear(all_head_size, hidden_size)
        # self.linear_output_2 = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def forward(self, x_o, y_o, attention_mask_1, attention_mask_2):
        x = self.layer_norm_1(x_o)
        y = self.layer_norm_2(y_o)

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # x to y:
        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s_1 = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s_1 = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s_1 = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # y to x:
        q_s_2 = self.linear_q(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s_2 = self.linear_k(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s_2 = self.linear_v(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention_mask_1 = attention_mask_1.eq(0).unsqueeze(1).unsqueeze(1)
        attention_mask_2 = attention_mask_2.eq(0).unsqueeze(1).unsqueeze(1)
        attention_1 = self.calculate_attention(q_s_1, k_s_1, v_s_1, attention_mask_2)
        attention_2 = self.calculate_attention(q_s_2, k_s_2, v_s_2, attention_mask_1)

        # attention : [batch_size , seq_length , num_heads * h_size]
        attention_1 = attention_1.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        attention_2 = attention_2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output_1 = self.linear_output(attention_1)
        output_2 = self.linear_output(attention_2)

        output_1 = self.dropout1(output_1)
        output_2 = self.dropout2(output_2)

        output_1 = output_1 + x_o
        output_2 = output_2 + y_o

        # output_cat : [batch_size , seq_length_x + seq_length_y, hidden_size]
        concat = torch.concat([output_1, output_2], dim=1)
        reversed_concat = torch.cat([output_2, output_1], dim=1)  # 反向拼接
        sym_concat = concat + reversed_concat  # 对称化
        return sym_concat


class Multi_Asymmetric_CrossAttention(nn.Module):

    def __init__(self, hidden_size, all_head_size, head_num, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        assert all_head_size % head_num == 0
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q_1 = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k_1 = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v_1 = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_q_2 = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k_2 = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v_2 = nn.Linear(hidden_size, all_head_size, bias=False)
        self.calculate_attention = CalculateAttention()
        self.linear_output_1 = nn.Linear(all_head_size, hidden_size)
        self.linear_output_2 = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def forward(self, x_o, y_o, attention_mask_1, attention_mask_2):
        x = self.layer_norm_1(x_o)
        y = self.layer_norm_2(y_o)

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # x to y:
        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s_1 = self.linear_q_1(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s_1 = self.linear_k_1(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s_1 = self.linear_v_1(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # y to x:
        q_s_2 = self.linear_q_2(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s_2 = self.linear_k_2(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s_2 = self.linear_v_2(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention_mask_1 = attention_mask_1.eq(0).unsqueeze(1).unsqueeze(1)
        attention_mask_2 = attention_mask_2.eq(0).unsqueeze(1).unsqueeze(1)
        attention_1 = self.calculate_attention(q_s_1, k_s_1, v_s_1, attention_mask_2)
        attention_2 = self.calculate_attention(q_s_2, k_s_2, v_s_2, attention_mask_1)

        # attention : [batch_size , seq_length , num_heads * h_size]
        attention_1 = attention_1.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        attention_2 = attention_2.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output_1 = self.linear_output_1(attention_1)
        output_2 = self.linear_output_2(attention_2)

        output_1 = self.dropout1(output_1)
        output_2 = self.dropout2(output_2)

        output_1 = output_1 + x_o
        output_2 = output_2 + y_o

        # output_cat : [batch_size , seq_length_x + seq_length_y, hidden_size]
        concat = torch.concat([output_1, output_2], dim=1)

        return concat
