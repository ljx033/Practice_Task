import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransE(nn.Module):
    """TransE: h + r ≈ t，score = -||h + r - t||_p"""

    def __init__(self, num_entities, num_relations, dim=200, p_norm=1, margin=1.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.p_norm = p_norm
        self.margin = margin

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        bound = 6.0 / math.sqrt(dim)
        nn.init.uniform_(self.entity_emb.weight, -bound, bound)
        nn.init.uniform_(self.relation_emb.weight, -bound, bound)
        # 关系向量归一化
        with torch.no_grad():
            self.relation_emb.weight.data = F.normalize(
                self.relation_emb.weight.data, p=2, dim=-1
            )

    def forward(self, h, r, t):
        h_emb = F.normalize(self.entity_emb(h), p=2, dim=-1)
        t_emb = F.normalize(self.entity_emb(t), p=2, dim=-1)
        r_emb = F.normalize(self.relation_emb(r), p=2, dim=-1)
        score = -torch.norm(h_emb + r_emb - t_emb, p=self.p_norm, dim=-1)
        return score

    def loss(self, pos_score, neg_score):
        """Margin ranking loss"""
        return F.margin_ranking_loss(
            pos_score, neg_score,
            torch.ones_like(pos_score),
            margin=self.margin
        )


class RotatE(nn.Module):
    """RotatE: 关系定义为复空间中的旋转，e_t = e_h ∘ e_r"""

    def __init__(self, num_entities, num_relations, dim=200, margin=12.0):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim  # 复数维度，实际嵌入为 2*dim
        self.margin = margin

        # 实体嵌入：实部 + 虚部，各 dim 维
        self.entity_emb_re = nn.Embedding(num_entities, dim)
        self.entity_emb_im = nn.Embedding(num_entities, dim)
        # 关系嵌入：角度 θ ∈ [0, 2π)
        self.relation_emb = nn.Embedding(num_relations, dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb_re.weight)
        nn.init.xavier_uniform_(self.entity_emb_im.weight)
        nn.init.uniform_(self.relation_emb.weight, 0, 2 * math.pi)

    def forward(self, h, r, t):
        h_re = self.entity_emb_re(h)
        h_im = self.entity_emb_im(h)
        t_re = self.entity_emb_re(t)
        t_im = self.entity_emb_im(t)
        r_theta = self.relation_emb(r)

        r_re = torch.cos(r_theta)
        r_im = torch.sin(r_theta)

        # 复数乘法：e_h ∘ e_r
        re_score = r_re * h_re - r_im * h_im - t_re
        im_score = r_re * h_im + r_im * h_re - t_im

        # 模长
        # score = torch.sqrt(re_score ** 2 + im_score ** 2 + 1e-10)
        score = torch.sqrt(re_score ** 2 + im_score ** 2 + 1e-10).sum(dim=-1)  #新加
        return -score

    def loss(self, pos_score, neg_score):
        return F.margin_ranking_loss(
            pos_score, neg_score,
            torch.ones_like(pos_score),
            margin=self.margin
        )


class ConvE(nn.Module):
    """ConvE: 2D 卷积 + 全连接打分"""

    def __init__(self, num_entities, num_relations, dim=200,
                 input_dropout=0.2, hidden_dropout=0.2,
                 feature_map_dropout=0.2, embedding_dropout=0.2,
                 label_smoothing=0.1):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.label_smoothing = label_smoothing

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        # 实体偏置
        self.entity_bias = nn.Embedding(num_entities, 1)

        # 卷积层
        self.inp_drop = nn.Dropout(input_dropout)
        self.hidden_drop = nn.Dropout(hidden_dropout)
        self.feature_map_drop = nn.Dropout2d(feature_map_dropout)

        # 将 h 和 r 堆叠成 2D 后尺寸
        self.emb_dim1 = int(math.sqrt(dim))  # 约 sqrt(200) ≈ 14
        self.emb_dim2 = dim // self.emb_dim1  # 200 // 14 = 14
        # 调整实际使用的维度
        self.dim = self.emb_dim1 * self.emb_dim2

        self.conv1 = nn.Conv2d(1, 32, (3, 3), 1, 0, bias=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(self.dim)

        # 全连接输出维度
        conv_out_h = self.emb_dim1 * 2 - 2  # 堆叠后的高度
        conv_out_w = self.emb_dim2 - 2
        fc_input_dim = 32 * conv_out_h * conv_out_w

        self.fc = nn.Linear(fc_input_dim, self.dim)
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(num_entities))
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.zeros_(self.entity_bias.weight)
        nn.init.zeros_(self.bias)

    def _get_combined(self, h, r):
        """将头实体和关系嵌入堆叠成 2D 图像"""
        e_h = self.entity_emb(h).view(-1, 1, self.emb_dim1, self.emb_dim2)
        e_r = self.relation_emb(r).view(-1, 1, self.emb_dim1, self.emb_dim2)
        return torch.cat([e_h, e_r], dim=2)  # (B, 1, 2*emb_dim1, emb_dim2)

    def forward(self, h, r, t=None):
        stacked = self._get_combined(h, r)
        stacked = self.inp_drop(stacked)
        stacked = self.bn0(stacked)

        x = self.conv1(stacked)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 与所有实体嵌入做内积
        all_entities = self.entity_emb.weight  # (E, D)
        all_entities = F.normalize(all_entities, p=2, dim=-1)
        x = F.normalize(x, p=2, dim=-1)
        score = torch.mm(x, all_entities.t()) + self.bias.unsqueeze(0)

        if t is not None:
            # 训练时返回正样本得分
            score = score.gather(1, t.unsqueeze(1)).squeeze(1)
        return score

    def loss(self, pos_score, neg_score):
        """ConvE 使用 BCE with label smoothing"""
        return F.binary_cross_entropy_with_logits(
            pos_score,
            torch.ones_like(pos_score) * (1 - self.label_smoothing),
            reduction='mean'
        )