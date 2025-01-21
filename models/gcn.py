import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义多头交叉注意力机制
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads."

        # 定义多头的线性变换
        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        # 输出线性变换
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, protein_features, dynamic_features):
        batch_size = protein_features.size(0)

        # 使用蛋白质特征作为查询
        queries = self.query_linear(protein_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 使用动态特征作为键和值
        keys = self.key_linear(dynamic_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_linear(dynamic_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 生成蛋白质特征的加权输出
        protein_output = torch.matmul(attention_weights, values).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 使用动态特征作为查询，蛋白质特征作为键和值
        dynamic_queries = self.query_linear(dynamic_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        dynamic_keys = self.key_linear(protein_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        dynamic_values = self.value_linear(protein_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算动态特征的注意力得分
        dynamic_attention_scores = torch.matmul(dynamic_queries, dynamic_keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        dynamic_attention_weights = F.softmax(dynamic_attention_scores, dim=-1)

        # 生成动态特征的加权输出
        dynamic_output = torch.matmul(dynamic_attention_weights, dynamic_values).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        # 使用线性变换合并多头的输出
        protein_output = self.out_proj(protein_output)
        dynamic_output = self.out_proj(dynamic_output)

        return protein_output, dynamic_output  # 返回两个与输入维度相同的加权向量


# 定义MLP模型  三层 MLP
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TFN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        :param input_dim: 每个模态输入特征的维度，例如 128。
        :param output_dim: 最终融合特征的输出维度，例如 128。
        """
        super(TFN, self).__init__()

        # 输出特征的线性压缩
        self.fusion_fc = nn.Linear((input_dim + 1) ** 3, output_dim)  # AxBxC 包含偏置扩展后的高阶张量

    def forward(self, x, protein_output, dynamic_output):
        """
        :param x: 第一个模态输入 (例如来自 GCN)，形状 [batch_size, input_dim]。
        :param protein_output: 第二个模态输入 (蛋白质特征)，形状 [batch_size, input_dim]。
        :param dynamic_output: 第三个模态输入 (动态特征)，形状 [batch_size, input_dim]。
        :return: 融合后的特征，形状 [batch_size, output_dim]。
        """
        batch_size = x.size(0)

        # 用 1 扩充偏置维度
        x = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)  # [batch_size, input_dim + 1]
        protein_output = torch.cat([protein_output, torch.ones(batch_size, 1, device=protein_output.device)], dim=1)  # [batch_size, input_dim + 1]
        dynamic_output = torch.cat([dynamic_output, torch.ones(batch_size, 1, device=dynamic_output.device)], dim=1)  # [batch_size, input_dim + 1]

        # 计算笛卡尔积
        A = x.unsqueeze(2)  # [batch_size, Ax, 1]
        B = protein_output.unsqueeze(1)  # [batch_size, 1, Bx]
        fusion_AB = torch.einsum('nxt,nty->nxy', A, B)  # [batch_size, Ax, Bx]

        fusion_AB = fusion_AB.flatten(start_dim=1).unsqueeze(1)  # [batch_size, Ax*Bx, 1]
        C = dynamic_output.unsqueeze(1)  # [batch_size, 1, Cx]
        fusion_ABC = torch.einsum('ntx,nty->nxy', fusion_AB, C)  # [batch_size, Ax*Bx, Cx]

        # 展平为 [batch_size, (Ax*Bx*Cx)]
        fusion_ABC = fusion_ABC.flatten(start_dim=1)  # [batch_size, AxBxCx]

        # 压缩高阶融合特征
        fusion_output = self.fusion_fc(fusion_ABC)  # [batch_size, output_dim]

        return fusion_output

# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64,num_features_xd=78,
                 num_features_xt=25, output_dim=64, dropout=0.2):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        # self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8, dilation=4)
        self.fc1_xt = nn.Linear(1152, output_dim)

        # cross attention layer
        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads = 4)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # 特征融合MLP
        self.mlp = MLP(input_dim=4, output_dim=64)  # 假设有四个特征需要融合
        self.tfn = TFN(input_dim = 64, output_dim= 256)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target
        physical_features = data.dynamic_features  # 假设这里包含"Avg. RMSF", "Avg. gyr.", "Div. SE", "Div. MM"
        physical_features = physical_features.view(-1, 4)
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # print(conv_xt.shape)
        # flatten
        xt = conv_xt.view(-1, 1152)
        xt = self.fc1_xt(xt)

        # 特征归一化并通过MLP处理
        normalized_features = self.mlp(physical_features.float())

        protein_output, dynamic_output = self.cross_attention(xt, normalized_features)

        protein_output = protein_output.squeeze(1)  # 变为 [512, 64]
        dynamic_output = dynamic_output.squeeze(1)  # 变为 [512, 64]

        # 拼接GCN输出、交叉注意力输出和药物特征
        # xc = torch.cat((x, protein_output, dynamic_output), 1)
        xc = self.tfn(x, protein_output, dynamic_output)

        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
