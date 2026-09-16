from torch.utils.data import Dataset
import torch
import numpy as np
import os


class SeismicDataset1(Dataset):
    def __init__(self, data_dir, catalogue, stations, lookup, N_sub, N_t):
        self.data_dir = data_dir
        self.cat = catalogue
        self.stations = stations
        self.lookup = lookup
        self.N_events = catalogue.shape[0]
        self.N_t = N_t
        self.N_sub = N_sub
        self.event_inds = np.arange(self.N_events).astype(int)
        self._generate_data()

    def __len__(self):
        return self.N_events

    def __getitem__(self, idx):
        waveforms = self.waveforms[idx]
        coords = self.station_coords[idx]
        weights = self.weights[idx]
        labels = self.labels[idx]
        loss_weight = self.loss_weights[idx]

        return (torch.tensor(waveforms, dtype=torch.float32),
                torch.tensor(coords, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32)), \
            torch.tensor(labels, dtype=torch.float32), \
            loss_weight  # 可视需要保留或忽略

    def _generate_data(self):
        N_events, N_sub, N_t = self.N_events, self.N_sub, self.N_t
        catalogue = self.cat
        stations = self.stations
        lookup = self.lookup

        np.random.shuffle(self.event_inds)

        waveforms = np.zeros((N_events, N_sub, N_t, 3))
        station_coords = np.zeros((N_events, N_sub, 1, 3))
        weights = np.ones((N_events, N_sub))
        labels = np.zeros((N_events, 4))
        loss_weights = np.zeros(N_events)

        for i in range(N_events):
            event = catalogue[self.event_inds[i]]
            event_id = int(event[0])
            station_codes = lookup[event_id]
            N_codes = len(station_codes)
            station_inds = np.arange(N_codes)

            event_data = np.load(os.path.join(self.data_dir, f"{event_id}.npy"))

            start = 820

            if N_codes >= N_sub:
                selection = np.random.choice(station_inds, size=N_sub, replace=False)
                coords = stations[["lat", "lon"]][stations["code"].isin(station_codes)].values[selection]
                waveforms[i] = event_data[selection][:, start:start + self.N_t, :]
                station_coords[i, :, 0, :2] = coords
            else:
                waveforms[i, :N_codes] = event_data[:, start:start + self.N_t, :]
                coords = stations[["lat", "lon"]][stations["code"].isin(station_codes)].values
                station_coords[i, :N_codes, 0, :2] = coords
                weights[i, N_codes:] = 0

            labels[i] = event[2:]
            loss_weights[i] = event[1]

        self.waveforms = waveforms
        self.station_coords = station_coords
        self.weights = weights
        self.labels = labels
        self.loss_weights = loss_weights


import random
import numpy as np
import torch
from torch.utils.data import Dataset


class SeismicDataset2(Dataset):
    def __init__(self, data_dir, catalogue, stations, lookup,
                 N_sub, N_t, N_t_raw=4096, augment_times=3):
        self.data_dir = data_dir
        self.cat = catalogue
        self.stations = stations
        self.lookup = lookup
        self.N_events = catalogue.shape[0]
        self.N_t = N_t  # 最终想要的段长 2048
        self.N_t_raw = N_t_raw  # 原始读取的段长 4096
        self.N_sub = N_sub
        self.augment_times = augment_times  # 每个样本增强次数
        self.event_inds = np.arange(self.N_events)
        self._generate_data()  # waveforms 存储 (N_events, N_sub, N_t_raw, 3)

    def __len__(self):
        return len(self.enhanced_waveforms)  # 增强后的样本数量

    def __getitem__(self, idx):
        # 获取增强后的样本
        waveforms = self.enhanced_waveforms[idx]
        coords = self.enhanced_coords[idx]
        weights = self.enhanced_weights[idx]
        labels = self.enhanced_labels[idx]
        loss_weight = self.enhanced_loss_weights[idx]

        # 转换为 torch Tensor
        return (
            torch.tensor(waveforms, dtype=torch.float32),
            torch.tensor(coords, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32)
        ), torch.tensor(labels, dtype=torch.float32), loss_weight

    def _generate_data(self):
        N_events, N_sub, N_t = self.N_events, self.N_sub, self.N_t
        catalogue = self.cat
        stations = self.stations
        lookup = self.lookup

        np.random.shuffle(self.event_inds)

        self.waveforms = np.zeros((N_events, N_sub, N_t, 3))
        self.station_coords = np.zeros((N_events, N_sub, 1, 3))
        self.weights = np.ones((N_events, N_sub))
        self.labels = np.zeros((N_events, 4))
        self.loss_weights = np.zeros(N_events)

        # 用于存储增强后的数据
        augmented_waveforms = []
        augmented_coords = []
        augmented_weights = []
        augmented_labels = []
        augmented_loss_weights = []

        for i in range(N_events):
            event = catalogue[self.event_inds[i]]
            event_id = int(event[0])
            station_codes = lookup[event_id]
            N_codes = len(station_codes)
            station_inds = np.arange(N_codes)

            event_data = np.load(os.path.join(self.data_dir, f"{event_id}.npy"))

            start = random.randint(600, 1000)

            if N_codes >= N_sub:
                selection = np.random.choice(station_inds, size=N_sub, replace=False)
                coords = stations[["lat", "lon"]][stations["code"].isin(station_codes)].values[selection]
                self.waveforms[i] = event_data[selection][:, start:start + self.N_t, :]
                self.station_coords[i, :, 0, :2] = coords
            else:
                self.waveforms[i, :N_codes] = event_data[:, start:start + self.N_t, :]
                coords = stations[["lat", "lon"]][stations["code"].isin(station_codes)].values
                self.station_coords[i, :N_codes, 0, :2] = coords
                self.weights[i, N_codes:] = 0

            self.labels[i] = event[2:]
            self.loss_weights[i] = event[1]

            # 对每个原始样本进行数据增强
            for _ in range(self.augment_times):
                waveforms = self.waveforms[i].copy()
                coords = self.station_coords[i].copy()
                weights = self.weights[i].copy()
                labels = self.labels[i].copy()
                loss_weight = self.loss_weights[i]

                # 执行增强
                self._augment_data(waveforms, coords, weights, labels)

                # 保存增强后的数据
                augmented_waveforms.append(waveforms)
                augmented_coords.append(coords)
                augmented_weights.append(weights)
                augmented_labels.append(labels)
                augmented_loss_weights.append(loss_weight)

        # 将增强后的数据保存
        self.enhanced_waveforms = np.array(augmented_waveforms)
        self.enhanced_coords = np.array(augmented_coords)
        self.enhanced_weights = np.array(augmented_weights)
        self.enhanced_labels = np.array(augmented_labels)
        self.enhanced_loss_weights = np.array(augmented_loss_weights)

    def _augment_data(self, waveforms, coords, weights, labels):
        """执行所有的数据增强操作"""

        # 随机选择并执行增强
        aug_func = random.choice([self.aug_rotation, self.aug_translation, self.aug_drop_weights])
        aug_func(waveforms, coords, weights, labels)

    def aug_rotation(self, waveforms, coords, weights, labels):
        """旋转"""
        k = random.choice([1, 2, 3])

        def rot_xy(x, y, k):
            if k == 1: return y, -x
            if k == 2: return -x, -y
            if k == 3: return -y, x

        lat = coords[:, 0, 0].copy()
        lon = coords[:, 0, 1].copy()
        coords[:, 0, 0], coords[:, 0, 1] = rot_xy(lat, lon, k)
        labels[0], labels[1] = rot_xy(labels[0], labels[1], k)
        wf0 = waveforms.copy()
        if k == 1:
            waveforms[:, :, 0] = -wf0[:, :, 1]
            waveforms[:, :, 1] = wf0[:, :, 0]
        elif k == 2:
            waveforms[:, :, 0] = -wf0[:, :, 0]
            waveforms[:, :, 1] = -wf0[:, :, 1]
        elif k == 3:
            waveforms[:, :, 0] = wf0[:, :, 1]
            waveforms[:, :, 1] = -wf0[:, :, 0]

    def aug_translation(self, waveforms, coords, weights, labels):
        """平移"""
        dx = random.uniform(-0.1, 0.1)
        dy = random.uniform(-0.1, 0.1)
        coords[:, 0, 0] += dx
        coords[:, 0, 1] += dy
        labels[0] += dx
        labels[1] += dy
        coords[:, 0, :2] = np.clip(coords[:, 0, :2], -1.0, 1.0)
        labels[:2] = np.clip(labels[:2], -1.0, 1.0)

    def aug_drop_weights(self, waveforms, coords, weights, labels):
        """丢弃台站权重"""
        idxs = np.where(weights == 1)[0]
        n = len(idxs)
        if n > 8:
            drop = random.sample(idxs.tolist(), 2)
        elif 6 <= n <= 8:
            drop = random.sample(idxs.tolist(), 1)
        else:
            drop = []
        for d in drop:
            weights[d] = 0



import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDropout(nn.Module):
    def __init__(self, p: float):
        super(GaussianDropout, self).__init__()
        self.p = p  # dropout rate

    def forward(self, x):
        if self.training:  # 只有在训练模式下应用
            noise = torch.randn_like(x) * self.p  # 生成标准正态分布噪声
            return x + noise  # 将噪声加到输入数据上
        else:
            return x  # 在评估模式下不做任何修改


class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.15):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim  # 每个头的特征维度
        self.total_hidden_dim = hidden_dim * num_heads

        self.Wq = nn.Linear(in_dim, self.total_hidden_dim, bias=False)
        self.Wk = nn.Linear(in_dim, self.total_hidden_dim, bias=False)
        self.Wv = nn.Linear(in_dim, self.total_hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(self.total_hidden_dim, in_dim)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.act = nn.ReLU()

    def forward(self, x, mask=None):
        # x: (B, N_sub, C)
        B, N, C = x.size()

        Q = self.Wq(x).view(B, N, self.num_heads, self.hidden_dim)  # (B, N, H, d)
        K = self.Wk(x).view(B, N, self.num_heads, self.hidden_dim)
        V = self.Wv(x).view(B, N, self.num_heads, self.hidden_dim)

        # 计算注意力权重
        scores = torch.einsum('bnhd,bmhd->bhnm', Q, K) / (self.hidden_dim ** 0.5)  # (B, H, N, N)

        if mask is not None:
            # mask shape (B, N), 将无效节点设为极小值
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,N)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, H, N, N)
        attn = self.dropout(attn)

        out = torch.einsum('bhnm,bmhd->bnhd', attn, V)  # (B, N, H, d)
        out = out.contiguous().view(B, N, self.total_hidden_dim)  # (B, N, H*d)

        out = self.out_proj(out)  # (B, N, C)
        out = self.layer_norm(out + x)  # 残差连接 + LayerNorm
        out = self.act(out)
        return out


class AttentionPooling_Avg(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling_Avg, self).__init__()
        # 可训练的 attention 权重
        self.attention_weight = nn.Parameter(torch.randn(input_dim, 1))  # [input_dim, 1]

    def forward(self, x, weights):
        """
        :param x: 输入特征张量，形状为 [batch_size, num_nodes, feature_dim]
        :param weights: 节点有效性掩码，形状为 [batch_size, num_nodes]
        :return: 图级表示，形状为 [batch_size, feature_dim]
        """
        # 计算每个节点的注意力分数
        attention_scores = torch.matmul(x, self.attention_weight)  # [batch_size, num_nodes, 1]
        # print(f"attention_scores:{attention_scores}")
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, num_nodes]

        # 将无效节点的注意力分数设置为负无穷，遮蔽掉无效节点
        attention_scores = attention_scores.masked_fill(weights == 0, float('-inf'))  # [batch_size, num_nodes]

        # 使用 softmax 归一化注意力分数，仅对有效节点进行处理
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_nodes]

        # 加权平均池化
        weighted_x = x * attention_weights.unsqueeze(-1)  # [batch_size, num_nodes, feature_dim]
        graph_representation = weighted_x.sum(dim=1)  # [batch_size, feature_dim]

        return graph_representation


class AttentionPooling_Max(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling_Max, self).__init__()
        # 可训练的 attention 权重
        self.attention_weight = nn.Parameter(torch.randn(input_dim, 1))  # [input_dim, 1]

    def forward(self, x, weights):
        """
        :param x: 输入特征张量，形状为 [batch_size, num_nodes, feature_dim]
        :param weights: 节点有效性掩码，形状为 [batch_size, num_nodes]
        :return: 图级表示，形状为 [batch_size, feature_dim]
        """
        # 计算每个节点的注意力分数
        attention_scores = torch.matmul(x, self.attention_weight)  # [batch_size, num_nodes, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, num_nodes]

        # 将无效节点的注意力分数设置为负无穷，遮蔽掉无效节点
        attention_scores = attention_scores.masked_fill(weights == 0, float('-inf'))  # [batch_size, num_nodes]

        # 使用 softmax 归一化注意力分数，仅对有效节点进行处理
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_nodes]

        # 对特征进行加权，并计算每个特征维度的最大值
        weighted_x = x * attention_weights.unsqueeze(-1)  # [batch_size, num_nodes, feature_dim]

        # 对每个特征维度，计算加权后的最大值
        graph_representation = weighted_x.max(dim=1)[0]  # [batch_size, feature_dim], 取最大值

        return graph_representation

class GraphNet(nn.Module):
    def __init__(self, N_sub=50, N_t=2048, f0=4, dropout_rate=0.15, activation="tanh", num_heads=8,struct_dim=64,
                 struct_nhead=4,
                 num_layers=2, theta = 1):
        super().__init__()

        self.N_sub = N_sub
        self.N_t = N_t
        self.f0 = f0
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.num_heads = num_heads

        self.kernel_size = (1, 5)
        self.use_bn = False
        self.use_dropout = True
        self.dropout_at_test = True

        self.feature_extractor = self._build_cnn_block()
        self.feature_merger = self._build_feature_merger()

        # 多头图注意力层，输入128维，输出128维（每个头维度32，4头）
        self.multihead_attention = MultiHeadGraphAttention(128, 64, num_heads=self.num_heads, dropout=dropout_rate)

        self.predictor = self._build_predictor()
        self.GaussianDropout = GaussianDropout(p = self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)

        # 图级聚合
        self.attention_pooling = AttentionPooling_Avg(128)
        # self.attention_pooling = AttentionPooling_Max(128)

        # 残差前fc
        self.fc1 = nn.Linear(128, 128)

    def _activation(self):
        return nn.Tanh() if self.activation == "tanh" else nn.ReLU()

    def _conv_block(self, in_channels, out_channels, kernel_size=None, dropout=True, activ=None):
        """
        Convolution block with optional Batch Normalization, Activation, and Dropout.

        Parameters:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        - kernel_size: 卷积核的大小
        - dropout: 是否使用Dropout
        - activ: 激活函数，如果没有提供则使用默认的 self.activation

        Returns:
        - 返回一个包含卷积、批归一化、激活和 Dropout 的层
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size or self.kernel_size, padding='same')
        ]

        if self.use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        # 如果提供了激活函数，就使用提供的激活函数，否则使用默认激活函数
        if activ is not None:
            layers.append(activ)
        else:
            layers.append(self._activation())  # 默认使用 self.activation

        if self.use_dropout and dropout:
            layers.append(nn.Dropout2d(self.dropout_rate))

        return nn.Sequential(*layers)

    def _build_cnn_block(self):
        layers = []
        c = 3  # initial channel for waveform input
        f = self.f0
        for _ in range(3):
            f *= 2
            for _ in range(3):
                layers.append(self._conv_block(c, f))
                c = f
            layers.append(nn.MaxPool2d(kernel_size=(1, 4)))

        # Final 2 + 1 conv layers
        f *= 2
        for _ in range(2):
            layers.append(self._conv_block(c, f))
            c = f
        # Last conv without dropout
        layers.append(self._conv_block(c, f, dropout=False,activ=nn.Tanh()))

        return nn.Sequential(*layers)

    def _build_feature_merger(self):
        return nn.Sequential(
            self._conv_block(64 + 2, 128, kernel_size=(1, 1)),
            self._conv_block(128, 128, kernel_size=(1, 1), dropout=False)
        )

    def _build_predictor(self):
        return nn.Sequential(
            nn.Linear(128, 128),
            GaussianDropout(p=self.dropout_rate),
            # nn.Dropout(self.dropout_rate),
            self._activation(),
            nn.Linear(128, 4),
            nn.Tanh()
        )



    def forward(self, waveforms, coords, weights):
        # print("waveforms:",waveforms.shape)
        B = waveforms.size(0)
        N_sub = waveforms.size(1)
        x = waveforms.permute(0, 3, 1, 2)  # (B, 3, N_sub, T)
        x = self.feature_extractor(x)  # (B, C, N_sub, T_reduced)
        x = torch.max(x, dim=3, keepdim=True)[0]  # temporal max-pooling# (B, C, N_sub, 1)
        x = self.GaussianDropout(x)
        # x = self.dropout(x)

        # pad coords to match x shape and concat
        coords = coords.permute(0, 3, 1, 2)
        x = torch.cat([x, coords[:, :2, :, :]], dim=1)# (B, C+2, N_sub, 1)

        # print(x.shape)
        x = self.feature_merger(x)  # (B, C+2, N_sub, 1)
        x = self.GaussianDropout(x)

        # 变形方便做多头注意力： (B, 128, N_sub, 1) → (B, N_sub, 128)
        x = x.squeeze(-1).permute(0, 2, 1)

        # 构造节点掩码，weights>0的视为有效节点 (B, N_sub)
        mask = (weights > 0).float()

        # 多头图注意力 + 节点级残差（模块内已经加了残差）
        x = self.multihead_attention(x, mask)

        # # 用权重加权聚合节点特征 (B, N_sub, 128) * (B, N_sub, 1)
        # weights_exp = weights.unsqueeze(-1)
        # x = (x * weights_exp).sum(dim=1)  # (B, 128)

        # x_masked = x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        # x, _ = x_masked.max(dim=1)  # (B, 128)

        x = self.attention_pooling(x, weights)



        x_graph = self.GaussianDropout(x)

        # --- 9. 拼接 & 预测 ---
        x_struct = self.fc1(x_graph)
        # x_final = x_graph + x_struct
        out = self.predictor(x_struct)  # (B,4)
        return out

if __name__ == "__main__":
    import torch

    # 参数设置
    N_sub = 50
    N_t = 2048
    batch_size = 4  # 随便设置一个小的 batch 测试

    # 构建模型
    model = GraphNet(N_sub=N_sub, N_t=N_t, activation="relu")
    model.eval()  # 测试模式（禁用 dropout 等）

    # 构造 dummy 输入
    waveforms = torch.randn(batch_size, N_sub, N_t, 3)
    coords = torch.rand(batch_size, N_sub, 1, 3)  # 可随机值
    weights = torch.ones(batch_size, N_sub)
    weights[:, -10:] = 0

    # 前向测试
    with torch.no_grad():
        output = model(waveforms, coords, weights)

    print("Output shape:", output.shape)
    assert output.shape == (batch_size, 4), "输出维度应为 (B, 4)"
