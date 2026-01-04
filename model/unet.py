# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# import numpy as np
# import math

# class Unet(nn.Module):
#     """
#     A deep neural network for the reverse diffusion preocess.
#     """
#     def __init__(self, in_dims, out_dims, emb_size,
#                  #cemb_size,
#                  time_type="cat", norm=False, dropout=0.5):
#         super(Unet, self).__init__()
#         self.in_dims = in_dims
#         self.out_dims = out_dims
#         assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
#         self.time_type = time_type
#         self.time_emb_dim = emb_size
#         #self.cemb_dim = cemb_size
#         self.norm = norm

#         self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

#         if self.time_type == "cat":
#             in_dims_temp = [self.in_dims[0] +
#                             self.time_emb_dim
#                             #+ self.cemb_dim
#                             ] + self.in_dims[1:]
#         else:
#             raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
#         out_dims_temp = self.out_dims
        
#         self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
#             for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
#         self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
#             for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
#         self.drop = nn.Dropout(dropout)
#         self.init_weights()
    
#     def init_weights(self):
#         for layer in self.in_layers:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0 / (fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for weights
#             layer.bias.data.normal_(0.0, 0.001)
        
#         for layer in self.out_layers:
#             # Xavier Initialization for weights
#             size = layer.weight.size()
#             fan_out = size[0]
#             fan_in = size[1]
#             std = np.sqrt(2.0 / (fan_in + fan_out))
#             layer.weight.data.normal_(0.0, std)

#             # Normal Initialization for weights
#             layer.bias.data.normal_(0.0, 0.001)
        
#         size = self.emb_layer.weight.size()
#         fan_out = size[0]
#         fan_in = size[1]
#         std = np.sqrt(2.0 / (fan_in + fan_out))
#         self.emb_layer.weight.data.normal_(0.0, std)
#         self.emb_layer.bias.data.normal_(0.0, 0.001)
    
#     def forward(self, x, timesteps): #, cemb):
#         time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
#         emb = self.emb_layer(time_emb)
#         if self.norm:
#             x = F.normalize(x)
#         x = self.drop(x)
#         h = torch.cat([x,
#                        emb,
#                        #cemb
#                        ], dim=-1)
#         for i, layer in enumerate(self.in_layers):
#             #print("layer: ", layer.weight.shape, layer.weight.dtype)
#             #print("h: ",h.shape, h.dtype)
#             h = layer(h)
#             h = torch.tanh(h)
        
#         for i, layer in enumerate(self.out_layers):
#             h = layer(h)
#             if i != len(self.out_layers) - 1:
#                 h = torch.tanh(h)
        
#         return h


# def timestep_embedding(timesteps, dim, max_period=10000):
#     """
#     Create sinusoidal timestep embeddings.

#     :param timesteps: a 1-D Tensor of N indices, one per batch element.
#                       These may be fractional.
#     :param dim: the dimension of the output.
#     :param max_period: controls the minimum frequency of the embeddings.
#     :return: an [N x dim] Tensor of positional embeddings.
#     """

#     half = dim // 2
#     freqs = torch.exp(
#         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
#     ).to(timesteps.device)
#     args = timesteps[:, None].float() * freqs[None]
#     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
#     if dim % 2:
#         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
#     return embedding

# import torch.nn as nn
# import torch.nn.functional as F
# import math
# import numpy as np

# # ==========================================
# class ResidualBlock(nn.Module):
#     """
#     带有残差连接的全连接块
#     """
#     def __init__(self, dim, dropout=0.1):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.activation = nn.SiLU()
#         self.linear1 = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim, dim)

#     def forward(self, x):
#         h = self.norm(x)
#         h = self.linear1(h)
#         h = self.activation(h)
#         h = self.dropout(h)
#         h = self.linear2(h)
#         return x + h

# class VelocityNet(nn.Module):
#     """
#     速度预测网络 v_t(x, t, condition)
#     """
#     def __init__(self, in_dim, out_dim, time_emb_dim=64, hidden_dim=256, num_layers=2, dropout=0.1):
#         super(VelocityNet, self).__init__()
#         self.time_emb_dim = time_emb_dim
        
#         # 时间嵌入层
#         self.time_mlp = nn.Sequential(
#             nn.Linear(time_emb_dim, time_emb_dim * 2),
#             nn.SiLU(),
#             nn.Linear(time_emb_dim * 2, time_emb_dim),
#         )

#         # 输入投影: x + t + condition
#         # 假设 condition 维度与 x 相同 (经过 CSFM 中的 sideinfo_encoder 处理)
#         input_total_dim = in_dim + time_emb_dim + in_dim
#         self.input_proj = nn.Linear(input_total_dim, hidden_dim)

#         # 骨干网络
#         self.blocks = nn.ModuleList([
#             ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
#         ])

#         # 输出层
#         self.final_norm = nn.LayerNorm(hidden_dim)
#         self.output_proj = nn.Linear(hidden_dim, out_dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x, t, condition):
#         # 1. 处理时间 t
#         if len(t.shape) > 1:
#             t = t.view(-1)
#         t_emb = timestep_embedding(t, self.time_emb_dim).to(x.device)
#         t_emb = self.time_mlp(t_emb)

#         # 2. 维度对齐 (Squeeze 多余的维度)
#         if len(x.shape) == 3 and x.shape[1] == 1:
#             x = x.squeeze(1)
#         if len(condition.shape) == 3 and condition.shape[1] == 1:
#             condition = condition.squeeze(1)

#         # 3. 拼接输入
#         inp = torch.cat([x, t_emb, condition], dim=-1)
        
#         # 4. 网络前向
#         h = self.input_proj(inp)
#         for block in self.blocks:
#             h = block(h)
#         h = self.final_norm(h)
#         out = self.output_proj(h)
        
#         return out

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class Unet(nn.Module):
    """
    A deep neural network for the reverse diffusion process (Modified for Flow Matching).
    """
    def __init__(self, in_dims, out_dims, emb_size,
                 #cemb_size,
                 time_type="cat", norm=False, dropout=0.5):
        super(Unet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        #self.cemb_dim = cemb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            # [修改点 1]: 输入维度调整
            # 原始是: self.in_dims[0] + self.time_emb_dim
            # Flow Matching 需要同时输入 x_t 和 x_T (Side Info)，且两者维度相同
            # 所以输入维度变为: input_dim * 2 + time_dim
            in_dims_temp = [self.in_dims[0] * 2 +
                            self.time_emb_dim
                            #+ self.cemb_dim
                            ] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    # [修改点 2]: 增加 x_T 和 **kwargs 参数
    def forward(self, x, timesteps, x_T=None, **kwargs): #, cemb):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        
        if self.norm:
            x = F.normalize(x)
        
        x = self.drop(x)
        
        # [修改点 3]: 处理 x_T (Side Info) 并拼接
        if x_T is None:
            # 如果万一没有传 x_T，用全 0 填充防止崩溃（虽然在你的逻辑里应该总是有值的）
            x_T = torch.zeros_like(x)
        
        if self.norm:
            x_T = F.normalize(x_T) # 对条件也做归一化，保持分布一致

        # 核心：拼接 x(当前状态), x_T(条件), emb(时间)
        h = torch.cat([x,
                       x_T, 
                       emb,
                       #cemb
                       ], dim=-1)
                       
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding