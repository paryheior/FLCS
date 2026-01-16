from typing import Callable, Union


import torch
import torch.nn as nn
import torch.optim

ModuleType = Union[str, Callable[..., nn.Module]]

import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor
from torchdiffeq import odeint_adjoint as odeint

import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
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



class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + h  # 残差连接


class ResNetMLP(nn.Module):
    def __init__(
            self,
            d_item=16,  # Item Embedding 维度
            d_condition=64,  # 【关键】这里必须填拼接后的总维度
            d_hidden=256,
            num_layers=3,
            dropout=0.1
    ):
        super().__init__()
        self.input_dim = d_item + d_condition
        self.d_item = d_item
        self.input_proj = nn.Linear(self.input_dim, d_hidden)
        self.time_dim = d_hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(d_item, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.blocks = nn.ModuleList([
            ResBlock(d_hidden, dropout) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(d_hidden)
        self.head = nn.Linear(d_hidden, d_item)

    def forward(self, condition, x_t, timesteps):
        """
        condition: [Batch, d_condition] <-- 已经是 2D 张量
        x_t:       [Batch, d_item]
        timesteps: [Batch]
        """
        net_input = torch.cat([x_t, condition], dim=1)
        h = self.input_proj(net_input)
        if timesteps.dim() == 2: timesteps = timesteps.squeeze()
        t_raw = timestep_embedding(timesteps, self.d_item)
        t_emb = self.time_mlp(t_raw)
        for block in self.blocks:
            h = h + t_emb
            h = block(h)
        h = self.output_norm(h)
        pred_item = self.head(h)

        return pred_item

class MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout, initialization='kaiming'):

        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None
        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)

        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a / b, dim=-1)

        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            n_layers: int,
            d_token: int,
            n_heads: int,
            d_out: int,
            d_ffn_factor: int,
            attention_dropout=0.0,
            ffn_dropout=0.0,
            residual_dropout=0.0,
            activation='relu',
            prenormalization=True,
            initialization='kaiming',
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()

            self.layers.append(layer)

        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        # self.activation = lib.get_activation_fn(activation)
        # self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
            )

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x



class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, d_in, dim_t=512, use_mlp=True):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        self.use_mlp = use_mlp

    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        emb = self.time_embed(emb)

        x = self.proj(x) + emb
        return self.mlp(x)



class UniModMLP(nn.Module):
    """
        Input:
            x_num: [bs, d_numerical]
            x_cat: [bs, len(categories)]
        Output:
            x_num_pred: [bs, d_numerical], the predicted mean for numerical data
            x_cat_pred: [bs, sum(categories)], the predicted UNORMALIZED logits for categorical data
    """

    def __init__(
            self, num_layers, d_token,
            n_head=1, factor=4, bias=True, d_in = 32, dim_t=512, use_mlp=True, **kwargs
    ):
        super().__init__()
        self.d_token = d_token
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.mlp = MLP(d_in = d_token, dim_t=dim_t, use_mlp=use_mlp)
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.model = nn.ModuleList([self.encoder, self.mlp, self.decoder])

    def forward(self, condition, x_t, timesteps):
        t_emb = timestep_embedding(timesteps, self.d_token)
        decoder_input = torch.cat([x_t.unsqueeze(1), condition, t_emb.unsqueeze(1)], dim=1)
        y = self.encoder(decoder_input)
        bsz, seq_len, dim = y.shape
        y_flat = y.reshape(-1, dim)
        t_expanded = timesteps.unsqueeze(1).repeat(1, seq_len).reshape(-1)
        pred_y_flat = self.mlp(y_flat, t_expanded)  # [Batch * 6, 16]
        pred_y = pred_y_flat.reshape(bsz, seq_len, dim)
        pred_e = self.decoder(pred_y)  # [Batch, 6, 16]
        item_pred = pred_e[:, 0, :]  # [Batch, 16]
        return item_pred



class Velocity(torch.nn.Module):
    def __init__(self, model, condition):
        super(Velocity, self).__init__()
        self.model = model
        self.condition = condition

    def forward(self, t, x):
        t_vec = t.expand(x.shape[0])
        pred_x1 = self.model(x_t=x, timesteps=t_vec, condition=self.condition)
        v_t = (pred_x1 - x) / (1 - t + 1e-5)
        return v_t




class ExFM(torch.nn.Module):
    def __init__(
            self,
            vf_fn,
            device=torch.device('cuda:0'),
            **kwargs
        ):
        super(ExFM, self).__init__()
        self.device = device
        self.vf_fn = vf_fn

    def _mvgloss(self, mu_t, x_num_t, t):
        n, k = mu_t.shape
        dev = mu_t.device

        dt = mu_t.dtype
        identity = torch.eye(k, device=dev, dtype=dt).unsqueeze(0).expand(n, -1, -1)
        scale = 1 - (1 - 0.01) * t.unsqueeze(1) ** 2
        sigma = scale * identity
        dist = torch.distributions.MultivariateNormal(mu_t, sigma)
        return -dist.log_prob(x_num_t).mean()

    def train_loss(self, condition, x):
        b = x.shape[0]
        t = torch.rand(b, device=self.device)
        t_view = t.view(b, 1)
        noise = torch.randn_like(x)
        x_t = t_view * x + (1 - t_view) * noise
        pred_x1 = self.vf_fn(x_t=x_t, timesteps=t, condition=condition)

        loss = F.mse_loss(pred_x1, x,reduction='mean')
        print(loss)
        return loss


    @torch.no_grad()
    def sample(self, condition, d_in = 16):
        bsz = condition.shape[0]
        dev = condition.device
        t = torch.tensor([0.0, 0.999]).to(dev)
        x0 = torch.randn(bsz, d_in, device=dev)
        vf = Velocity(self.vf_fn, condition)
        trajectory = odeint(vf, x0, t, method="euler")
        out = trajectory[-1]
        return out

