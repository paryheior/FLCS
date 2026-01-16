import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import math
import numpy as np
import pdb


def get_named_beta_schedule(schedule_name='linear', num_diffusion_timesteps=1000,
                            noise_scale=0.1,
                            noise_min=0.0001,
                            noise_max=0.02
                            ) -> np.ndarray:
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    print("in get_named.. func: ")
    print("noise_scale: ", noise_scale)
    print("noise_min: ", noise_min)
    print("noise_max: ", noise_max)
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        # scale = 1000 / num_diffusion_timesteps
        # beta_start = scale * 0.0001
        # beta_end = scale * 0.02

        beta_start = noise_scale * noise_min
        beta_end = noise_scale * noise_max
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class GaussianDiffusion(nn.Module):
    def __init__(self, dtype: torch.dtype, model, betas: np.ndarray, w: float, v: float,
                 noise_scale, noise_min, noise_max, eta, timesteps,
                 device: torch.device):
        super().__init__()
        self.dtype = dtype
        self.model = model.to(device) # unet
        self.model.dtype = self.dtype
        self.device = device
        self.betas = torch.tensor(betas, dtype=self.dtype).to(self.device)
        self.w = w
        self.v = v
        self.T = len(betas)
        self.noise_scale = noise_scale,
        self.noise_min = noise_min,
        self.noise_max = noise_max,
        self.eta = eta
        self.timesteps = timesteps
        self.device = device
        self.alphas = 1 - self.betas
        # pdb.set_trace()
        self.log_alphas = torch.log(self.alphas).to(self.device)

        self.log_alphas_bar = torch.cumsum(self.log_alphas, dim=0).to(self.device)
        self.alphas_bar = torch.exp(self.log_alphas_bar).to(self.device)
        # self.alphas_bar = torch.cumprod(self.alphas, dim = 0)

        self.log_alphas_bar_prev = F.pad(self.log_alphas_bar[:-1], [1, 0], 'constant', 0).to(self.device)
        self.alphas_bar_prev = torch.exp(self.log_alphas_bar_prev).to(self.device)
        self.log_one_minus_alphas_bar_prev = torch.log(1.0 - self.alphas_bar_prev).to(self.device)
        # self.alphas_bar_prev = F.pad(self.alphas_bar[:-1],[1,0],'constant',1)

        # calculate parameters for q(x_t|x_{t-1})
        self.log_sqrt_alphas = 0.5 * self.log_alphas.to(self.device)
        self.sqrt_alphas = torch.exp(self.log_sqrt_alphas).to(self.device)
        # self.sqrt_alphas = torch.sqrt(self.alphas)

        # calculate parameters for q(x_t|x_0)
        self.log_sqrt_alphas_bar = 0.5 * self.log_alphas_bar.to(self.device)
        self.sqrt_alphas_bar = torch.exp(self.log_sqrt_alphas_bar).to(self.device)
        # self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.log_one_minus_alphas_bar = torch.log(1.0 - self.alphas_bar).to(self.device)
        self.sqrt_one_minus_alphas_bar = torch.exp(0.5 * self.log_one_minus_alphas_bar).to(self.device)

        # calculate parameters for q(x_{t-1}|x_t,x_0)
        # log calculation clipped because the \tilde{\beta} = 0 at the beginning
        self.tilde_betas = self.betas * torch.exp(
            self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar).to(self.device)
        self.log_tilde_betas_clipped = torch.log(torch.cat((self.tilde_betas[1].view(-1), self.tilde_betas[1:]), 0)).to(
            self.device)
        self.mu_coef_x0 = self.betas * torch.exp(0.5 * self.log_alphas_bar_prev - self.log_one_minus_alphas_bar).to(
            self.device)
        self.mu_coef_xt = torch.exp(
            0.5 * self.log_alphas + self.log_one_minus_alphas_bar_prev - self.log_one_minus_alphas_bar).to(self.device)
        # using \beta_t
        # self.vars = torch.cat((self.tilde_betas[1:2],self.betas[1:]), 0).to(self.device)
        # using \tilde\beta_t
        self.vars = self.tilde_betas
        self.coef1 = torch.exp(-self.log_sqrt_alphas).to(self.device)
        self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar.to(self.device)
        # calculate parameters for predicted x_0
        self.sqrt_recip_alphas_bar = torch.exp(-self.log_sqrt_alphas_bar).to(self.device)
        # self.sqrt_recip_alphas_bar = torch.sqrt(1.0 / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.exp(self.log_one_minus_alphas_bar - self.log_sqrt_alphas_bar).to(
            self.device)
        # self.sqrt_recipm1_alphas_bar = torch.sqrt(1.0 / self.alphas_bar - 1)

        # Coupled diffusion modeling (xT)
        ele = torch.zeros((self.T, self.T), dtype=torch.float64)
        for t in range(self.T):
            for s in range(t + 1):
                ele[t, s] = torch.exp(
                    0.5 * (torch.log(self.alphas_bar[t]) + torch.log(self.betas[s]) - torch.log(self.alphas_bar[s])))

        c_numerator = torch.sum(ele, dim=1)
        c_denominator = c_numerator[-1]

        self.const_c = c_denominator.to(self.device)
        self.ct = torch.exp(torch.log(c_numerator) - torch.log(c_denominator)).to(self.device)
        self.ct_prev = F.pad(self.ct[:-1], [1, 0], 'constant', 0.0).to(self.device)
        self.ct_prev_tmp = F.pad(self.ct[:-1], [1, 0], 'constant', 1.0).to(self.device)

        # reverse process of xT
        # part_1(cp1), part_2(cp2), part_3(cp3), (cp1 + cp2 + cp3) * x_T
        self.cp1 = torch.exp(
            torch.log(self.ct_prev_tmp) + torch.log(self.betas) + torch.log(0.5 * self.alphas)
            - self.log_one_minus_alphas_bar)
        self.cp1 = F.pad(self.cp1[1:], [1, 0], 'constant', 0.0).to(self.device)

        alphas_minus_alphas_bar = self.alphas - self.alphas_bar
        alphas_minus_alphas_bar = F.pad(alphas_minus_alphas_bar[1:], [1, 0], 'constant', 1.0).to(self.device)
        self.cp2 = - torch.exp(
            torch.log(alphas_minus_alphas_bar) + 0.5 * torch.log(self.betas)
            - torch.log(self.const_c) - self.log_one_minus_alphas_bar)
        self.cp2 = F.pad(self.cp2[1:], [1, 0], 'constant', 0.0).to(self.device)

        self.cp3 = - torch.exp(
            torch.log(self.betas) + torch.log(self.ct) -
            self.log_one_minus_alphas_bar).to(self.device)

        # self.coef1 = torch.exp(-self.log_sqrt_alphas).to(self.device)
        # self.coef2 = self.coef1 * self.betas / self.sqrt_one_minus_alphas_bar.to(self.device)
        self.coef3 = self.coef1 * (self.cp1 + self.cp2 + self.cp3)
        self.coef3 = self.coef3.to(self.device, dtype=torch.float32)
        self.const_c = self.const_c.to(dtype=torch.float32)
        self.ct = self.ct.to(dtype=torch.float32)
        self.ct_prev = self.ct_prev.to(dtype=torch.float32)

        # print("debug: \n")
        # print("betas:", self.betas)
        # print("ele: ", ele)
        # print("self.ct: ", self.ct)
        # print("self.ct_prev: ", self.ct_prev)
        # print("self.const_c: ", self.const_c)
        # print("self.cp1: ", self.cp1)
        # print("self.cp2: ", self.cp2)
        # print("self.cp3: ", self.cp3)
        # print("self.alphas: ", self.alphas)
        # print("slef.alphas_bar: ", self.alphas_bar)
        # print("self.coef3: ", self.coef3)

    @staticmethod
    def _extract(coef: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        input:

        coef : an array
        t : timestep
        x_shape : the shape of tensor x that has K dims(the value of first dim is batch size)

        output:

        a tensor of shape [batchsize,1,...] where the length has K dims.
        """
        assert t.shape[0] == x_shape[0]
        # pdb.set_trace()

        neo_shape = torch.ones_like(torch.tensor(x_shape))
        neo_shape[0] = x_shape[0]
        neo_shape = neo_shape.tolist()
        chosen = coef[t]
        chosen = chosen.to(t.device)
        return chosen.reshape(neo_shape)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, x_T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sample from q(x_t|x_0)
        """
        # pdb.set_trace()
        eps = torch.randn_like(x_0, requires_grad=False)
        return self._extract(self.sqrt_alphas_bar.to(self.device), t, x_0.shape) * x_0 \
               + self._extract(self.ct.to(self.device), t, x_0.shape) * F.dropout(x_T, p=0.5) \
               + self._extract(self.sqrt_one_minus_alphas_bar.to(self.device), t, x_0.shape) * eps, eps



    def compute_c(self, c, t):
        c = torch.cat([torch.zeros(1).to(c.device), c], dim=0)
        r = c.index_select(0, t + 1).view(-1, 1)
        return r

    def get_seq(self):
        ans = list(range(0, self.T, self.timesteps))
        return ans

    def generalized_steps(self, x_t, seq, **model_kwargs) -> torch.Tensor:
        seq = self.get_seq()

        n = x_t.size(0)
        seq_next = [-1] + list(seq[:-1])
        # x0_preds = []
        # xs  = [x]
        x_T = model_kwargs['x_T']
        cvt_model_kwargs = {}
        for k, v in model_kwargs.items():
            if k != 'x_T':
                cvt_model_kwargs[k] = v
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x_t.device)
            next_t = (torch.ones(n) * j).to(x_t.device)

            at = self.compute_alpha(self.betas, t.long())
            at_next = self.compute_alpha(self.betas, next_t.long())

            ct = self.compute_c(self.ct, t.long())
            ct_next = self.compute_c(self.ct, next_t.long())

            et = self.model(x_t, t, **cvt_model_kwargs)

            # print("et, ct, x_t, at.shape:", et.shape, ct.shape, x_t.shape, at.shape)
            x0_t = (x_t - ct * x_T - et * (1 - at).sqrt()) / at.sqrt()

            c1 = self.eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()

            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            x_t = at_next.sqrt() * x0_t + c1 * torch.randn_like(x_t) + c2 * et + ct_next * x_T
        return x_t

    def rec_backward_ddim(self, x_t, **model_kwargs) -> torch.Tensor:
        # local_rank = get_rank()
        local_rank = 0
        # if local_rank == 0:
        #    print('Start generating...')
        if model_kwargs == None:
            model_kwargs = {}
        # p sample:
        tlist = torch.ones([x_t.shape[0]], device=self.device) * (self.T - 1)
        x_T_dict = {"x_T": x_t}
        model_kwargs.update(x_T_dict)
        seq = self.get_seq()
        return self.generalized_steps(x_t, seq, **model_kwargs)
        # print("model_kwargs:", model_kwargs)
        # for _ in range(self.T):
        #    #tlist -= 1
        #    #x_t = p_saple(xxx)
        #    x_t = self.rec_p_sample(x_t, tlist, **model_kwargs)
        #    tlist -= 1
        # return x_t

    def compute_alpha(self, betas, t):
        betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0)
        a = (1 - betas).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
        return a


    def trainloss(self, x_0: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        calculate the loss of denoising diffusion probabilistic model
        """
        if model_kwargs == None:
            model_kwargs = {}
        t = torch.randint(self.T, size=(x_0.shape[0],), device=self.device)
        if 'x_T' not in model_kwargs:
            raise NotImplementedError
        else:
            x_T = model_kwargs['x_T']
        x_t, eps = self.q_sample(x_0, t, x_T=x_T)
        # pred_eps = self.model(x_t, t, **model_kwargs)
        # print("eps: ", eps.shape, pred_eps.shape)
        # cemb_shape = model_kwargs['cemb'].shape
        cvt_model_kwargs = {}
        for k, v in model_kwargs.items():
            if k != 'x_T':
                cvt_model_kwargs[k] = v
                # print("v.shape:", v.shape)
        pred_eps = self.model(x_t, t, **cvt_model_kwargs)
        # model_kwargs['cemb'] = torch.zeros(cemb_shape, device = self.device)
        # pred_eps_uncond = self.model(x_t, t, **model_kwargs)
        # pred_eps = (1 + self.w) * pred_eps_cond - self.w * pred_eps_uncond
        # pred_eps = pred_eps_cond
        loss = F.mse_loss(pred_eps, eps, reduction='mean')
        return loss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class GaussianDiffusion(nn.Module):
#     def __init__(self, dtype: torch.dtype, model, betas: np.ndarray, w: float, v: float,
#                  noise_scale, noise_min, noise_max, eta, timesteps,
#                  device: torch.device):
#         super().__init__()
#         self.dtype = dtype
#         self.model = model.to(device)
#         self.model.dtype = self.dtype
#         self.device = device
        
#         # --- 兼容性参数 (虽然FM不需要beta schedule，但为了保持接口一致，保留它们) ---
#         self.betas = torch.tensor(betas, dtype=self.dtype).to(self.device)
#         self.w = w
#         self.v = v
#         self.T = len(betas) # 总的时间步范围，例如 1000
#         self.timesteps = timesteps # 采样时的步数，例如 100
#         self.noise_scale = noise_scale
#         self.noise_min = noise_min
#         self.noise_max = noise_max
#         self.eta = eta
        
#         # FM 只需要知道这是一个极小值，防止 sigma=0 时的数值不稳定
#         self.sigma_min = 1e-4 

#     def _scale_t(self, t):
#         """
#         将 [0, 1] 的时间 t 映射回 [0, T] 的范围。
#         因为原本的 Unet 可能是基于 0-1000 的整数位置编码设计的。
#         """
#         return t * (self.T - 1)

#     def trainloss(self, x_0: torch.Tensor, **model_kwargs) -> torch.Tensor:
#         """
#         Flow Matching 的 Loss 计算。
#         x_0: 目标 (Target)，即真实的 Item ID Embedding [B, D]
#         model_kwargs['x_T']: 条件 (Source)，即 Side Info Embedding [B, D]
#         """
#         if model_kwargs is None:
#             model_kwargs = {}
        
#         # 1. 获取 Source (x_T) 和 Target (x_0)
#         if 'x_T' not in model_kwargs:
#             raise NotImplementedError("Conditional Flow Matching requires x_T (Side Info)")
        
#         x_1 = x_0  # 目标：ID Embedding
#         x_source = model_kwargs['x_T'] # 起点：Side Info
        
#         # 保持与原代码一致的 Dropout 逻辑：
#         # 有一定概率丢弃 Condition，增强模型的鲁棒性（学习无条件流 + 条件流）
#         # 如果 drop 掉，起点就变成了 0 (或者纯噪声，这里设为0保持 consistent)
#         x_source_dropped = F.dropout(x_source, p=0.1)

#         batch_size = x_1.shape[0]

#         # 2. 随机采样时间 t ~ Uniform[0, 1]
#         t = torch.rand(batch_size, device=self.device).type(self.dtype)
        
#         # 3. 构造中间状态 x_t (Optimal Transport Path / Linear Interpolation)
#         # 路径公式: x_t = (1 - t) * x_source + t * x_1
#         # 为了数值稳定，通常在 x_source 侧保留一点极小的 sigma
#         # 广播 t 的维度到 [B, 1] 以便相乘
#         t_b = t.view(-1, 1)
        
#         # Flow Matching 的加噪公式 (插值)
#         # x_t 位于 x_source 和 x_1 的连线上
#         x_t = (1 - (1 - self.sigma_min) * t_b) * x_source_dropped + t_b * x_1
        
#         # 4. 计算目标速度 (Ground Truth Velocity)
#         # 也就是向量: Target - Source
#         target_v = x_1 - (1 - self.sigma_min) * x_source_dropped

#         # 5. 模型预测
#         # 准备模型参数，剔除 x_T 防止重复传入 (与原代码逻辑一致)
#         cvt_model_kwargs = {k: v for k, v in model_kwargs.items() if k != 'x_T'}
        
#         # 将 t [0, 1] 映射回模型习惯的 [0, T]
#         t_input = self._scale_t(t)
        
#         # 模型现在的任务是：给定当前位置 x_t 和时间 t，预测流向 target_v
#         pred_v = self.model(x_t, t_input, **cvt_model_kwargs)

#         # 6. 计算 MSE Loss
#         loss = F.mse_loss(pred_v, target_v, reduction='mean')
        
#         return loss


#     def rec_backward_ddim(self, x_t, **model_kwargs) -> torch.Tensor:
#         """
#         使用 Euler 方法求解 ODE 进行采样。
#         x_t: 这里传入的其实是起点 (Source/Side Info)，对应原代码的 mean_p
#         """
#         if model_kwargs is None:
#             model_kwargs = {}

#         # 这里的 x_t 是起点 (t=0)
#         x_curr = x_t.clone()
        
#         # 准备模型参数
#         # 注意：在 CFG (Classifier-Free Guidance) 或者 Condition 场景下，
#         # 原代码在 trainloss 里把 x_T 剔除了，但在 sampling 时 x_t 本身就是 x_T。
#         # 我们这里不需要额外的 x_T 放入 model_kwargs，因为 x_curr 本身携带了信息。
#         cvt_model_kwargs = {k: v for k, v in model_kwargs.items() if k != 'x_T'}

#         # 确定步数
#         # steps = self.timesteps # 例如 10 或 20，FM 通常需要的步数比 DDPM 少得多
#         steps = 2 # 默认值
#         # if steps is None:
#         #     steps = 100 # 默认值
            
#         dt = 1.0 / steps # 时间步长
        
#         # Euler ODE Solver 循环
#         # 从 t=0 (Source) 走到 t=1 (Target)
#         for i in range(steps):
#             # 当前时间 t (归一化到 0-1)
#             t_float = i / steps
            
#             # 构造 batch 的时间张量
#             batch_size = x_curr.shape[0]
#             t = torch.full((batch_size,), t_float, device=self.device, dtype=self.dtype)
            
#             # 映射回模型的 [0, T] 范围
#             t_input = self._scale_t(t)
            
#             # 预测速度 v
#             pred_v = self.model(x_curr, t_input, **cvt_model_kwargs)
            
#             # 更新位置: x_{new} = x_{old} + v * dt
#             x_curr = x_curr + pred_v * dt
            
#         return x_curr