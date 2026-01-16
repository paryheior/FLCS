import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RectifiedFlowWrapper(nn.Module):
    def __init__(self, velocity_model, device):
        """
        velocity_model: 预测速度 v 的神经网络 (CSDM 中的 Unet 或 MLP)
        device: 运行设备
        """
        super().__init__()
        self.net = velocity_model.to(device)
        self.device = device

    def get_time_embedding(self, batch_size):
        # 随机采样时间 t ~ U[0, 1]
        return torch.rand(batch_size, device=self.device)

    def step(self, x_t, t_start, t_end, x_cond=None):
        """
        欧拉积分步：从 t_start 到 t_end
        x_t: 当前状态
        x_cond: 条件信息 (可选，如果网络需要额外条件输入)
        """
        batch_size = x_t.shape[0]
        
        # 构造时间输入 tensor
        t = torch.full((batch_size,), t_start, device=self.device)
        
        # 预测速度 v_t
        # 注意：这里假设 self.net 接受 (x, t) 作为输入，与 CSDM 的 Unet 接口保持一致
        v_pred = self.net(x_t, t) 
        
        # Euler update: x_{t+1} = x_t + v * dt
        dt = t_end - t_start
        x_next = x_t + v_pred * dt
        
        return x_next

    def get_loss(self, x_1, x_0):
        """
        计算 Flow Matching Loss
        Args:
            x_1: Target (真实的 Item ID Embedding)
            x_0: Source (物品的 Side Information Embedding)
        """
        batch_size = x_1.shape[0]
        
        # 1. 随机采样时间 t [Batch]
        t = self.get_time_embedding(batch_size)
        
        # 2. 维度对齐，用于广播计算 [Batch, 1]
        # 官方代码中使用了 [:, None, None, None] 是因为处理的是图片
        # 你的数据是 Embedding [Batch, Dim]，所以只需要扩展一维
        t_expand = t.view(-1, 1)
        
        # 3. 构造路径 (Path) - 对应官方代码 create_flow
        # x_t = t * x_1 + (1 - t) * x_0
        # t=0 时是 Side Info (x_0), t=1 时是 Item Emb (x_1)
        x_t = t_expand * x_1 + (1 - t_expand) * x_0
        
        # 4. 计算目标速度 (Target Velocity)
        # 直线的导数 v = x_1 - x_0
        v_target = x_1 - x_0
        
        # 5. 模型预测速度 (Driver)
        # 传入 x_t 和 t
        v_pred = self.net(x_t, t)
        
        # 6. 计算 MSE Loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss

    def sample(self, x_0, steps=10):
        """
        推理阶段：从 x_0 (Side Info) 生成 x_1 (Item Emb)
        """
        x_t = x_0.clone()
        
        # 生成时间步序列，例如 [0.0, 0.1, ..., 1.0]
        times = torch.linspace(0, 1, steps + 1, device=self.device)
        
        for i in range(steps):
            t_start = times[i]
            t_end = times[i+1]
            
            # 执行一步欧拉积分
            x_t = self.step(x_t, t_start, t_end)
            
        return x_t