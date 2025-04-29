from packaging import version
import torch
import torch.nn as nn     
import numpy as np
import torch.nn.functional as F

def sinkhorn(dot, max_iter=100):
    """使用Sinkhorn算法对输入的dot进行归一化处理
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    K = dot
    # K: n x in_size x out_size
    u = K.new_ones((n, in_size))# 初始化u矩阵，形状为 n x in_size，元素为1
    v = K.new_ones((n, out_size))# 初始化v矩阵，形状为 n x out_size，元素为1
    a = float(out_size / in_size) # 计算常数a，用于归一化
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)# 更新u矩阵
        v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)# 更新v矩阵
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))# 根据u和v更新K矩阵
    return K

# def compute_alpha_beta(q, k):
#     """计算条件输入和示例的质量向量"""
#     n, in_size, in_dim = q.shape
#     m, out_size, _ = k.shape
    
#     # 计算条件输入和示例的全局特征均值
#     q_mean = torch.mean(q, dim=1, keepdim=True)  # n x 1 x in_dim
#     k_mean = torch.mean(k, dim=1, keepdim=True)  # m x 1 x in_dim
    
#     # 计算质量向量 α 和 β
#     alpha = torch.sum(q * q_mean, dim=-1)  # n x in_size
#     beta = torch.sum(k * k_mean, dim=-1)    # m x out_size
    
#     return alpha, beta

# def sinkhornKL(u, v, K, alpha, beta, tau, eps, max_iter):
#     """带KL散度约束的Sinkhorn迭代"""
#     for _ in range(max_iter):
#         u = alpha / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + eps)
#         v = beta / (torch.matmul(u.unsqueeze(-2), K).squeeze(-2) + eps)
#     return u, v

# def UnbalancedOT(q, k, eps=1.0, max_iter=100, tau=1.0):
#     n, in_size, in_dim = q.shape
#     m, out_size, _ = k.shape
    
#     # 计算成本矩阵C
#     C = torch.einsum('bid,bod->bio', q, k)
#     # 调整K的形状
#     K = 1-C.view(-1, in_size, out_size)
#     npatches = q.size(1)
#     # 创建对角矩阵，并在K中将对角线元素置为-10
#     mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
#     diagonal = torch.eye(npatches, device=q.device, dtype=mask_dtype)[None, :, :]
#     K.masked_fill_(diagonal, -10)
#     # 计算质量向量α和β
#     alpha, beta = compute_alpha_beta(q, k)
#     # 初始化u和v
#     u = torch.ones_like(K[:, :, 0])
#     v = torch.ones_like(K[:, 0, :])
#     # 应用带KL散度的Sinkhorn迭代
#     u, v = sinkhornKL(u, v, K, alpha, beta, tau, eps, max_iter)
#     # 计算最终的传输矩阵T
#     T = u.view(n, in_size, 1) * K * v.view(n, 1, out_size)
    
#     return T

def sinkhornKL(u, v, K, alpha, beta, tau, eps, max_iter):
    """带KL散度约束的Sinkhorn迭代"""
    B, N, = alpha.shape
    _, M, = beta.shape
    
    for _ in range(max_iter):
        # 更新u：KL(u || α) + Sinkhorn项
        u_new = alpha - tau * torch.log(torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + eps)
        u_new = u_new - torch.logsumexp(u_new, dim=1, keepdim=True)  # 稳定化
        
        # 更新v：KL(v || β) + Sinkhorn项
        v_new = beta - tau * torch.log(torch.matmul(u_new.unsqueeze(-2), K).squeeze(-2) + eps)
        v_new = v_new - torch.logsumexp(v_new, dim=1, keepdim=True)  # 稳定化
        
        # 收敛检查（可选）
        if torch.max(torch.abs(u_new - u)) < 1e-5 and torch.max(torch.abs(v_new - v)) < 1e-5:
            break
        u, v = u_new, v_new
    
    return u, v

def UnbalancedOT(q, k, alpha=None, beta=None, eps=1.0, max_iter=100, tau=1.0):
    """
    实现Unbalanced Optimal Transport
    Args:
        q: 源样本 [B, N, D]
        k: 目标样本 [B, M, D]
        alpha: 源质量向量 [B, N] (默认None，自动计算为均匀分布)
        beta: 目标质量向量 [B, M] (默认None，自动计算为均匀分布)
        eps: 正则化系数
        max_iter: Sinkhorn迭代次数
        tau: KL散度权重
    Returns:
        T: 传输矩阵 [B, N, M]
    """
    B, N, D = q.shape
    _, M, _ = k.shape
    
    # 计算成本矩阵C
    C = torch.cdist(q.view(B*N, D), k.view(B*M, D), p=2).view(B, N, M)
    
    # 构建核矩阵K = exp(-(C)/eps)
    K = (-C / eps).exp()
    
    # 处理质量向量
    if alpha is None:
        alpha = torch.ones(B, N, device=q.device) / N  # 默认均匀分布
    if beta is None:
        beta = torch.ones(B, M, device=q.device) / M  # 默认均匀分布
    
    # 初始化对偶变量
    u = torch.zeros(B, N, device=q.device)
    v = torch.zeros(B, M, device=q.device)
    
    # 执行Sinkhorn迭代
    u, v = sinkhornKL(u, v, K, alpha, beta, tau, eps, max_iter)
    
    # 计算传输矩阵T
    T = torch.matmul(u.unsqueeze(-1), v.unsqueeze(-2)) * K
    
    return T

def OT(q, k, eps=1.0, max_iter=100, cost_type='easy'):
    """使用Sinkhorn OT算法计算权重
    q: n x in_size x in_dim
    k: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = q.shape
    m, out_size = k.shape[:-1]
    # 计算输入的C矩阵，形状为 n x m x in_size x out_size
    C = torch.einsum('bid,bod->bio', q, k)
    if cost_type == 'easy':
        K = 1 - C.clone() # 余弦距离
    elif cost_type == 'hard':
        K = C.clone()
    #K = 1 - C.clone()
    npatches = q.size(1)
    # 创建对角矩阵，并在K中将对角线元素置为-10
    mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
    diagonal = torch.eye(npatches, device=q.device, dtype=mask_dtype)[None, :, :]
    K.masked_fill_(diagonal, -10)

    # 将K重新调整形状为 nm x in_size x out_size
    # K: n x m x in_size x out_size
    K = K.reshape(-1, in_size, out_size)
    # K: nm x in_size x out_size
    # 对K进行指数运算，然后使用Sinkhorn算法进行归一化处理
    K = torch.exp(K / eps)
    K = sinkhorn(K, max_iter=max_iter)
    # K: nm x in_size x out_size
    K = K.permute(0, 2, 1).contiguous()
    # print("K的shape：",K.shape) 
    return K


class MC_Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        #self.l2_norm = Normalize(2)

    def forward(self, feat_src, feat_tgt, feat_gen):
        batchSize = feat_src.shape[0]
        dim = feat_src.shape[1]   
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size // len(self.opt.gpu_ids)
        # print(feat_src.shape,feat_tgt.shape,feat_gen.shape)
        # if self.loss_type == 'MoNCE':
        ot_src = feat_src.view(batch_dim_for_bmm, -1, dim).detach()
        ot_tgt = feat_tgt.view(batch_dim_for_bmm, -1, dim).detach()
        ot_gen = feat_gen.view(batch_dim_for_bmm, -1, dim)
        #print("ot_src:",ot_src.shape)
        #print("ot_tgt:",ot_tgt.shape)
        #print("ot_gen:",ot_gen.shape)
        f1 = OT(ot_src, ot_tgt, eps=self.opt.eps, max_iter=50, )
        # print("F1:",f1.shape)
        f2 = OT(ot_tgt, ot_gen, eps=self.opt.eps, max_iter=50)
        # print("F2:",f2.shape)
        
        MC_Loss = F.l1_loss(f1, f2)
        return MC_Loss

class UMC_Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        #self.l2_norm = Normalize(2)

    def forward(self, feat_src, feat_tgt, feat_gen):
        batchSize = feat_src.shape[0]
        dim = feat_src.shape[1]   
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size // len(self.opt.gpu_ids)
        # print(feat_src.shape,feat_tgt.shape,feat_gen.shape)
        # if self.loss_type == 'MoNCE':
        ot_src = feat_src.view(batch_dim_for_bmm, -1, dim).detach()
        ot_tgt = feat_tgt.view(batch_dim_for_bmm, -1, dim).detach()
        ot_gen = feat_gen.view(batch_dim_for_bmm, -1, dim)
        #print("ot_src:",ot_src.shape)
        #print("ot_tgt:",ot_tgt.shape)
        #print("ot_gen:",ot_gen.shape)
        f1 = UnbalancedOT(ot_src, ot_tgt, eps=self.opt.eps, max_iter=50, )
        # print("F1:",f1.shape)
        f2 = UnbalancedOT(ot_tgt, ot_gen, eps=self.opt.eps, max_iter=50)
        # print("F2:",f2.shape)
        
        MC_Loss = F.l1_loss(f1, f2)
        return MC_Loss