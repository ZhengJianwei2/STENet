import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple
from torch import Tensor


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, n_win, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.p = n_win ** 2
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c) (1,16,64)

        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)

        attn_logit[:, torch.arange(self.p), torch.arange(self.p)] = 1  # 1 to 1 set value on diagonal

        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k) 路由权重矩阵

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v


class BiLevelRoutingCrossAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(self, dim, num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  n_win=self.n_win,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x1, x2, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
        N, H, W, C = x1.size()
        assert H % self.n_win == 0 and W % self.n_win == 0

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x1 = rearrange(x1, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)   #[1, 64, 8, 8, 64]
        x2 = rearrange(x2, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        q1, kv1 = self.qkv(x1)    # q1 [1, 64, 8, 8, 64], kv1 [1, 64, 8, 8, 128]
        q2, kv2 = self.qkv(x2)    # q1 [1, 64, 4, 4, 128], kv1 [1, 64, 4, 4, 256]

        q1_pix = rearrange(q1, 'n p2 h w c -> n p2 (h w) c')  # q1 [[1, 64, 64, 64]]
        q2_pix = rearrange(q2, 'n p2 h w c -> n p2 (h w) c')

        kv_pix1 = self.kv_down(rearrange(kv1, 'n p2 h w c -> (n p2) c h w'))  # [[64, 128, 4, 4]]
        kv_pix2 = self.kv_down(rearrange(kv2, 'n p2 h w c -> (n p2) c h w'))
        kv_pix1 = rearrange(kv_pix1, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)  # [1, 64, 16, 128]
        kv_pix2 = rearrange(kv_pix2, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q1_win, k2_win = q1.mean([2, 3]), kv2[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk) region-level queries and keys (1,64,64)
        q2_win, k1_win = q2.mean([2, 3]), kv1[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk) region-level queries and keys (1,64,64)

        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe1 = self.lepe(rearrange(kv1[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                    i=self.n_win).contiguous())
        lepe1 = rearrange(lepe1, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
        lepe2 = self.lepe(rearrange(kv2[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe2 = rearrange(lepe2, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        r_weight1, r_idx1 = self.router(q2_win, k1_win)  # both are (n, p^2, topk)    r_weight1(1,64,k)
        r_weight2, r_idx2 = self.router(q1_win, k2_win)  # both are (n, p^2, topk)

        kv_pix_sel1 = self.kv_gather(r_idx=r_idx1, r_weight=r_weight1, kv=kv_pix1)
        k_pix_sel1, v_pix_sel1 = kv_pix_sel1.split([self.qk_dim, self.dim], dim=-1)  # k_pix_sel1 [1, 64, 4, 16, 64]  [1, 64, 16, 16, 128]

        kv_pix_sel2 = self.kv_gather(r_idx=r_idx2, r_weight=r_weight2, kv=kv_pix2)
        k_pix_sel2, v_pix_sel2 = kv_pix_sel2.split([self.qk_dim, self.dim], dim=-1)

        k_pix_sel1 = rearrange(k_pix_sel1, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                               m=self.num_heads)                        # k_pix_sel1  [64, 8, 8, 64], [64, 8, 16, 256]
        v_pix_sel1 = rearrange(v_pix_sel1, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                               m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q2_pix = rearrange(q2_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                           m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)   q2_pix [64, 8, 64, 8], [64, 8, 16, 16]
        k_pix_sel2 = rearrange(k_pix_sel2, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)
        v_pix_sel2 = rearrange(v_pix_sel2, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q1_pix = rearrange(q1_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        attn_weight1 = (q2_pix * self.scale) @ k_pix_sel1
        attn_weight1 = self.attn_act(attn_weight1)     # attn_weight1  [64, 8, 64, 64] , [64, 8, 16, 256]
        attn_weight2 = (q1_pix * self.scale) @ k_pix_sel2
        attn_weight2 = self.attn_act(attn_weight2)     # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)

        out1 = attn_weight1 @ v_pix_sel1      # out1 [64, 8, 64, 8]  [64, 8, 16, 16]
        out1 = rearrange(out1, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                         h=H // self.n_win, w=W // self.n_win)     # out1 [1, 64, 64, 64]  [1, 32, 32, 128]
        out2 = attn_weight2 @ v_pix_sel2
        out2 = rearrange(out2, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                         h=H // self.n_win, w=W // self.n_win)

        out1 = out1 + lepe1
        out2 = out2 + lepe2
        # output linear
        out1 = self.wo(out1)
        out2 = self.wo(out2)

        return out1, out2

# if __name__ == '__main__':
#
#     input1 = torch.randn(1, 128, 32, 32).cuda(0)
#     input2 = torch.randn(1, 128, 32, 32).cuda(0)
#     model = BiLevelRoutingCrossAttention(dim=128, kv_downsample_mode='ada_avgpool').cuda(0)
#     out, out_v = model(input1, input2)

