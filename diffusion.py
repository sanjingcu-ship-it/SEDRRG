import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# class DiscreteDiffusion(nn.Module):
#     def __init__(self, args, tokenizer):
#         super().__init__()
#         self.vocab_size = args.vocab_size + 1
#         self.num_timesteps = args.num_diffusion_steps
#         self.tokenizer = tokenizer
#
#         # 1. 视觉特征投影
#         self.att_proj = nn.Linear(2048, args.d_model)  # att_feats投影
#         self.fc_proj = nn.Linear(2048, args.d_model)  # fc_feats投影
#
#         # 2. 新增时间步嵌入层
#         self.time_embed = nn.Embedding(args.num_diffusion_steps, args.d_model)
#
#         # 3. Token嵌入层
#         self.token_embed = nn.Embedding(self.vocab_size, args.d_model)
#
#         # 4. 去噪网络
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=args.d_model,
#             nhead=args.num_heads,
#             dim_feedforward=args.d_ff,
#             dropout=args.dropout
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
#
#         # 5. 输出头
#         self.head = nn.Linear(args.d_model, self.vocab_size)
#
#         # 6. 转移矩阵
#         self.register_buffer('transition_matrix', self.build_transition_matrix())
#
#     def build_transition_matrix(self):
#         matrix = torch.ones(self.vocab_size, self.vocab_size) * 0.1 / (self.vocab_size - 1)
#         matrix.fill_diagonal_(0.9)
#         return matrix
#
#     def forward(self, x_0, att_feats, fc_feats, t):
#         """前向扩散过程：对输入 x_0 添加噪声"""
#         batch_size, seq_len = x_0.shape
#
#         # 采样噪声
#         noise = torch.multinomial(
#             self.transition_matrix[x_0.flatten()],
#             num_samples=1
#         ).view(batch_size, seq_len)
#
#         # 修正：确保 t 的形状能与 x_0 比较
#         t = t.view(-1, 1)  # 从 [batch_size] 变为 [batch_size, 1]
#         mask = (torch.rand_like(x_0.float()) < (t / self.num_timesteps))
#
#         x_t = torch.where(mask, noise, x_0)
#         return x_t
#
#     def denoise_step(self, x_t, att_feats, fc_feats, t):
#         """完整修正后的去噪步骤"""
#
#         if isinstance(t, int):
#             t = torch.tensor([t], device=x_t.device).expand(x_t.size(0))
#         t = t.long().to(x_t.device)
#
#         assert t.dim() == 1, f"时间步t应为1D张量，得到{t.shape}"
#         assert t.size(0) == x_t.size(0), "批次大小不匹配"
#
#         # 输入验证
#         assert x_t.dim() == 2, f"x_t应为2D张量，实际得到{x_t.shape}"
#         # assert t.dim() == 1, f"t应为1D张量，实际得到{t.shape}"
#
#         # 1. 视觉条件处理
#         att_emb = self.att_proj(att_feats.mean(1))  # [16,49,2048]->[16,512]
#         fc_emb = self.fc_proj(fc_feats)  # [16,2048]->[16,512]
#         cond = att_emb + fc_emb
#
#         # 2. 时间步嵌入
#         t_emb = self.time_embed(t)  # [16]->[16,512]
#
#         # 3. Token嵌入
#         x_emb = self.token_embed(x_t)  # [16,seq_len]->[16,seq_len,512]
#
#         # 4. 合并条件
#         x_emb = x_emb + cond.unsqueeze(1) + t_emb.unsqueeze(1)
#
#         # 5. Transformer处理
#         logits = self.transformer(x_emb)  # [16,seq_len,512]
#         logits = self.head(logits)  # [16,seq_len,vocab_size]
#
#         return logits
from modules.encoder_decoder import RelationalMemory
import math
import copy
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
class VisualConditioner(nn.Module):
    def __init__(self, visual_dim, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 视觉特征投影
        self.att_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.fc_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout))

        # 多模态注意力
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid())

        # 输出归一化
        self.out_norm = LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)

    # def forward(self, att_feats, fc_feats):
    #     """
    #     att_feats: [batch, num_regions, visual_dim]
    #     fc_feats: [batch, visual_dim]
    #     """
    #     # 投影视觉特征
    #     att_proj = self.att_proj(att_feats)  # [B, N, D]
    #     fc_proj = self.fc_proj(fc_feats).unsqueeze(1)  # [B, 1, D]
    #
    #     # 跨注意力机制
    #     context = self.cross_attn(
    #         query=fc_proj,  # 用全局特征作为查询
    #         key=att_proj,
    #         value=att_proj
    #     )  # [B, 1, D]
    #
    #     # 门控融合
    #     combined = torch.cat([fc_proj, context], dim=-1)
    #     gate = self.gate(combined)
    #     fused = gate * fc_proj + (1 - gate) * context
    #
    #     # 输出处理
    #     return self.out_drop(self.out_norm(fused.squeeze(1)))  # [B, D]

    def forward(self, att_feats, fc_feats, return_attn=False):
        """
        att_feats: [B, N, D]
        fc_feats:  [B, D]
        return_attn: 是否返回 cross-attention 热图
        """
        # 1) 投影视觉特征
        att_proj = self.att_proj(att_feats)  # [B, N, D]
        fc_proj = self.fc_proj(fc_feats).unsqueeze(1)  # [B, 1, D]

        # 2) 跨注意力
        context = self.cross_attn(
            query=fc_proj,
            key=att_proj,
            value=att_proj
        )  # [B, 1, D]

        # 3) 从 MultiHeadedAttention 里取出注意力权重
        # self.cross_attn.attn: [B, num_heads, 1, N]
        attn_map = None
        if return_attn and self.cross_attn.attn is not None:
            attn_map = self.cross_attn.attn.mean(dim=1).squeeze(1)  # [B, N]

        # 4) 门控融合
        combined = torch.cat([fc_proj, context], dim=-1)
        gate = self.gate(combined)
        fused = gate * fc_proj + (1 - gate) * context

        # 5) 输出处理
        fused_out = self.out_drop(self.out_norm(fused.squeeze(1)))  # [B, D]

        if return_attn:
            return fused_out, attn_map
        return fused_out



class VisualLanguageBridge(nn.Module):
    """
    将 patch/global 视觉特征桥接成一小组更适合文本生成器使用的 bridge tokens。
    返回:
        bridge_tokens: [B, Q, D]
        pooled_cond:   [B, D]
        attn_map:      [B, N] (可选，仅 patch 部分)
    """
    def __init__(self, visual_dim, d_model, num_heads, num_query_tokens=8, dropout=0.1):
        super().__init__()
        self.num_query_tokens = num_query_tokens

        self.patch_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.global_proj = nn.Sequential(
            nn.Linear(visual_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)

    def forward(self, att_feats, fc_feats, return_attn=False):
        patch_tokens = self.patch_proj(att_feats)                 # [B, N, D]
        global_token = self.global_proj(fc_feats).unsqueeze(1)    # [B, 1, D]
        visual_tokens = torch.cat([global_token, patch_tokens], dim=1)  # [B, 1+N, D]

        queries = self.query_tokens.expand(att_feats.size(0), -1, -1)
        queries = self.query_norm(queries)

        bridge_tokens = self.cross_attn(
            query=queries,
            key=visual_tokens,
            value=visual_tokens
        )  # [B, Q, D]

        attn_map = None
        if return_attn and self.cross_attn.attn is not None:
            # [B, H, Q, 1+N] -> [B, 1+N]
            attn_map = self.cross_attn.attn.mean(dim=1).mean(dim=1)
            if attn_map.size(1) == att_feats.size(1) + 1:
                attn_map = attn_map[:, 1:]  # 去掉 global token 对应位置，仅保留 patch 热图

        pooled_bridge = bridge_tokens.mean(dim=1)
        pooled_global = global_token.squeeze(1)

        gate = self.gate(torch.cat([pooled_bridge, pooled_global], dim=-1))
        pooled_cond = gate * pooled_bridge + (1.0 - gate) * pooled_global

        bridge_tokens = self.out_drop(self.out_norm(bridge_tokens))
        pooled_cond = self.out_drop(self.out_norm(pooled_cond))

        if return_attn:
            return bridge_tokens, pooled_cond, attn_map
        return bridge_tokens, pooled_cond


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super().__init__()
        # 确保参数是整数
        self.d_model = int(d_model)
        self.rm_num_slots = int(rm_num_slots)
        self.rm_d_model = int(rm_d_model)

        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

        # 更稳定的MLP设计
        self.mlp_gamma = nn.Sequential(
            nn.Linear(self.rm_num_slots * self.rm_d_model, self.rm_d_model),
            nn.LayerNorm(self.rm_d_model),
            nn.GELU(),
            nn.Linear(self.rm_d_model, self.d_model)
        )

        self.mlp_beta = nn.Sequential(
            nn.Linear(self.rm_num_slots * self.rm_d_model, self.rm_d_model),
            nn.LayerNorm(self.rm_d_model),
            nn.GELU(),
            nn.Linear(self.rm_d_model, self.d_model)
        )

        # 初始化
        nn.init.xavier_uniform_(self.mlp_gamma[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.mlp_beta[-1].weight, gain=0.01)
        nn.init.constant_(self.mlp_gamma[-1].bias, 0.)
        nn.init.constant_(self.mlp_beta[-1].bias, 0.)

    def forward(self, x, memory):
        # 输入形状处理
        if memory.dim() == 3:  # [B, num_slots, d_model]
            memory = memory.flatten(start_dim=1)  # [B, num_slots*d_model]

        # 计算delta
        delta_gamma = self.mlp_gamma(memory).unsqueeze(1)  # [B, 1, d_model]
        delta_beta = self.mlp_beta(memory).unsqueeze(1)  # [B, 1, d_model]

        # 标准化
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return (self.gamma + delta_gamma) * (x - mean) / (std + self.eps) + (self.beta + delta_beta)
class DiscreteDiffusion(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.relational_memory = RelationalMemory(num_slots=3, d_model=args.d_model)
        self.vocab_size = args.vocab_size + 1  # 包含[PAD]
        self.num_timesteps = args.num_diffusion_steps
        self.tokenizer = tokenizer
        self.args = args

        self.cond_layer_norm = ConditionalLayerNorm(
            d_model=args.d_model,
            rm_num_slots=args.rm_num_slots,
            rm_d_model=args.rm_d_model
        )
        self.dropout = nn.Dropout(args.dropout)

        # # 改进1: 更灵活的特征投影
        # self.visual_conditioner = VisualConditioner(
        #     visual_dim=2048,
        #     d_model=args.d_model,
        #     num_heads=args.num_heads
        # )
        # 替换原有的visual_conditioner
        self.visual_conditioner = VisualConditioner(
            visual_dim=768,  # 假设视觉特征维度#如果是res101就换成2048
            d_model=args.d_model,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
        self.visual_bridge = VisualLanguageBridge(
            visual_dim=768,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_query_tokens=getattr(args, "bridge_num_queries", 8),
            dropout=args.dropout
        )
        self.token_visual_attn = MultiHeadedAttention(
            args.num_heads,
            args.d_model,
            args.dropout
        )
        self.token_visual_norm = nn.LayerNorm(args.d_model)
        self.token_visual_drop = nn.Dropout(args.dropout)
        self.time_mlp = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 4),
            nn.GELU(),
            nn.Linear(args.d_model * 4, args.d_model),
            LayerNorm(args.d_model))

        # 改进2: 增强的时间步处理
        self.time_embed = nn.Sequential(
            nn.Embedding(args.num_diffusion_steps, args.d_model),
            nn.Linear(args.d_model, args.d_model),
            nn.SiLU(),
            nn.Linear(args.d_model, args.d_model)
        )

        # 改进3: 更丰富的token嵌入
        self.token_embed = nn.Sequential(
            nn.Embedding(self.vocab_size, args.d_model),
            nn.LayerNorm(args.d_model)
        )

        # 改进4: 增强的Transformer架构
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.num_heads,
            dim_feedforward=args.d_ff * 2,  # 扩大FFN维度
            dropout=args.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 前置LayerNorm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_layers
        )

        # 改进5: 自适应输出头
        self.head = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 2),
            nn.GELU(),
            nn.Linear(args.d_model * 2, self.vocab_size)
        )

        # 改进6: 可学习的转移矩阵
        self.transition_logits = nn.Parameter(
            torch.randn(self.vocab_size, self.vocab_size) * 0.02)
        self.transition_norm = nn.Softmax(dim=-1)

        # 改进7: 噪声调度注册
        self.register_buffer('sqrt_alphas', self._create_noise_schedule('cosine'))
        self.register_buffer('sqrt_one_minus_alphas', torch.sqrt(1 - self.sqrt_alphas ** 2))

        self.pos_embed = nn.Parameter(torch.randn(1, 60, args.d_model))
        # 在DiscreteDiffusion的__init__中




    def _create_noise_schedule(self, schedule_type, s=0.008):
        """创建噪声调度表"""
        steps = self.num_timesteps
        if schedule_type == 'linear':
            return torch.linspace(1 - s, s, steps)
        elif schedule_type == 'cosine':
            x = torch.linspace(0, steps, steps + 1)
            alphas = torch.cos((x / steps + s) / (1 + s) * math.pi / 2) ** 2
            return alphas[:-1] / alphas[0]
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    @property
    def transition_matrix(self):
        """可学习的转移矩阵"""
        return self.transition_norm(self.transition_logits)

    def forward(self, x_0, att_feats, fc_feats, t):
        """改进的扩散过程 - 融入视觉条件"""
        batch_size, seq_len = x_0.shape

        # 1. 获取视觉条件（与denoise_step一致）
        _, cond = self.visual_bridge(att_feats, fc_feats, return_attn=False)  # [batch, d_model]

        # 2. 条件相关的噪声采样
        with torch.no_grad():
            # 将条件信息融入转移概率
            token_embedding_weight = self.token_embed[0].weight  # 从Sequential中取出Embedding层
            cond_weights = torch.sigmoid(cond @ token_embedding_weight.t())

            transition_probs = self.transition_matrix[x_0.flatten()]  # [batch*seq_len, vocab_size]

            # 调整转移概率（偏向与视觉条件相关的token）
            adjusted_probs = transition_probs * cond_weights.repeat(1, seq_len).view(-1, self.vocab_size)
            adjusted_probs = adjusted_probs / adjusted_probs.sum(-1, keepdim=True)

            noise = torch.multinomial(adjusted_probs, num_samples=1).view(batch_size, seq_len)

        # 3. 基于调度表的噪声混合
        sqrt_alpha = self.sqrt_alphas[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas[t].view(-1, 1)

        # 视觉条件感知的噪声混合
        cond_strength = torch.sigmoid(cond.mean(-1, keepdim=True))  # [batch, 1]
        noise_ratio = sqrt_one_minus_alpha * (0.8 + 0.2 * cond_strength)  # 条件越强，保留更多原始信息

        mask = (torch.rand_like(x_0.float()) < noise_ratio)
        x_t = torch.where(mask, noise, x_0)

        return x_t

    def denoise_step(self, x_t, att_feats, fc_feats, t, memory=None, return_attn=False):

        B = x_t.size(0)
        device = x_t.device

        # 1. 初始化记忆
        if memory is None:
            memory = torch.zeros(B, 3, self.args.d_model, device=device)

        # 2. 调整视觉特征维度
        # cond = self.visual_conditioner(att_feats, fc_feats)  # [B, 2048]
        # cond = cond[:, :self.args.d_model]  # 截取前d_model维 [B, d_model]

        # visual_cond = self.visual_conditioner(att_feats, fc_feats)  # [B, D]
        if return_attn:
            bridge_tokens, visual_cond, attn_map = self.visual_bridge(
                att_feats, fc_feats, return_attn=True
            )
        else:
            bridge_tokens, visual_cond = self.visual_bridge(
                att_feats, fc_feats, return_attn=False
            )
            attn_map = None

        # 3. 时间步嵌入
        t_emb = self.time_embed(t)  # [B, d_model]
        t_cond = self.time_mlp(t_emb)  # [B, D]

        # 4. Token嵌入
        x_emb = self.token_embed(x_t)  # [B, seq_len, d_model]

        # 5. pooled visual condition + time condition
        combined_cond = visual_cond + t_cond
        x_cond = x_emb + combined_cond.unsqueeze(1)

        # 6. 位置编码
        x_cond = x_cond + self._positional_encoding(x_cond.size(1), device)

        # 7. token-level visual cross-attention
        x_visual = self.token_visual_attn(
            query=self.token_visual_norm(x_cond),
            key=bridge_tokens,
            value=bridge_tokens
        )
        x_cond = x_cond + self.token_visual_drop(x_visual)

        # 8. 使用Transformer提取特征
        trans_out = self.transformer(x_cond)  # [B, seq_len, d_model]

        # 7c. 使用RelationalMemory进一步处理序列上下文
        memory = self.relational_memory(memory, trans_out)  # [B, slots, d_model]

        # 7d. 将记忆拼接或融合回输出
        # 方式1：取memory最后一个slot，拼接
        memory_summary = memory[:, -1, :].unsqueeze(1).expand(-1, x_cond.size(1), -1)  # [B, seq_len, d_model]

        # 7e. 合并transformer输出和记忆
        # final_rep = trans_out + memory_summary  # [B, seq_len, d_model]
        # 修改denoise_step中的特征处理
        # 直接应用条件层归一化
        trans_out_norm = self.cond_layer_norm(trans_out + memory_summary, memory)

        # 添加残差连接和dropout
        final_rep = trans_out + self.dropout(trans_out_norm)

        # Output predictions.
        logits = self.head(final_rep)

        # 添加logits检查
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("警告：logits包含NaN或Inf值")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e8, neginf=-1e8)

        # 添加logits归一化
        logits = logits / torch.max(torch.abs(logits)) * 10  # 将logits范围限制在[-10,10]之间

        #return logits, memory
        if return_attn:
            return logits, memory, attn_map
        return logits, memory

    def _positional_encoding(self, seq_len, device):
        """正弦位置编码"""
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.args.d_model, 2, device=device) *
                             (-math.log(10000.0) / self.args.d_model))
        pe = torch.zeros(1, seq_len, self.args.d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

# class VisualConditioner(nn.Module):
#     """改进的视觉条件处理器"""
#
#     def __init__(self, visual_dim, d_model, num_heads):
#         super().__init__()
#         # 注意力特征处理
#         self.att_proj = nn.Linear(visual_dim, d_model)
#         self.att_norm = nn.LayerNorm(d_model)
#
#         # 全局特征处理
#         self.fc_proj = nn.Linear(visual_dim, d_model)
#         self.fc_norm = nn.LayerNorm(d_model)
#
#         # 交叉注意力融合
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             batch_first=True
#         )
#
#     def forward(self, att_feats, fc_feats):
#         # 处理注意力特征
#         att_emb = self.att_norm(self.att_proj(att_feats.mean(1)))
#
#         # 处理全局特征
#         fc_emb = self.fc_norm(self.fc_proj(fc_feats))
#
#         # 交叉注意力融合
#         cond, _ = self.cross_attn(
#             query=fc_emb.unsqueeze(1),
#             key=att_emb.unsqueeze(1),
#             value=att_emb.unsqueeze(1)
#         )
#         return cond.squeeze(1)

class TransformerDenoiser(nn.Module):
    """去噪网络：基于 Transformer 的条件扩散模型"""

    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size + 1, args.d_model)
        self.time_embed = nn.Embedding(args.num_diffusion_steps, args.d_model)

        # Transformer 层（简化版）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.num_heads,
                dim_feedforward=args.d_ff,
                dropout=args.dropout
            ),
            num_layers=args.num_layers
        )

        self.head = nn.Linear(args.d_model, args.vocab_size + 1)

    def forward(self, x_t, image_emb, t):
        # 输入嵌入
        x_emb = self.embedding(x_t)  # [batch, seq_len, d_model]
        t_emb = self.time_embed(t)  # [batch, d_model]

        # 融合视觉条件 + 时间步
        x_emb = x_emb + image_emb.unsqueeze(1) + t_emb.unsqueeze(1)

        # Transformer 处理
        x_out = self.transformer(x_emb)

        # Predict logits.
        return self.head(x_out)
