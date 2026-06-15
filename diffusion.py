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
#
#         self.time_embed = nn.Embedding(args.num_diffusion_steps, args.d_model)
#
#         self.token_embed = nn.Embedding(self.vocab_size, args.d_model)
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=args.d_model,
#             nhead=args.num_heads,
#             dim_feedforward=args.d_ff,
#             dropout=args.dropout
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
#
#         self.head = nn.Linear(args.d_model, self.vocab_size)
#
#         self.register_buffer('transition_matrix', self.build_transition_matrix())
#
#     def build_transition_matrix(self):
#         matrix = torch.ones(self.vocab_size, self.vocab_size) * 0.1 / (self.vocab_size - 1)
#         matrix.fill_diagonal_(0.9)
#         return matrix
#
#     def forward(self, x_0, att_feats, fc_feats, t):
#         batch_size, seq_len = x_0.shape
#
#         noise = torch.multinomial(
#             self.transition_matrix[x_0.flatten()],
#             num_samples=1
#         ).view(batch_size, seq_len)
#
#         mask = (torch.rand_like(x_0.float()) < (t / self.num_timesteps))
#
#         x_t = torch.where(mask, noise, x_0)
#         return x_t
#
#     def denoise_step(self, x_t, att_feats, fc_feats, t):
#
#         if isinstance(t, int):
#             t = torch.tensor([t], device=x_t.device).expand(x_t.size(0))
#         t = t.long().to(x_t.device)
#
#
#
#         att_emb = self.att_proj(att_feats.mean(1))  # [16,49,2048]->[16,512]
#         fc_emb = self.fc_proj(fc_feats)  # [16,2048]->[16,512]
#         cond = att_emb + fc_emb
#
#         t_emb = self.time_embed(t)  # [16]->[16,512]
#
#         x_emb = self.token_embed(x_t)  # [16,seq_len]->[16,seq_len,512]
#
#         x_emb = x_emb + cond.unsqueeze(1) + t_emb.unsqueeze(1)
#
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

        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid())

        self.out_norm = LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)

    # def forward(self, att_feats, fc_feats):
    #     """
    #     att_feats: [batch, num_regions, visual_dim]
    #     fc_feats: [batch, visual_dim]
    #     """
    #     att_proj = self.att_proj(att_feats)  # [B, N, D]
    #     fc_proj = self.fc_proj(fc_feats).unsqueeze(1)  # [B, 1, D]
    #
    #     context = self.cross_attn(
    #         key=att_proj,
    #         value=att_proj
    #     )  # [B, 1, D]
    #
    #     combined = torch.cat([fc_proj, context], dim=-1)
    #     gate = self.gate(combined)
    #     fused = gate * fc_proj + (1 - gate) * context
    #
    #     return self.out_drop(self.out_norm(fused.squeeze(1)))  # [B, D]

    def forward(self, att_feats, fc_feats, return_attn=False):
        """
        att_feats: [B, N, D]
        fc_feats:  [B, D]
        return_attn: whether to return the cross-attention heatmap
        """
        att_proj = self.att_proj(att_feats)  # [B, N, D]
        fc_proj = self.fc_proj(fc_feats).unsqueeze(1)  # [B, 1, D]

        context = self.cross_attn(
            query=fc_proj,
            key=att_proj,
            value=att_proj
        )  # [B, 1, D]

        # self.cross_attn.attn: [B, num_heads, 1, N]
        attn_map = None
        if return_attn and self.cross_attn.attn is not None:
            attn_map = self.cross_attn.attn.mean(dim=1).squeeze(1)  # [B, N]

        combined = torch.cat([fc_proj, context], dim=-1)
        gate = self.gate(combined)
        fused = gate * fc_proj + (1 - gate) * context

        fused_out = self.out_drop(self.out_norm(fused.squeeze(1)))  # [B, D]

        if return_attn:
            return fused_out, attn_map
        return fused_out



class VisualLanguageBridge(nn.Module):
    """
    Bridge patch-level and global visual features into compact tokens for the text generator.
    Returns:
        bridge_tokens: [B, Q, D]
        pooled_cond:   [B, D]
        attn_map:      [B, N], optional patch-level attention map
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
                attn_map = attn_map[:, 1:]

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
        self.d_model = int(d_model)
        self.rm_num_slots = int(rm_num_slots)
        self.rm_d_model = int(rm_d_model)

        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))
        self.eps = eps

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

        nn.init.xavier_uniform_(self.mlp_gamma[-1].weight, gain=0.01)
        nn.init.xavier_uniform_(self.mlp_beta[-1].weight, gain=0.01)
        nn.init.constant_(self.mlp_gamma[-1].bias, 0.)
        nn.init.constant_(self.mlp_beta[-1].bias, 0.)

    def forward(self, x, memory):
        if memory.dim() == 3:  # [B, num_slots, d_model]
            memory = memory.flatten(start_dim=1)  # [B, num_slots*d_model]

        delta_gamma = self.mlp_gamma(memory).unsqueeze(1)  # [B, 1, d_model]
        delta_beta = self.mlp_beta(memory).unsqueeze(1)  # [B, 1, d_model]

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return (self.gamma + delta_gamma) * (x - mean) / (std + self.eps) + (self.beta + delta_beta)
class DiscreteDiffusion(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.relational_memory = RelationalMemory(num_slots=3, d_model=args.d_model)
        self.vocab_size = args.vocab_size + 1
        self.num_timesteps = args.num_diffusion_steps
        self.tokenizer = tokenizer
        self.args = args

        self.cond_layer_norm = ConditionalLayerNorm(
            d_model=args.d_model,
            rm_num_slots=args.rm_num_slots,
            rm_d_model=args.rm_d_model
        )
        self.dropout = nn.Dropout(args.dropout)

        # self.visual_conditioner = VisualConditioner(
        #     visual_dim=2048,
        #     d_model=args.d_model,
        #     num_heads=args.num_heads
        # )
        # Replace the original visual conditioner.
        self.visual_conditioner = VisualConditioner(
            visual_dim=768,  # visual feature dimension; use 2048 for ResNet-101 features
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

        # Paper-aligned denoising-time token-wise global-local gate.
        # C_g = broadcast(W_g f_g), C_p = Attn(Z W_Q, F_p W_K, F_p W_V),
        # G_t = sigmoid(W_gamma [Z; C_g; C_p; E_t] + b_gamma),
        # C_t = G_t * C_g + (1 - G_t) * C_p, followed by W_C and residual LN.
        self.global_cond_proj = nn.Linear(args.d_model, args.d_model)
        self.global_local_gate = nn.Sequential(
            nn.Linear(args.d_model * 4, args.d_model),
            nn.GELU(),
            nn.Linear(args.d_model, args.d_model),
            nn.Sigmoid()
        )
        self.condition_proj = nn.Linear(args.d_model, args.d_model)
        self.condition_norm = nn.LayerNorm(args.d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 4),
            nn.GELU(),
            nn.Linear(args.d_model * 4, args.d_model),
            LayerNorm(args.d_model))

        self.time_embed = nn.Sequential(
            nn.Embedding(args.num_diffusion_steps, args.d_model),
            nn.Linear(args.d_model, args.d_model),
            nn.SiLU(),
            nn.Linear(args.d_model, args.d_model)
        )

        self.token_embed = nn.Sequential(
            nn.Embedding(self.vocab_size, args.d_model),
            nn.LayerNorm(args.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.num_heads,
            dim_feedforward=args.d_ff * 2,
            dropout=args.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_layers
        )

        self.head = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 2),
            nn.GELU(),
            nn.Linear(args.d_model * 2, self.vocab_size)
        )

        self.transition_logits = nn.Parameter(
            torch.randn(self.vocab_size, self.vocab_size) * 0.02)
        self.transition_norm = nn.Softmax(dim=-1)

        self.register_buffer('sqrt_alphas', self._create_noise_schedule('cosine'))
        self.register_buffer('sqrt_one_minus_alphas', torch.sqrt(1 - self.sqrt_alphas ** 2))

        self.pos_embed = nn.Parameter(torch.randn(1, int(getattr(args, 'max_seq_length', 100)), args.d_model))




    def _create_noise_schedule(self, schedule_type, s=0.008):
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
        return self.transition_norm(self.transition_logits)

    def forward(self, x_0, att_feats, fc_feats, t):
        """Discrete mask corruption used for training-time denoising.

        Sampling starts from all <mask> tokens. Therefore training should expose
        the denoiser to masked inputs rather than arbitrary transition-matrix
        token substitutions. Higher t means a higher mask ratio.
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # t in [0, T-1] -> mask_prob in (0, 1].
        mask_prob = (t.float() + 1.0) / float(self.num_timesteps)
        mask_prob = mask_prob.view(-1, 1).clamp(0.0, 1.0)

        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0))
        mask_id = int(getattr(self.tokenizer, "mask_token_id"))

        valid = x_0.ne(pad_id)
        corrupt = (torch.rand(batch_size, seq_len, device=device) < mask_prob) & valid

        x_t = torch.where(
            corrupt,
            torch.full_like(x_0, mask_id),
            x_0
        )
        return x_t, corrupt

    def denoise_step(self, x_t, att_feats, fc_feats, t, memory=None, return_attn=False):

        B = x_t.size(0)
        device = x_t.device

        if memory is None:
            memory = torch.zeros(B, 3, self.args.d_model, device=device)


        attn_map = None

        t_emb = self.time_embed(t)  # [B, d_model]
        t_cond = self.time_mlp(t_emb)  # [B, D]

        x_emb = self.token_embed(x_t)  # [B, seq_len, d_model]

        # Paper-aligned denoising-time token-wise global-local conditioning.
        #
        # Z_t: current token hidden states with timestep and position encoding.
        # C_g: projected global image condition, broadcast to each report token.
        # C_p: token-specific local condition from cross-attention over patch tokens.
        # G_t: token-wise and channel-wise gate computed from [Z_t; C_g; C_p; E_t].
        # C_t: gated fusion of global and local visual conditions.
        seq_len = x_emb.size(1)
        pos = self._positional_encoding(seq_len, device)

        t_cond_token = t_cond.unsqueeze(1).expand(-1, seq_len, -1)       # [B, L, D]
        z_t = x_emb + t_cond_token + pos                                 # [B, L, D]

        # W_p^{cond} F_p and W_g^{cond} f_g are implemented by the existing
        # VisualLanguageBridge projections so the local/global evidence streams
        # share the same visual projection space used by the released model.
        patch_tokens = self.visual_bridge.patch_proj(att_feats)           # [B, N, D]
        global_token = self.visual_bridge.global_proj(fc_feats)           # [B, D]

        c_g = self.global_cond_proj(global_token).unsqueeze(1)            # [B, 1, D]
        c_g = c_g.expand(-1, seq_len, -1)                                 # [B, L, D]

        c_p = self.token_visual_attn(
            query=self.token_visual_norm(z_t),
            key=patch_tokens,
            value=patch_tokens
        )                                                                # [B, L, D]

        gate_input = torch.cat([z_t, c_g, c_p, t_cond_token], dim=-1)      # [B, L, 4D]
        g_t = self.global_local_gate(gate_input)                          # [B, L, D]

        c_t = g_t * c_g + (1.0 - g_t) * c_p                               # [B, L, D]
        c_t = self.condition_proj(c_t)

        x_cond = self.condition_norm(z_t + self.token_visual_drop(c_t))   # residual injection

        if return_attn and self.token_visual_attn.attn is not None:
            # [B, H, L, N] -> [B, N], averaged over heads and report-token positions.
            attn_map = self.token_visual_attn.attn.mean(dim=1).mean(dim=1)

        trans_out = self.transformer(x_cond)  # [B, seq_len, d_model]

        memory = self.relational_memory(memory, trans_out)  # [B, slots, d_model]

        memory_summary = memory[:, -1, :].unsqueeze(1).expand(-1, x_cond.size(1), -1)  # [B, seq_len, d_model]

        # final_rep = trans_out + memory_summary  # [B, seq_len, d_model]
        trans_out_norm = self.cond_layer_norm(trans_out + memory_summary, memory)

        final_rep = trans_out + self.dropout(trans_out_norm)

        logits = self.head(final_rep)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: logits contain NaN or Inf values")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e8, neginf=-1e8)

        logits = logits / torch.max(torch.abs(logits)) * 10

        #return logits, memory
        if return_attn:
            return logits, memory, attn_map
        return logits, memory

    def _positional_encoding(self, seq_len, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.args.d_model, 2, device=device) *
                             (-math.log(10000.0) / self.args.d_model))
        pe = torch.zeros(1, seq_len, self.args.d_model, device=device)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

# class VisualConditioner(nn.Module):
#
#     def __init__(self, visual_dim, d_model, num_heads):
#         super().__init__()
#         self.att_proj = nn.Linear(visual_dim, d_model)
#         self.att_norm = nn.LayerNorm(d_model)
#
#         self.fc_proj = nn.Linear(visual_dim, d_model)
#         self.fc_norm = nn.LayerNorm(d_model)
#
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             batch_first=True
#         )
#
#     def forward(self, att_feats, fc_feats):
#         att_emb = self.att_norm(self.att_proj(att_feats.mean(1)))
#
#         fc_emb = self.fc_norm(self.fc_proj(fc_feats))
#
#         cond, _ = self.cross_attn(
#             query=fc_emb.unsqueeze(1),
#             key=att_emb.unsqueeze(1),
#             value=att_emb.unsqueeze(1)
#         )
#         return cond.squeeze(1)

class TransformerDenoiser(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.embedding = nn.Embedding(args.vocab_size + 1, args.d_model)
        self.time_embed = nn.Embedding(args.num_diffusion_steps, args.d_model)

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
        x_emb = self.embedding(x_t)  # [batch, seq_len, d_model]
        t_emb = self.time_embed(t)  # [batch, d_model]

        x_emb = x_emb + image_emb.unsqueeze(1) + t_emb.unsqueeze(1)

        x_out = self.transformer(x_emb)

        return self.head(x_out)
