import torch
from torch import nn, einsum
from einops import rearrange, reduce
import math
import torch.nn.functional as F
from inspect import isfunction


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, topic):
        return self.fn(x, topic) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, topic):
        x = self.norm(x)
        return self.fn(x, topic)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, ctx_dim=None, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        ctx_dim = default(ctx_dim, dim)

        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(ctx_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(ctx_dim, hidden_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        """
        x: [batch, seq_len, dim]
        context: [batch, context_len, ctx_dim]
        """
        context = default(context, x)

        assert x.shape[-1] == self.to_q.in_features, \
            f"Input dim {x.shape[-1]} != query dim {self.to_q.in_features}"

        q = self.to_q(x)  # [b,seq_len,hidden_dim]
        k = self.to_k(context)  # [b,context_len,hidden_dim]
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class BasicTransformerBlock(nn.Module):
    def __init__(self, max_sent, max_word, words_emb_dim, hidden_dim):
        super(BasicTransformerBlock, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.feedback = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.catt = Residual(PreNorm(hidden_dim, CrossAttention(max_sent * max_word, words_emb_dim)))

    def forward(self, pack):
        x, topic = pack
        x = self.catt(x, topic)
        x = self.catt(x, topic)
        return (self.feedback(self.layernorm(x)), topic)


class SpatialTransformer(nn.Module):
    def __init__(self, max_sent, max_word, words_emb_dim, hidden_dim, block_num):
        super(SpatialTransformer, self).__init__()
        self.transformer = nn.Sequential(
            *[BasicTransformerBlock(max_sent, max_word, words_emb_dim, hidden_dim) for _ in range(block_num)]
        )

    def forward(self, x, topic):
        return self.transformer((x, topic))[0]


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim)
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        # x = self.dropout(x)
        return x


class FiLM(nn.Module):
    def __init__(self, time_emb_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, 2 * hidden_dim),
            nn.GELU()
        )

    def forward(self, x, time_emb):
        gamma_beta = self.mlp(time_emb)  # (batch_size, 2 * hidden_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        gamma = gamma.unsqueeze(1).expand(-1, x.shape[1], -1)  # (batch_size, seq_len, hidden_dim)
        beta = beta.unsqueeze(1).expand(-1, x.shape[1], -1)  # (batch_size, seq_len, hidden_dim)

        return x * (gamma + 1) + beta

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=4, alpha=8):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank

        for param in base_layer.parameters():
            param.requires_grad = False

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = x @ self.lora_A @ self.lora_B * self.scaling
        return base_output + lora_output


class EnhancedLSTMWithTimeEmb(nn.Module):
    def __init__(self, max_sent, max_word, time_emb_dim, words_emb_dim, hidden_dim):
        super().__init__()
        self.time_emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim * 2)
        )
        self.proj_dim = nn.Linear(time_emb_dim, hidden_dim)
        super(EnhancedLSTMWithTimeEmb, self).__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )



        self.learned_sinusoidal_cond = None
        self.self_condition = False
        self.film = FiLM(time_emb_dim, hidden_dim)

        self.time_emb_dim = time_emb_dim
        sinu_pos_emb = SinusoidalPosEmb(self.time_emb_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=6)

        self.time_emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2)
        )
        self.proj_dim = nn.Linear(time_emb_dim, hidden_dim)
        self.res_catt = Residual(PreNorm(hidden_dim, CrossAttention(max_sent * max_word, words_emb_dim)))
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.o_fc = nn.Linear(words_emb_dim, words_emb_dim)
        # self.transformer = SpatialTransformer(max_sent, max_word, words_emb_dim, hidden_dim, 16)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.mlp1 = nn.Sequential(
            *[MixerBlock(max_sent * max_word, hidden_dim, 512, 512) for _ in range(16)])
        self.mlp2 = nn.Sequential(
            *[MixerBlock(max_sent * max_word, hidden_dim, 512, 512) for _ in range(16)])
        self.fc1 = nn.Linear(words_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, words_emb_dim)

        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)  # or whatever dimensions you need
        # self.proj = LoRALayer(nn.Linear(hidden_dim, time_emb_dim))

    def time_embedding(self, x, time_emb):  # (b sw d) (b d)
        time_emb = self.time_emb_mlp(time_emb)
        # time_emb = rearrange(time_emb, 'b c -> b 1 c')
        scale, shift = time_emb.chunk(2, dim=1)
        # h = self.layernorm(x)
        # h = h * (scale + 1) + shift
        # return x + h  # (b sw d)
        scale = self.proj_dim(scale).unsqueeze(1)  # [batch, 1, hidden_dim]
        shift = self.proj_dim(shift).unsqueeze(1)

        return x * (scale + 1) + shift

    def lstm_with_sum(self, x, lstm):  # (b sw d)
        # s = x.shape[1]
        # x = rearrange(x, 'b s w d -> b (s w) d')
        x, _ = lstm(x)
        # x = rearrange(x, 'b (s w) d -> b s w d', s=s)
        return x

    def forward(self, x, t, topic):  # (b s w d2) (b) (b n1 d1)
        # r = x.clone()
        visual_feat = self.visual_proj(topic)  # [16,64,256] -> [16,64,hidden_dim]

        time_emb = self.time_mlp(t)
        x = self.time_embedding(x, time_emb)

        attn_out, _ = self.cross_attn(
            query=x,
            key=visual_feat,
            value=visual_feat
        )

        x = x + 0.3 * attn_out

        time_emb = self.time_mlp(t)  # f
        b, s, w, d = x.shape
        x = rearrange(x, 'b s w d -> b (s w) d')
        x = self.fc1(x)  # word_dim -> hidden_dim

        x = x + self.time_proj(time_emb).unsqueeze(1)

        # x = self.proj(x)

        x = self.transformer(x)

        l1 = x.clone()

        x = self.lstm_with_sum(self.film(x, time_emb), self.lstm1)  # (b sw d)
        l2 = x.clone()
        x = self.mlp1(self.res_catt(self.film(x, time_emb), topic))  # (b sw d)
        l3 = x.clone()
        # x = self.res_catt(self.time_embedding(x, time_emb), topic)
        # x = self.transformer(self.film(x, time_emb), topic)
        x = self.mlp2(self.res_catt(self.film(x, time_emb) + l3, topic))
        x = self.lstm_with_sum(self.film(x, time_emb) + l2, self.lstm5)  # (b sw d)
        x = self.fc2(self.film(x, time_emb) + l1)
        x = rearrange(x, 'b (s w) d -> b s w d', s=s)
        return self.o_fc(x)


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, topic):
        x = self.norm(x)
        topic_pooled = topic.mean(dim=1)  # [b,d]
        gamma = self.scale * topic_pooled.unsqueeze(1).unsqueeze(1) + self.bias
        beta = gamma
        return x * (gamma + 1) + beta
class LSTM_with_timeemb(nn.Module):
    def __init__(self, max_sent, max_word, time_emb_dim, words_emb_dim, hidden_dim):
        super(LSTM_with_timeemb, self).__init__()
        self.adapt_norm1 = AdaptiveLayerNorm(hidden_dim)
        # self.adapt_norm2 = AdaptiveLayerNorm(hidden_dim)

        self.learned_sinusoidal_cond = None
        self.self_condition = False
        self.film = FiLM(time_emb_dim, hidden_dim)

        self.time_emb_dim = time_emb_dim
        sinu_pos_emb = SinusoidalPosEmb(self.time_emb_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=6)



        self.time_emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim * 2)
        )
        self.res_catt = Residual(PreNorm(hidden_dim, CrossAttention(max_sent * max_word, words_emb_dim)))
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.o_fc = nn.Linear(words_emb_dim, words_emb_dim)
        # self.transformer = SpatialTransformer(max_sent, max_word, words_emb_dim, hidden_dim, 16)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.mlp1 = nn.Sequential(
            *[MixerBlock(max_sent * max_word, hidden_dim, 512, 512) for _ in range(16)])
        self.mlp2 = nn.Sequential(
            *[MixerBlock(max_sent * max_word, hidden_dim, 512, 512) for _ in range(16)])
        self.fc1 = nn.Linear(words_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, words_emb_dim)

        self.time_proj = nn.Linear(time_emb_dim, hidden_dim)  # or whatever dimensions you need
        # self.proj = LoRALayer(nn.Linear(hidden_dim, time_emb_dim))

    def time_embedding(self, x, time_emb):  # (b sw d) (b d)
        time_emb = self.time_emb_mlp(time_emb)
        time_emb = rearrange(time_emb, 'b c -> b 1 c')
        scale, shift = time_emb.chunk(2, dim=2)
        h = self.layernorm(x)
        h = h * (scale + 1) + shift
        return x + h  # (b sw d)

    def lstm_with_sum(self, x, lstm):  # (b sw d)
        # s = x.shape[1]
        # x = rearrange(x, 'b s w d -> b (s w) d')
        x, _ = lstm(x)
        # x = rearrange(x, 'b (s w) d -> b s w d', s=s)
        return x

    def forward(self, x, t, topic):  # (b s w d2) (b) (b n1 d1)
        # print(x.shape)
        x = self.adapt_norm1(x, topic)
        # print(x.shape)
        time_emb = self.time_mlp(t)  # f
        b, s, w, d = x.shape
        # x = rearrange(x, 'b s w d -> b (s w) d')
        x = self.fc1(x)  # word_dim -> hidden_dim
        # print(x.shape)
        x = self.adapt_norm1(x, topic)
        # print(x.shape)
        x = rearrange(x, 'b s w d -> b (s w) d')


        x = x + self.time_proj(time_emb).unsqueeze(1)

        # x = self.proj(x)

        x = self.transformer(x)
        # print(x.shape)

        l1 = x.clone()

        x = self.lstm_with_sum(self.film(x, time_emb), self.lstm1)  # (b sw d)
        l2 = x.clone()
        x = self.mlp1(self.res_catt(self.film(x, time_emb), topic))  # (b sw d)
        l3 = x.clone()
        # x = self.res_catt(self.time_embedding(x, time_emb), topic)
        # x = self.transformer(self.film(x, time_emb), topic)
        x = self.mlp2(self.res_catt(self.film(x, time_emb) + l3, topic))
        x = self.lstm_with_sum(self.film(x, time_emb) + l2, self.lstm5)  # (b sw d)
        x = self.fc2(self.film(x, time_emb) + l1)
        x = rearrange(x, 'b (s w) d -> b s w d', s=s)
        return self.o_fc(x)


# class Embeddings(nn.Module):
#     def __init__(self, vocab_size, words_emb_dim, num_sent):
#         super(Embeddings, self).__init__()
#         self.emb = nn.Embedding(vocab_size, words_emb_dim)
#         self.norm = nn.GroupNorm(1, num_sent)
#
#     def forward(self, x):
#         return self.norm(self.emb(x))

class MCLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.mlp_gamma = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.mlp_beta = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())

    def forward(self, x, memory):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        gamma = self.gamma + self.mlp_gamma(memory)
        beta = self.beta + self.mlp_beta(memory)
        return gamma * (x - mean) / (std + 1e-5) + beta



class RelationalMemory(nn.Module):
    def __init__(self, mem_slots=3, head_size=64, input_size=256):
        super().__init__()
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.mem_size = 256
        self.input_size = input_size

        self.Wq = nn.Linear(self.mem_size, self.mem_size)
        self.Wk = nn.Linear(self.mem_size + input_size, self.mem_size)
        self.Wv = nn.Linear(self.mem_size + input_size, self.mem_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.mem_size, self.mem_size),
            nn.ReLU(),
            nn.Linear(self.mem_size, self.mem_size)
        )

        self.Uf = nn.Linear(self.mem_size, self.mem_size)
        self.Ui = nn.Linear(self.mem_size, self.mem_size)
        self.Wf = nn.Linear(input_size, self.mem_size)
        self.Wi = nn.Linear(input_size, self.mem_size)

    def forward(self, memory, y_prev):  # memory: [B, S, D], y_prev: [B, D]
        B, S, D = memory.shape
        y_expand = y_prev.unsqueeze(1).expand(-1, S, -1)  # [B, S, D]

        q = self.Wq(memory)  # [B, S, D]
        kv_input = torch.cat([memory, y_expand], dim=-1)
        k = self.Wk(kv_input)
        v = self.Wv(kv_input)

        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5), dim=-1)
        Z = torch.matmul(attn, v)

        M_tilde = self.mlp(Z + memory) + Z + memory

        Gf = torch.sigmoid(self.Wf(y_expand) + self.Uf(memory))
        Gi = torch.sigmoid(self.Wi(y_expand) + self.Ui(memory))

        memory_updated = Gf * memory + Gi * torch.tanh(M_tilde)
        return memory_updated


#########################################418￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥
class ImprovedDenoiser(nn.Module):
    def __init__(self,
                 max_sent=4,
                 max_word=32,
                 hidden_dim=256,
                 time_emb_dim=256,
                 num_heads=8,
                 dropout=0.1,
                 use_img_feats=True):
        super().__init__()

        self.relational_memory = RelationalMemory(mem_slots=3, input_size=hidden_dim)

        self.max_sent = max_sent
        self.max_word = max_word
        self.seq_len = max_sent * max_word  # 4*32=128
        self.use_img_feats = use_img_feats

        self.init_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, hidden_dim)
        )

        self.topic_proj = nn.Linear(hidden_dim, hidden_dim)

        if self.use_img_feats:
            self.img_proj = nn.Linear(512, hidden_dim)
        else:
            self.img_proj = None

        self.condition_fusion = nn.ModuleList([
            ConditionBlock(hidden_dim, num_heads, dropout)
            for _ in range(3)
        ])

        self.noise_layers = nn.ModuleList([
            DenoiseBlock(hidden_dim, hidden_dim)
            for _ in range(6)
        ])

        self.skip_weight_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.final_norm = AdaptiveLayerNorm1(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, t, topic=None, img_feats=None):

        """
        x: [batch, num_sentences, num_words, hidden_dim]
           e.g. [16,4,32,256]
        t: [batch] e.g. [16]
        """
        memory = torch.zeros(16, 3, 256).to(x.device)  # [B, 3, 256]

        b, s, w, d = x.shape
        h = rearrange(x, 'b s w d -> b (s w) d')  # [16,128,256]

        h = self.init_proj(h)
        time_emb = self.time_mlp(t)  # [16,256]
        h = h + time_emb.unsqueeze(1)  # [16,128,256]

        conditions = self._prepare_conditions(topic, img_feats)
        for fuse_layer in self.condition_fusion:
            h = fuse_layer(h, conditions, time_emb)

        skip_weight = self.skip_weight_proj(time_emb)  # [16,256]->[16,256]
        h_final = 0
        for layer in self.noise_layers:
            memory = self.relational_memory(memory, h.mean(dim=1))  # h: [B, 128, 256]
            memory_cond = memory.mean(dim=1).unsqueeze(1).expand(-1, h.shape[1], -1)  # [B, 128, 256]
            h = h + memory_cond

            h = layer(h, time_emb)
            h_final = h_final + h * skip_weight.unsqueeze(1)

        out = self.final_norm(h_final, time_emb)
        out = self.output_proj(out)
        # print(out.shape)
        return rearrange(out, 'b (s w) d -> b s w d', s=4)
    def _prepare_conditions(self, topic, img_feats):
        conditions = []
        if topic is not None:
            conditions.append(self.topic_proj(topic.mean(1)))  # [b,64,256]->[b,256]
        if img_feats is not None and self.img_proj:
            conditions.append(self.img_proj(img_feats.flatten(1)))  # [b,512]->[b,256]
        return torch.stack(conditions).mean(0) if conditions else None


class ConditionBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.cross_attn = CrossAttention(
            dim=hidden_dim,
            ctx_dim=hidden_dim,
            heads=num_heads
        )
        self.gate = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond, time_emb):
        if cond is None:
            return x
        if cond.dim() == 2:  # [b,d]
            cond = cond.unsqueeze(1)  # [b,1,d]
        elif cond.dim() == 3 and cond.size(1) > 1:  # [b,n,d]
            cond = cond.mean(dim=1, keepdim=True)

        return self.cross_attn(x, cond)



        # attn_out = self.cross_attn(x, cond)
        # gate = torch.sigmoid(self.gate(torch.cat([x, attn_out], dim=-1)))
        # return self.dropout(gate * attn_out + (1 - gate) * x)


class DenoiseBlock(nn.Module):
    def __init__(self, hidden_dim, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x, time_emb):
        time_feat = self.time_mlp(time_emb).unsqueeze(1)  # [b,1,d]
        x = x * (1 + time_feat)

        x, _ = self.attn(x, x, x)
        return x


class AdaptiveLayerNorm1(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim * 2)

    def forward(self, x, time_emb):
        time_feat = self.time_proj(time_emb)
        gamma, beta = time_feat.chunk(2, dim=-1)
        return self.norm(x) * (gamma.unsqueeze(1) + 1) + beta.unsqueeze(1)









#############################418-2##########################
import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear = nn.Linear(1, emb_dim)

    def forward(self, t):
        t = t.float().unsqueeze(-1)  # [B, 1]
        return self.linear(t)  # [B, D]

class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, image_features):
        x_norm = self.norm(x)
        attn_out, _ = self.cross_attn(query=x_norm, key=image_features, value=image_features)
        return attn_out + x

class DenoiserWithCrossAttention(nn.Module):
    def __init__(self, emb_dim, image_dim):
        super().__init__()
        self.time_embed = TimeEmbedding(emb_dim)
        self.input_proj = nn.Linear(emb_dim, emb_dim)
        self.image_proj = nn.Linear(image_dim, emb_dim)
        self.transformer = CrossAttentionBlock(emb_dim)

    def forward(self, x, t, image_feat):
        # x: [B, T, D] - noisy tokens
        # t: [B] - timestep
        # image_feat: [B, N, image_dim]

        B, T, D = x.size()
        t_emb = self.time_embed(t)  # [B, D]
        t_emb = t_emb.unsqueeze(1).expand(B, T, D)  # [B, T, D]

        x = self.input_proj(x + t_emb)  # Add time embedding
        image_feat = self.image_proj(image_feat)

        out = self.transformer(x, image_feat)
        return out



