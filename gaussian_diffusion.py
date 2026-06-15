import torch
from torch import nn
import torch.nn.functional as F


# from diffusion.ddiffusion.diffusion.gt import captions
from lstm_diffusion import LSTM_with_timeemb,LoRALayer,EnhancedLSTMWithTimeEmb, ImprovedDenoiser, DenoiserWithCrossAttention
from encoder import Encoder,ImprovedEncoder,EncoderCNN
from tqdm.auto import tqdm
import math
from einops import rearrange, reduce
from collections import namedtuple
from functools import partial


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def identity(t, *args, **kwargs):
    return t

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 timesteps,
                 max_sent,
                 max_word,
                 vocab_size,
                 words_emb_dim,
                 hidden_dim,
                 pred_method,
                 beta_schedule='cosine',
                 p2_loss_weight_k=1,
                 p2_loss_weight_gamma=0,
                 loss_type='l1'):
        super(GaussianDiffusion, self).__init__()

        self.max_sent = max_sent
        self.max_length = max_word
        self.words_emb_dim = words_emb_dim
        self.vocab_size = vocab_size
        self.loss_type = loss_type
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        # self.model = LSTM_with_timeemb(max_sent, max_word, hidden_dim, words_emb_dim, hidden_dim)
        self.model = ImprovedDenoiser(hidden_dim, words_emb_dim)

        self.objective = pred_method
        # self.model = Unet(dim=64, channels=num_sent, dim_mults=(1, 2, 4))
        self.num_timesteps = timesteps
        self.encoder = Encoder(words_emb_dim, -2)
        # self.encoder = EncoderCNN()
        self.emb = Embeddings(vocab_size, words_emb_dim)
        self.anti_emb = AntiEmbeddings(vocab_size, words_emb_dim)
        self.criterion = nn.CrossEntropyLoss()
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        register_buffer('p2_loss_weight',
                        (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, t, captions, topic, noise=None, loss_type="l1", train='emb'):
        if train == 'emb':
            x_start = self.emb(captions)
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_out = self.model(x_noisy, t, topic)

        elif train == 'dm':
            with torch.no_grad():
                x_start = self.emb(captions)
            noise = torch.randn_like(x_start)
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_out = self.model(x_noisy, t, topic)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start

        words_out = self.anti_emb(x_start)[0]  # (batch_size, max_sent, max_word, vocab_size)
        # print("b",words_out.shape)

        ce_loss = self.criterion(words_out.view(-1, self.vocab_size), captions.view(-1))

        if train == 'dm':
            loss_noisy = self.loss_fn(model_out, target, reduction='none')
            loss_noisy = reduce(loss_noisy, 'b ... -> b (...)', 'mean')
            loss_noisy = loss_noisy * extract(self.p2_loss_weight, t, loss_noisy.shape)
            loss_noisy = loss_noisy.mean()
            return loss_noisy, ce_loss
        elif train == 'emb':
            loss_emb = self.loss_fn(model_out, target)
            return loss_emb, ce_loss

    def forward(self, captions, img):

        b, device = captions.shape[0], captions.device  # x0(b,n,l)
        # print('b:',b)
        t_dm = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        t_emb = torch.full((b,), 1, device=device, dtype=torch.long)
        topic = self.encoder(img)
        # topic = torch.randn_like(topic1)
        # topic = self.encoder(img)
        # print(topic.shape)

        return self.train_mode('dm', t_emb, t_dm, captions, topic)

    def train_mode(self, mode, t_emb, t_dm, captions, topic):
        loss_emb, ce_loss2 = self.p_losses(t_emb, captions, topic, train='emb') if mode == 'emb' else (0., 0.)
        loss_dm, ce_loss1 = self.p_losses(t_dm, captions, topic, train='dm') if mode == 'dm' else (0., 0.)

        if mode == 'emb':
            loss = loss_emb + ce_loss2
            loss_dm_i = loss_dm
            loss_emb_i = loss_emb.item()
            loss_ce = ce_loss1 + ce_loss2.item()
        elif mode == 'dm':
            loss = loss_dm + ce_loss1
            loss_dm_i = loss_dm.item()
            loss_emb_i = loss_emb
            loss_ce = ce_loss1.item() + ce_loss2

        return loss, loss_dm_i, loss_emb_i, loss_ce

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, topic=None):
        model_output = self.model(x, t, topic)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)


        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, topic=None):
        preds = self.model_predictions(x, t, topic)
        x_start = preds.pred_x_start


        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x, t, topic):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, topic=topic)

        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img

    @torch.no_grad()
    def p_sample_loop(self, shape, topic, captions):
        b = shape[0]
        # xt = torch.randn(shape, device=topic.device)
        # w = captions
        w = torch.randint(0, self.vocab_size, shape, device=topic.device)
        xt = self.emb(w)

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            xt = self.p_sample(xt, t, topic)
        xt = self.anti_emb(xt)
        return xt

    @torch.no_grad()
    def sample(self, topic, batch_size, captions):
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, self.max_sent, self.max_length), topic, captions)


class Embeddings_LoRA(nn.Module):
    def __init__(self, vocab_size, words_emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, words_emb_dim)
        self.norm = nn.LayerNorm(words_emb_dim)

        self.word_emb = nn.Sequential(
            LoRALayer(nn.Linear(words_emb_dim, words_emb_dim)),
            nn.GELU(),
            LoRALayer(nn.Linear(words_emb_dim, words_emb_dim))
        )
    def forward(self, x):
        return self.word_emb(self.norm(self.emb(x)))
class Embeddings(nn.Module):
    def __init__(self, vocab_size, words_emb_dim):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, words_emb_dim)
        self.norm = nn.LayerNorm(words_emb_dim)
        self.word_emb = nn.Sequential(
            nn.Linear(words_emb_dim, words_emb_dim),
            nn.GELU(),
            nn.Linear(words_emb_dim, words_emb_dim)
        )

    def forward(self, x):
        return self.word_emb(self.norm(self.emb(x)))


class AntiEmbeddings(nn.Module):
    def __init__(self, vocab_size, words_emb_dim):
        super(AntiEmbeddings, self).__init__()
        self.anti_emb = nn.Sequential(
            nn.Linear(words_emb_dim, words_emb_dim),
            nn.GELU(),
            nn.Linear(words_emb_dim, vocab_size)
        )
    def forward(self, x):
        x = self.anti_emb(x)
        pred = torch.max(x, dim=-1)[1]
        return x, pred


import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedAntiEmbeddings(nn.Module):
    def __init__(self, vocab_size, words_emb_dim, num_heads=4, temperature=1.0):
        super(ImprovedAntiEmbeddings, self).__init__()
        self.temperature = temperature

        self.attention = nn.MultiheadAttention(embed_dim=words_emb_dim, num_heads=num_heads)

        self.anti_emb = nn.Sequential(
            nn.Linear(words_emb_dim, words_emb_dim * 2),
            nn.GELU(),
            nn.Linear(words_emb_dim * 2, words_emb_dim),
            nn.GELU(),
            nn.Linear(words_emb_dim, vocab_size)
        )

    def forward(self, x):
        batch_size, max_sent, max_word, emb_dim = x.shape

        x = x.view(batch_size * max_sent, max_word, emb_dim)  # (batch_size * max_sent, max_word, emb_dim)
        x = x.permute(1, 0, 2)  # (max_word, batch_size * max_sent, emb_dim)

        x, _ = self.attention(x, x, x)  # (max_word, batch_size * max_sent, emb_dim)
        x = x.permute(1, 0, 2)  # (batch_size * max_sent, max_word, emb_dim)

        x = x.view(batch_size, max_sent, max_word, emb_dim)  # (batch_size, max_sent, max_word, emb_dim)

        x = self.anti_emb(x)  # (batch_size, max_sent, max_word, vocab_size)

        x = x / self.temperature

        pred = torch.max(x, dim=-1)[1]  # (batch_size, max_sent, max_word)

        return x, pred
if __name__ == '__main__':
    # device = torch.device('cuda')
    # torch.backends.cudnn.benchmark = False
    gd = GaussianDiffusion(100, 4, 32, 100, 128, 256, 'pred_x0', loss_type='l2')
    params_dicts = [
        {'params': gd.model.parameters(), 'lr': 1e-3},
        {'params': gd.anti_emb.parameters(), 'lr': 1e-3},
    ]
    optim_model = torch.optim.Adam(params=params_dicts)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_model, T_max=200)
    print(sum([param.nelement() for param in gd.parameters()]))
    x0 = torch.randint(0, 100, [8, 4, 32])
    img = torch.randn([8, 3, 224, 224])
    for i in range(500):
        optim_model.zero_grad()
        loss, loss1, loss2 = gd(x0, img)
        # loss = gd.p_losses(model, x0, t)
        loss.backward()
        optim_model.step()
        print('epoch:%d loss: noisy(%.3f) emb(%.3f) lr: noisy(%.4f) emb(%.4f)' %
              (i, loss1, loss2, *[i['lr'] for i in optim_model.param_groups]))
        # scheduler.step()
    pred = gd.sample(8)
    # print(pred, x0)
