import torch
import torch.nn as nn
import numpy as np
from diffusion import DiscreteDiffusion
from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
import torch.nn.functional as F

class R2GenModel1(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel1, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)  # 只计算一次
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    对logits进行top-k和top-p过滤

    参数:
        logits: 模型输出的原始logits [batch_size, seq_len, vocab_size]
        top_k: 保留概率最高的k个token (0表示不限制)
        top_p: 保留累计概率达到p的最小token集合 (1.0表示不限制)
        filter_value: 被过滤token设置的值

    返回:
        过滤后的logits [batch_size, seq_len, vocab_size]
    """
    # 确保输入是3D张量
    assert logits.dim() == 3, "Logits应该是3D张量[batch, seq_len, vocab]"

    batch_size, seq_len, vocab_size = logits.shape
    filtered_logits = logits.clone()

    for b in range(batch_size):
        for s in range(seq_len):
            # 获取当前时间步的logits
            current_logits = logits[b, s, :]

            # Top-k过滤
            if top_k > 0:
                # 获取top-k的阈值
                indices_to_remove = current_logits < torch.topk(current_logits, top_k)[0][..., -1, None]
                current_logits[indices_to_remove] = filter_value

            # Top-p (nucleus)过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(current_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累计概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                current_logits[indices_to_remove] = filter_value

            filtered_logits[b, s, :] = current_logits

    return filtered_logits

#######################扩散模型版##########################
def sample_next_token(logits, temperature=1.0, top_k=50):
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        v, ix = torch.topk(logits, k=top_k)
        logits_filtered = torch.full_like(logits, -float('inf'))
        logits_filtered.scatter_(0, ix, v)
        probs = F.softmax(logits_filtered, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)

    token_id = torch.multinomial(probs, 1).item()
    return token_id
def prevent_repeated_punctuation(seq, max_repeat=2):
    new_seq = []
    count = 0
    for token in seq:
        if token == ".":
            count += 1
            if count > max_repeat:
                continue
        else:
            count = 0
        new_seq.append(token)
    return new_seq

class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.diffusion = DiscreteDiffusion(args, tokenizer)
        self.num_timesteps = args.num_diffusion_steps  # 需从 args 传入扩散步数
        self.vocab_size = args.vocab_size + 1  # 需包含 [PAD]

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def train_step(self, fc_feats, att_feats, targets):
        # print(f"[MASK] id: {self.tokenizer.mask_token_id}")
        # print(f"[.] id: {self.tokenizer.token_to_id('.')}")  # 或 self.tokenizer.token2idx["."] 视你用的 tokenizer 而定
        # print(f"<eos> id: {self.tokenizer.eos_token_id}")

        # 1. 确保输入连续
        targets = targets.contiguous()

        # 2. 扩散过程
        t = torch.randint(0, self.num_timesteps, (targets.size(0),),
                          device=targets.device, dtype=torch.long)
        x_t = self.diffusion.forward(targets, att_feats, fc_feats, t)

        # 3. 去噪过程
        # Compatibility with diffusion.q_sample() variants that return
        # (corrupted_tokens, corruption_mask) rather than only corrupted_tokens.
        if isinstance(x_t, (tuple, list)):
            x_t = x_t[0]
        logits, memory = self.diffusion.denoise_step(x_t, att_feats, fc_feats, t)

        # Also expose the pooled visual condition used by the diffusion denoiser.
        # This enables a lightweight image-memory contrastive MFSL term without
        # changing the denoising architecture.
        try:
            _, visual_cond = self.diffusion.visual_bridge(att_feats, fc_feats, return_attn=False)
        except Exception:
            visual_cond = None

        # 4. 计算主扩散 token 重建损失
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

        # 5. 返回完整训练字典，供 MFSL 使用
        return {
            'logits': logits,
            'loss': loss,
            'targets': targets,
            'memory': memory,
            'visual_cond': visual_cond,
            'att_feats': att_feats,
            'fc_feats': fc_feats
        }

    def sample(self, fc_feats, att_feats, max_len=None, alpha=None, temperature=None, top_k=None, ngram_boost=None):
        if max_len is None:
            max_len = int(getattr(self.args, 'sample_max_len', 22))
        if alpha is None:
            alpha = float(getattr(self.args, 'sample_alpha', 1.2))
        if temperature is None:
            temperature = float(getattr(self.args, 'sample_temperature', 0.8))
        if top_k is None:
            top_k = int(getattr(self.args, 'sample_top_k', 4))
        if ngram_boost is None:
            ngram_boost = float(getattr(self.args, 'sample_ngram_boost', 1.5))

        trace_enabled = bool(getattr(self.args, 'export_denoising_trace', False))

        with torch.no_grad():
            batch_size = att_feats.size(0)
            device = att_feats.device
            mask_id = int(self.tokenizer.mask_token_id)

            # 初始化 - 使用mask token
            x_t = torch.full((batch_size, max_len),
                             mask_id,
                             device=device)

            # 准备高 n-gram ID 列表（提前移到 GPU）
            high_ngram_ids = torch.tensor(self.tokenizer.high_ngram_ids, device=device)

            sample_steps = getattr(self.args, 'sample_diffusion_steps', None)
            if sample_steps is None or sample_steps <= 0 or sample_steps >= self.num_timesteps:
                sample_t_list = list(reversed(range(self.num_timesteps)))
            else:
                sample_t_list = torch.linspace(
                    self.num_timesteps - 1, 0, steps=sample_steps,
                    device=fc_feats.device
                ).round().long().tolist()
                dedup = []
                for _t in sample_t_list:
                    if int(_t) not in dedup:
                        dedup.append(int(_t))
                sample_t_list = dedup

            trace_steps_raw = str(getattr(self.args, 'trace_steps', '') or '').strip()
            if trace_steps_raw:
                trace_steps = set(int(x.strip()) for x in trace_steps_raw.split(',') if x.strip())
            else:
                trace_steps = set(int(t) for t in sample_t_list)

            trace_records = [[] for _ in range(batch_size)]

            def _append_trace(stage, timestep=None, logits_snapshot=None, prev_tokens=None):
                if not trace_enabled:
                    return

                mask_ratio = x_t.eq(mask_id).float().mean(dim=1).detach().cpu().tolist()
                observed_ratio = (~x_t.eq(mask_id)).float().mean(dim=1).detach().cpu().tolist()

                changed_ratio = [None] * batch_size
                if prev_tokens is not None:
                    changed_ratio = x_t.ne(prev_tokens).float().mean(dim=1).detach().cpu().tolist()

                entropy_vals = [None] * batch_size
                top_conf_vals = [None] * batch_size
                if logits_snapshot is not None:
                    probs_snapshot = F.softmax(logits_snapshot, dim=-1)
                    entropy_tensor = -(probs_snapshot * torch.log(probs_snapshot.clamp_min(1e-12))).sum(dim=-1).mean(dim=1)
                    top_conf_tensor = probs_snapshot.max(dim=-1)[0].mean(dim=1)
                    entropy_vals = entropy_tensor.detach().cpu().tolist()
                    top_conf_vals = top_conf_tensor.detach().cpu().tolist()

                x_cpu = x_t.detach().cpu().tolist()
                observed_cpu = (~x_t.eq(mask_id)).detach().cpu().tolist()

                for b in range(batch_size):
                    trace_records[b].append({
                        "stage": str(stage),
                        "timestep": None if timestep is None else int(timestep),
                        "token_ids": [int(v) for v in x_cpu[b]],
                        "committed": [bool(v) for v in observed_cpu[b]],
                        "mask_ratio": float(mask_ratio[b]),
                        "committed_ratio": float(observed_ratio[b]),
                        "changed_ratio_from_previous": None if changed_ratio[b] is None else float(changed_ratio[b]),
                        "mean_entropy": None if entropy_vals[b] is None else float(entropy_vals[b]),
                        "mean_top_confidence": None if top_conf_vals[b] is None else float(top_conf_vals[b])
                    })

            if trace_enabled:
                _append_trace(stage="initial_all_mask", timestep=None, logits_snapshot=None, prev_tokens=None)

            for t in sample_t_list:
                prev_x_t = x_t.clone() if trace_enabled else None

                t_tensor = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
                logits, _ = self.diffusion.denoise_step(x_t, att_feats, fc_feats, t_tensor)

                # ✅ 应用长度惩罚
                current_alpha = min(alpha * (1 + int(t) / self.num_timesteps), 2.0)
                for pos in range(max_len):
                    length_penalty = ((5 + pos) / (5 + 1)) ** current_alpha
                    logits[:, pos, :] = logits[:, pos, :] / length_penalty

                # ✅ 屏蔽 <unk>
                logits[:, :, self.tokenizer.unk_token_id] = -float('inf')

                # ✅ 提升高 n-gram 的 logits
                logits[:, :, high_ngram_ids] *= ngram_boost  # e.g. 1.5 ~ 2.0

                # ✅ 温度采样 & top-k
                probs = F.softmax(logits / temperature, dim=-1)
                x_t = self._sample_with_topk(probs, top_k)

                if trace_enabled and int(t) in trace_steps:
                    _append_trace(stage="reverse_step", timestep=int(t), logits_snapshot=logits, prev_tokens=prev_x_t)

            if trace_enabled:
                _append_trace(stage="final_sample", timestep=0, logits_snapshot=None, prev_tokens=None)
                return x_t.cpu().numpy(), trace_records

            return x_t.cpu().numpy()


    def _sample_with_topk(self, probs, top_k):
        """优化后的Top-K采样，支持n-gram奖励的向量化计算"""
        batch_size, seq_len, vocab_size = probs.shape
        device = probs.device

        # === 预计算n-gram索引和权重 ===
        if not hasattr(self, '_n_gram_indices'):  # 缓存计算结果
            self._n_gram_indices = {
                2: torch.tensor([idx for idx, t in self.tokenizer.idx2token.items()
                                 if len(t.split()) == 2], device=device),
                3: torch.tensor([idx for idx, t in self.tokenizer.idx2token.items()
                                 if len(t.split()) == 3], device=device),
                4: torch.tensor([idx for idx, t in self.tokenizer.idx2token.items()
                                 if len(t.split()) == 4], device=device)
            }
            self._n_gram_weights = torch.ones(vocab_size, device=device)
            # for n, indices in self._n_gram_indices.items():
            #     self._n_gram_weights[indices] = 1.0 - 0 * n  # n-gram奖励系数

        # === 向量化n-gram奖励 ===
        boosted_probs = probs * self._n_gram_weights.unsqueeze(0).unsqueeze(0)  # [B,L,V]

        # === Top-K过滤 ===
        if top_k > 0:
            topk_probs, topk_indices = torch.topk(boosted_probs, top_k, dim=-1)  # [B,L,K]

            # 重建概率分布（保持原始形状）
            sampled_probs = torch.zeros_like(boosted_probs).scatter_(
                -1, topk_indices, topk_probs)
            sampled_probs = sampled_probs / sampled_probs.sum(dim=-1, keepdim=True)
        else:
            sampled_probs = boosted_probs

        # === 并行采样 ===
        samples = torch.multinomial(
            sampled_probs.view(-1, vocab_size),  # 展平为[B*L, V]
            num_samples=1
        ).view(batch_size, seq_len)

        return samples



    def forward_iu_xray(self, images, targets=None, mode='train'):
        """Forward path for IU X-Ray using discrete report-token diffusion."""
        att_feats, fc_feats = self.visual_extractor(images)

        if mode == 'train':
            outputs = self.train_step(fc_feats, att_feats, targets)
            assert 'logits' in outputs, "train_step must return 'logits'"
            return outputs

        elif mode == 'sample':
            return self.sample(fc_feats, att_feats)

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        """Forward path for MIMIC-CXR using the same discrete-diffusion model as IU X-Ray."""
        att_feats, fc_feats = self.visual_extractor(images)

        if mode == 'train':
            outputs = self.train_step(fc_feats, att_feats, targets)
            assert 'logits' in outputs, "train_step must return 'logits'"
            return outputs

        elif mode == 'sample':
            return self.sample(fc_feats, att_feats)

        else:
            raise ValueError(f"Unsupported mode: {mode}")
