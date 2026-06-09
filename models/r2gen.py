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
        """
        Progressive mask-ratio training objective for discrete diffusion report generation.

        Motivation:
        - This objective combines standard diffusion denoising with masked reconstruction.
        - This objective mixes:
            1) regular diffusion corruption loss,
            2) progressive masked reconstruction loss at ratios such as 30/50/70/90%,
            3) a lighter all-mask alignment loss for inference compatibility.
        """
        targets = targets.contiguous()
        pad_id = int(self.tokenizer.pad_token_id)
        mask_id = int(self.tokenizer.mask_token_id)
        valid = targets.ne(pad_id)

        # 1) Regular diffusion corruption path.
        t = torch.randint(
            0,
            self.num_timesteps,
            (targets.size(0),),
            device=targets.device,
            dtype=torch.long
        )
        x_t = self.diffusion.forward(targets, att_feats, fc_feats, t)
        logits_reg, memory = self.diffusion.denoise_step(x_t, att_feats, fc_feats, t)

        loss_reg = F.cross_entropy(
            logits_reg.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=pad_id
        )

        # 2) Progressive mask-ratio path.
        # Easier than all-mask reconstruction and scales better to 1k/10k.
        ratio_choices = getattr(self.args, "progressive_mask_ratios", None)
        if ratio_choices is None:
            ratio_choices = [0.30, 0.50, 0.70, 0.90]
        if isinstance(ratio_choices, str):
            ratio_choices = [float(x) for x in ratio_choices.split(",") if x.strip()]

        B = targets.size(0)
        ratios = torch.tensor(ratio_choices, device=targets.device, dtype=torch.float)
        pick = torch.randint(0, ratios.numel(), (B,), device=targets.device)
        batch_ratios = ratios[pick].view(B, 1)

        rand = torch.rand(targets.shape, device=targets.device)
        prog_mask = valid & (rand < batch_ratios)

        # Guarantee at least one masked valid token per sample when possible.
        for b in range(B):
            if valid[b].any() and not prog_mask[b].any():
                pos = valid[b].nonzero(as_tuple=False).flatten()
                j = pos[torch.randint(0, pos.numel(), (1,), device=targets.device)]
                prog_mask[b, j] = True

        x_prog = targets.clone()
        x_prog[prog_mask] = mask_id

        # Use a high timestep for masked reconstruction to align with denoising difficulty.
        t_prog = torch.full(
            (B,),
            self.num_timesteps - 1,
            device=targets.device,
            dtype=torch.long
        )
        logits_prog, _ = self.diffusion.denoise_step(x_prog, att_feats, fc_feats, t_prog)

        loss_prog = F.cross_entropy(
            logits_prog.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=pad_id
        )

        # 3) Light all-mask alignment path.
        # Keep this term, but do not let it dominate early training.
        allmask_weight = float(getattr(self.args, "allmask_loss_weight", 0.30))
        if allmask_weight > 0:
            x_all = targets.clone()
            x_all[valid] = mask_id
            t_all = torch.full(
                (B,),
                self.num_timesteps - 1,
                device=targets.device,
                dtype=torch.long
            )
            logits_all, _ = self.diffusion.denoise_step(x_all, att_feats, fc_feats, t_all)
            loss_all = F.cross_entropy(
                logits_all.reshape(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=pad_id
            )
            logits = logits_all
        else:
            loss_all = torch.zeros_like(loss_reg)
            logits = logits_prog

        reg_weight = float(getattr(self.args, "regular_diffusion_loss_weight", 0.50))
        prog_weight = float(getattr(self.args, "progressive_mask_loss_weight", 1.00))

        loss = reg_weight * loss_reg + prog_weight * loss_prog + allmask_weight * loss_all

        return {
            'logits': logits,
            'loss': loss,
            'loss_reg': loss_reg,
            'loss_prog': loss_prog,
            'loss_mask': loss_all,
            'loss_allmask': loss_all,
            'targets': targets
        }

    def sample(self, fc_feats, att_feats, max_len=None, alpha=None, temperature=None, top_k=None, ngram_boost=None):
        """
        One-shot all-mask denoising sampler.

        This matches the mask-aligned training path:
        all <mask> tokens at the highest diffusion timestep to estimate x_0.
        It is deterministic and avoids repeatedly overwriting already plausible
        tokens during reverse sampling.
        """
        if max_len is None:
            max_len = int(getattr(self.args, 'sample_max_len', 80))

        with torch.no_grad():
            batch_size = att_feats.size(0)
            device = att_feats.device
            mask_id = int(self.tokenizer.mask_token_id)

            x_mask = torch.full(
                (batch_size, max_len),
                mask_id,
                device=device,
                dtype=torch.long
            )

            t_mask = torch.full(
                (batch_size,),
                self.num_timesteps - 1,
                device=device,
                dtype=torch.long
            )

            logits, _ = self.diffusion.denoise_step(x_mask, att_feats, fc_feats, t_mask)

            # Ban non-text special tokens. Keep EOS allowed, but not too early.
            ban_ids = set()
            for attr in ["pad_token_id", "unk_token_id", "mask_token_id", "cls_token_id"]:
                if hasattr(self.tokenizer, attr):
                    v = getattr(self.tokenizer, attr)
                    if v is not None and 0 <= int(v) < logits.size(-1):
                        ban_ids.add(int(v))

            if ban_ids:
                ban_ids_tensor = torch.tensor(sorted(ban_ids), device=device, dtype=torch.long)
                logits[:, :, ban_ids_tensor] = -float("inf")

            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            eos_id = None if eos_id is None else int(eos_id)
            min_len = int(getattr(self.args, "sample_min_len", 5))
            if eos_id is not None and 0 <= eos_id < logits.size(-1) and min_len > 0:
                logits[:, :min(min_len, max_len), eos_id] = -float("inf")

            # Optional medical phrase / n-gram boost during inference.
            # A factor of 1.0 disables the boost. A factor larger than 1.0
            # increases phrase-token logits by log(boost_factor), which is
            # equivalent to multiplying their probabilities by boost_factor.
            if ngram_boost is None:
                ngram_boost = float(getattr(self.args, "sample_ngram_boost", 1.0))
            boost_factor = float(ngram_boost)

            if boost_factor > 0.0 and abs(boost_factor - 1.0) > 1e-12:
                phrase_ids = [
                    int(idx)
                    for idx, tok in getattr(self.tokenizer, "idx2token", {}).items()
                    if isinstance(tok, str) and len(tok.split()) >= 2
                ]
                if len(phrase_ids) > 0:
                    phrase_ids = torch.as_tensor(phrase_ids, device=device, dtype=torch.long)
                    phrase_ids = phrase_ids[
                        (phrase_ids >= 0) & (phrase_ids < logits.size(-1))
                    ]
                    if phrase_ids.numel() > 0:
                        log_boost = torch.log(
                            torch.tensor(boost_factor, device=device, dtype=logits.dtype)
                        )
                        logits[:, :, phrase_ids] = logits[:, :, phrase_ids] + log_boost

            pred = logits.argmax(dim=-1)
            return pred.cpu().numpy()

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
        att_feats, fc_feats = self.visual_extractor(images)

        if mode == 'train':
            outputs = self.train_step(fc_feats, att_feats, targets)
            # 添加检查确保键存在
            assert 'logits' in outputs, "train_step 必须返回 'logits' 键"
            return outputs['logits']

        elif mode == 'sample':
            return self.sample(fc_feats, att_feats)
#########################################################






#
# import torch
# import torch.nn as nn
# import numpy as np
#
# from modules.visual_extractor import VisualExtractor
# from modules.encoder_decoder import EncoderDecoder
#
#
#
# # from transformers import AutoModelForMaskedLM
# # diffusion = AutoModelForMaskedLM.from_pretrained("kuleshov-group/bd3lm-owt-block_size16", trust_remote_code=True)
# #
# # class R2GenModel(nn.Module):
# #     def __init__(self, args, tokenizer):
# #         super(R2GenModel, self).__init__()
# #         self.args = args
# #         self.tokenizer = tokenizer
# #         self.visual_extractor = VisualExtractor(args)
# #         self.encoder_decoder = EncoderDecoder(args, tokenizer)
# #         if args.dataset_name == 'iu_xray':
# #             self.forward = self.forward_iu_xray
# #         else:
# #             self.forward = self.forward_mimic_cxr
# #
# #     def __str__(self):
# #         model_parameters = filter(lambda p: p.requires_grad, self.parameters())
# #         params = sum([np.prod(p.size()) for p in model_parameters])
# #         return super().__str__() + '\nTrainable parameters: {}'.format(params)
# #
# #     def forward_iu_xray(self, images, targets=None, mode='train'):
# #         att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
# #         att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
# #         fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
# #         att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
# #         if mode == 'train':
# #             output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
# #         elif mode == 'sample':
# #             output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
# #         else:
# #             raise ValueError
# #         return output
# #
# #     def forward_mimic_cxr(self, images, targets=None, mode='train'):
# #         att_feats, fc_feats = self.visual_extractor(images)
# #         if mode == 'train':
# #             output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
# #         elif mode == 'sample':
# #             output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
# #         else:
# #             raise ValueError
# #         return output
#
#
#
# #
# #
# # from transformers import AutoModelForMaskedLM
# # diffusion = AutoModelForMaskedLM.from_pretrained("kuleshov-group/bd3lm-owt-block_size16", trust_remote_code=True)
#
# class R2GenModel(nn.Module):
#     def __init__(self, args, tokenizer):
#         super(R2GenModel, self).__init__()
#         self.args = args
#         self.tokenizer = tokenizer
#         self.visual_extractor = VisualExtractor(args)
#         self.encoder_decoder = EncoderDecoder(args, tokenizer)
#         if args.dataset_name == 'iu_xray':
#             self.forward = self.forward_iu_xray
#         else:
#             self.forward = self.forward_mimic_cxr
#
#     def __str__(self):
#         model_parameters = filter(lambda p: p.requires_grad, self.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         return super().__str__() + '\nTrainable parameters: {}'.format(params)
#
#     def forward_iu_xray(self, images, targets=None, mode='train'):
#         att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
#         att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
#         fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
#         att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
#         if mode == 'train':
#             output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
#         elif mode == 'sample':
#             output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
#         else:
#             raise ValueError
#         return output
#
#     def forward_mimic_cxr(self, images, targets=None, mode='train'):
#         att_feats, fc_feats = self.visual_extractor(images)
#         if mode == 'train':
#             output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
#         elif mode == 'sample':
#             output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
#         else:
#             raise ValueError
#         return output
#
if __name__ == '__main__':
    def model_diagnosis(model, sample_batch):
        model.eval()
        images, reports = sample_batch
        att_feats, fc_feats = model.visual_extractor(images)

        # 检查视觉特征
        print(f"视觉特征均值: {att_feats.mean().item():.4f} ± {att_feats.std().item():.4f}")

        # 检查生成过程
        for t in [10, 5, 1]:  # 不同时间步
            logits, _ = model.diffusion.denoise_step(
                torch.full((1, 20), model.tokenizer.mask_token_id),
                att_feats[:1], fc_feats[:1],
                torch.tensor([t])
            )
            print(f"t={t}时logits范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")

        # 检查参数更新
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}梯度: {param.grad.abs().mean().item():.4f}")
    model_diagnosis(R2GenModel,16)
# ---- Added for diffusion R2GenModel forward compatibility on MIMIC-CXR ----
# The diffusion-version R2GenModel.__init__ switches to self.forward_mimic_cxr
# when dataset_name=mimic_cxr, but the method is absent in the reproduced code.
# This patch adds forward_iu_xray and forward_mimic_cxr for the diffusion model.

def _sedrrg_extract_visual_feats_for_diffusion(self, images):
    import torch

    # Case 1: multi-view tensor [B, V, C, H, W]
    if hasattr(images, "dim") and images.dim() == 5:
        att_feats_list = []
        fc_feats_list = []
        num_views = images.size(1)

        for v in range(num_views):
            att_v, fc_v = self.visual_extractor(images[:, v])
            att_feats_list.append(att_v)
            fc_feats_list.append(fc_v)

        att_feats = torch.stack(att_feats_list, dim=0).mean(dim=0)
        fc_feats = torch.stack(fc_feats_list, dim=0).mean(dim=0)
        return att_feats, fc_feats

    # Case 2: single-view tensor [B, C, H, W]
    return self.visual_extractor(images)


def _sedrrg_forward_iu_xray_diffusion(self, images, targets=None, mode='train'):
    att_feats, fc_feats = _sedrrg_extract_visual_feats_for_diffusion(self, images)

    if mode == 'train':
        return self.train_step(fc_feats, att_feats, targets)
    elif mode == 'sample':
        return self.sample(fc_feats, att_feats)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _sedrrg_forward_mimic_cxr_diffusion(self, images, targets=None, mode='train'):
    att_feats, fc_feats = _sedrrg_extract_visual_feats_for_diffusion(self, images)

    if mode == 'train':
        return self.train_step(fc_feats, att_feats, targets)
    elif mode == 'sample':
        return self.sample(fc_feats, att_feats)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


R2GenModel.forward_iu_xray = _sedrrg_forward_iu_xray_diffusion
R2GenModel.forward_mimic_cxr = _sedrrg_forward_mimic_cxr_diffusion
# ---- End diffusion R2GenModel forward compatibility patch ----
