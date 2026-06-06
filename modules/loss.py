import torch
import torch.nn as nn


# class LanguageModelCriterion(nn.Module):
#     def __init__(self):
#         super(LanguageModelCriterion, self).__init__()
#
#     def forward(self, input, target, mask):
#         # truncate to the same size
#         target = target[:, :input.size(1)]
#         mask = mask[:, :input.size(1)]
#         output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
#         output = torch.sum(output) / torch.sum(mask)
#
#         return output
#
#
# def compute_loss(output, reports_ids, reports_masks):
#     criterion = LanguageModelCriterion()
#     loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
#     return loss

import torch.nn.functional as F

class LanguageModelCriterion(nn.Module):
    def forward(self, outputs, targets, masks):
        # [SEDRRG-DIFFUSION-LOSS-PASSTHROUGH]
        # Diffusion R2GenModel.train_step already computes a differentiable CE loss.
        # When present, use it directly instead of requiring legacy outputs['pred_ids'].
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs['logits']
        seq_len = min(logits.size(1), targets.size(1))
        logits = logits[:, :seq_len, :]
        targets = targets[:, :seq_len]
        masks = masks[:, :seq_len]

        logits = logits.reshape(-1, logits.size(-1))   # [B*T, vocab_size]
        targets = targets.reshape(-1)                  # [B*T]
        masks = masks.reshape(-1).bool()               # [B*T]

        logits = logits[masks]                         # 只保留有效 token
        targets = targets[masks]

        loss = F.cross_entropy(logits, targets, reduction='mean')
        return loss

# def compute_loss(output, reports_ids, reports_masks):
#     criterion = LanguageModelCriterion()
#     return criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])


import torch
import torch.nn as nn
import torch.nn.functional as F

class MedFactStructLoss(nn.Module):
    def __init__(self, tokenizer, λ1=1.0, λ2=0.8, λ3=0.5, λ4=0.3):
        super(MedFactStructLoss, self).__init__()
        self.tokenizer = tokenizer
        self.language_loss = LanguageModelCriterion()
        self.λ1 = λ1
        self.λ2 = λ2
        self.λ3 = λ3
        self.λ4 = λ4
        # 可选：你可以加载实体抽取模型、图文对齐模块等

    def forward(self, outputs, reports_ids, reports_masks, images=None):
        # [SEDRRG-DIFFUSION-MEDFACT-PASSTHROUGH]
        # Diffusion R2GenModel.train_step already computes a differentiable CE loss.
        # For diffusion outputs, return it directly instead of requiring legacy outputs['pred_ids'].
        if isinstance(outputs, dict) and "loss" in outputs:
            return outputs["loss"]
        # 基础语言建模损失
        L_lm = self.language_loss(outputs, reports_ids, reports_masks)

        # 生成文本
        if isinstance(outputs, dict):
            generated_ids = outputs["pred_ids"]
        else:
            generated_ids = torch.argmax(outputs, dim=-1)

        gen_text = self.tokenizer.decode_batch(generated_ids)
        ref_text = self.tokenizer.decode_batch(reports_ids)

        # 2. 医学实体损失（你需要集成 RadGraph 或其他 NER）
        try:
            L_entity = self._entity_consistency_loss(ref_text, gen_text)
        except:
            L_entity = torch.tensor(0.0, device=L_lm.device)

        # 3. 图文对齐损失（可选：你需定义 encoder/embedding 相似度模块）
        try:
            L_contrastive = self._image_text_contrastive(images, gen_text)
        except:
            L_contrastive = torch.tensor(0.0, device=L_lm.device)

        # 4. 结构损失（你可以写一个模板或结构分类器来监督句子顺序）
        try:
            L_struct = self._structure_loss(gen_text)
        except:
            L_struct = torch.tensor(0.0, device=L_lm.device)

        total_loss = self.λ1 * L_lm + self.λ2 * L_entity + self.λ3 * L_contrastive + self.λ4 * L_struct
        return total_loss

    def _entity_consistency_loss(self, ref_texts, gen_texts):
        # Entity-level consistency loss can be replaced by a task-specific clinical entity scorer.
        loss = 0.0
        for ref, gen in zip(ref_texts, gen_texts):
            ref_set = set(ref.split())
            gen_set = set(gen.split())
            if len(ref_set) == 0:
                continue
            overlap = len(ref_set & gen_set) / len(ref_set)
            loss += (1.0 - overlap)
        return torch.tensor(loss / len(ref_texts), requires_grad=True)

    def _image_text_contrastive(self, images, gen_texts):
        # Image-text contrastive alignment can be implemented with task-specific paired features.
        return torch.tensor(0.0, requires_grad=True, device=images.device)

    def _structure_loss(self, gen_texts):
        # Structural order matching can be implemented with task-specific report section constraints.
        return torch.tensor(0.0, requires_grad=True, device='cuda' if torch.cuda.is_available() else 'cpu')
def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    return criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
