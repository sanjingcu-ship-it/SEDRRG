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

        logits = logits[masks]
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
    """Medical factuality and structure learning objective for SEDRRG.

    The primary objective remains the discrete diffusion token reconstruction
    loss returned by R2GenModel.train_step. MFSL adds differentiable auxiliary
    terms over decoder logits:
      1) medical token/phrase consistency,
      2) report-structure token consistency,
      3) lightweight image-memory contrastive alignment when visual/text
         representations are available.

    This avoids non-differentiable decoded-text overlap losses and keeps the
    implementation aligned with the manuscript description.
    """

    def __init__(self, tokenizer, λ1=1.0, λ2=0.8, λ3=0.5, λ4=0.3):
        super(MedFactStructLoss, self).__init__()
        self.tokenizer = tokenizer
        self.language_loss = LanguageModelCriterion()
        self.λ1 = λ1
        self.λ2 = λ2
        self.λ3 = λ3
        self.λ4 = λ4

        self.clinical_terms = {
            # CheXbert 14-observation vocabulary and common surface forms
            "atelectasis", "cardiomegaly", "edema", "consolidation",
            "pneumonia", "pneumothorax", "effusion", "pleural", "opacity",
            "lesion", "fracture", "support", "device", "cardiomediastinal",
            "enlarged", "lung", "lungs", "airspace", "disease", "clear",
            "normal", "abnormality", "acute", "process", "granuloma",

            # common clinically important phrases, retained when present
            "pleural effusion", "no pleural effusion",
            "pneumothorax", "no pneumothorax",
            "focal consolidation", "no focal consolidation",
            "airspace disease", "no acute cardiopulmonary",
            "cardiomediastinal silhouette", "heart size",
            "support devices", "lung opacity", "lung lesion",

            # negation and uncertainty cues emphasized as factual tokens
            "no", "without", "negative", "absence", "absent", "unchanged",
            "stable", "mild", "moderate", "severe"
        }

        self.structure_terms = {
            ".", "<eos>", "findings", "impression", "comparison",
            "indication", "history"
        }

    def forward(self, outputs, reports_ids, reports_masks, images=None):
        # New diffusion path: outputs is a dict returned by R2GenModel.train_step.
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            target_ids = outputs.get("targets", reports_ids)
            main_loss = outputs.get("loss", None)
        else:
            logits = outputs
            target_ids = reports_ids
            main_loss = None

        seq_len = min(logits.size(1), target_ids.size(1))
        logits = logits[:, :seq_len, :]
        target_ids = target_ids[:, :seq_len].long()

        pad_id = int(getattr(self.tokenizer, "pad_token_id", 0))
        valid_mask = target_ids.ne(pad_id)

        # Exclude special tokens from auxiliary MFSL terms.
        for attr in ["cls_token_id", "mask_token_id"]:
            sid = getattr(self.tokenizer, attr, None)
            if sid is not None:
                valid_mask = valid_mask & target_ids.ne(int(sid))

        if main_loss is None:
            main_loss = self.language_loss(logits, target_ids, valid_mask.long())

        L_entity = self._medical_token_phrase_loss(logits, target_ids, valid_mask)
        L_struct = self._structure_token_loss(logits, target_ids, valid_mask)
        L_contrastive = self._image_memory_contrastive(outputs) if isinstance(outputs, dict) else self._zero_like(logits)

        total_loss = (
            self.λ1 * main_loss
            + self.λ2 * L_entity
            + self.λ3 * L_contrastive
            + self.λ4 * L_struct
        )

        # Expose detached components for debugging without affecting training.
        self.last_components = {
            "L_lm": float(main_loss.detach().cpu()),
            "L_entity": float(L_entity.detach().cpu()),
            "L_contrastive": float(L_contrastive.detach().cpu()),
            "L_struct": float(L_struct.detach().cpu()),
            "L_total": float(total_loss.detach().cpu()),
        }
        return total_loss

    def _zero_like(self, logits):
        return logits.sum() * 0.0

    def _ids_for_terms(self, terms):
        ids = set()

        # Include high-frequency phrase tokens constructed from training reports.
        for idx in getattr(self.tokenizer, "high_ngram_ids", []):
            try:
                idx = int(idx)
                if idx >= 0:
                    ids.add(idx)
            except Exception:
                pass

        token2idx = getattr(self.tokenizer, "token2idx", {})
        for term in terms:
            if term in token2idx:
                ids.add(int(token2idx[term]))

        return sorted(ids)

    def _position_mask_from_ids(self, target_ids, valid_mask, ids):
        if not ids:
            return valid_mask & False

        id_tensor = torch.tensor(ids, device=target_ids.device, dtype=target_ids.dtype)
        return (target_ids.unsqueeze(-1) == id_tensor.view(1, 1, -1)).any(dim=-1) & valid_mask

    def _masked_ce(self, logits, target_ids, pos_mask):
        if pos_mask.sum().item() == 0:
            return self._zero_like(logits)
        return F.cross_entropy(
            logits[pos_mask],
            target_ids[pos_mask],
            reduction="mean"
        )

    def _medical_token_phrase_loss(self, logits, target_ids, valid_mask):
        ids = self._ids_for_terms(self.clinical_terms)
        pos_mask = self._position_mask_from_ids(target_ids, valid_mask, ids)
        return self._masked_ce(logits, target_ids, pos_mask)

    def _structure_token_loss(self, logits, target_ids, valid_mask):
        token2idx = getattr(self.tokenizer, "token2idx", {})
        ids = set()

        for term in self.structure_terms:
            if term in token2idx:
                ids.add(int(token2idx[term]))

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            ids.add(int(eos_id))

        pos_mask = self._position_mask_from_ids(target_ids, valid_mask, sorted(ids))
        return self._masked_ce(logits, target_ids, pos_mask)

    def _image_memory_contrastive(self, outputs):
        memory = outputs.get("memory", None)
        visual_cond = outputs.get("visual_cond", None)

        if memory is None or visual_cond is None:
            # Try to return a graph-connected zero if possible.
            logits = outputs.get("logits", None)
            if logits is not None:
                return self._zero_like(logits)
            return torch.tensor(0.0, device=visual_cond.device if visual_cond is not None else "cpu")

        if memory.dim() == 3:
            text_repr = memory.mean(dim=1)
        else:
            text_repr = memory

        visual_repr = visual_cond

        if text_repr.dim() != 2 or visual_repr.dim() != 2:
            return self._zero_like(outputs["logits"])

        if text_repr.size(-1) != visual_repr.size(-1):
            return self._zero_like(outputs["logits"])

        if text_repr.size(0) < 2:
            # InfoNCE needs at least two samples. For batch size 1, use cosine distance.
            text_norm = F.normalize(text_repr, dim=-1)
            visual_norm = F.normalize(visual_repr, dim=-1)
            return 1.0 - (text_norm * visual_norm).sum(dim=-1).mean()

        text_norm = F.normalize(text_repr, dim=-1)
        visual_norm = F.normalize(visual_repr, dim=-1)

        temperature = 0.07
        sim = torch.matmul(text_norm, visual_norm.t()) / temperature
        labels = torch.arange(sim.size(0), device=sim.device)

        return 0.5 * (
            F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.t(), labels)
        )


def compute_loss(output, reports_ids, reports_masks):
    criterion = LanguageModelCriterion()
    return criterion(output, reports_ids[:, 1:], reports_masks[:, 1:])
