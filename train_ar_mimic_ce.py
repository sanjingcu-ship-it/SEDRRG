import os, json, csv, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from models.r2gen import R2GenModel1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_batch(tokenizer, arr):
    out = []
    for x in arr:
        if hasattr(x, "detach"):
            x = x.detach().cpu().tolist()
        out.append(tokenizer.decode(x))
    return out


def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--ann_path", required=True)
    parser.add_argument("--dataset_name", default="mimic_cxr")
    parser.add_argument("--max_seq_length", type=int, default=100)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_ve", type=float, default=1e-5)
    parser.add_argument("--lr_ed", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--record_dir", required=True)

    # model args expected by legacy code
    parser.add_argument("--visual_extractor", default="swim_transformer_v2")
    parser.add_argument("--visual_extractor_pretrained", action="store_true")
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--d_vf", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--logit_layers", type=int, default=1)
    parser.add_argument("--bos_idx", type=int, default=0)
    parser.add_argument("--eos_idx", type=int, default=0)
    parser.add_argument("--pad_idx", type=int, default=0)
    parser.add_argument("--use_bn", type=int, default=0)
    parser.add_argument("--drop_prob_lm", type=float, default=0.5)
    parser.add_argument("--rm_num_slots", type=int, default=3)
    parser.add_argument("--rm_num_heads", type=int, default=8)
    parser.add_argument("--rm_d_model", type=int, default=512)
    parser.add_argument("--sample_method", default="beam_search")
    parser.add_argument("--beam_size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sample_n", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--output_logsoftmax", type=int, default=1)
    parser.add_argument("--decoding_constraint", type=int, default=0)
    parser.add_argument("--block_trigrams", type=int, default=1)
    parser.add_argument("--n_gpu", type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.record_dir, exist_ok=True)

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = Tokenizer(args)
    args.vocab_size = tokenizer.get_vocab_size()
    args.pad_idx = int(tokenizer.pad_token_id)
    args.eos_idx = int(tokenizer.eos_token_id)
    args.bos_idx = int(tokenizer.cls_token_id)

    print("[AR] vocab_size/class_count:", args.vocab_size)
    print("[AR] pad/bos/eos:", args.pad_idx, args.bos_idx, args.eos_idx)

    train_loader = R2DataLoader(args, tokenizer, split="train", shuffle=True)
    val_loader = R2DataLoader(args, tokenizer, split="val", shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = R2GenModel1(args, tokenizer).to(device)

    ve_params = list(model.visual_extractor.parameters())
    ve_ids = {id(p) for p in ve_params}
    ed_params = [p for p in model.parameters() if id(p) not in ve_ids and p.requires_grad]

    optimizer = torch.optim.Adam([
        {"params": ve_params, "lr": args.lr_ve},
        {"params": ed_params, "lr": args.lr_ed},
    ], weight_decay=args.weight_decay)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tok = 0.0, 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs} train")
        for _, images, reports_ids, reports_masks in pbar:
            images = images.to(device, non_blocking=True)
            reports_ids = reports_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(images, reports_ids, mode="train")
            targets = reports_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                out.reshape(-1, out.size(-1)),
                targets.reshape(-1),
                ignore_index=args.pad_idx
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n_tok = int(targets.ne(args.pad_idx).sum().item())
            total_loss += float(loss.detach().cpu()) * max(n_tok, 1)
            total_tok += max(n_tok, 1)
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        train_loss = total_loss / max(total_tok, 1)

        model.eval()
        val_loss_sum, val_tok = 0.0, 0
        preview_rows = []

        with torch.no_grad():
            for batch_i, (ids, images, reports_ids, reports_masks) in enumerate(tqdm(val_loader, desc=f"epoch {epoch}/{args.epochs} val")):
                images = images.to(device, non_blocking=True)
                reports_ids = reports_ids.to(device, non_blocking=True)

                out = model(images, reports_ids, mode="train")
                targets = reports_ids[:, 1:].contiguous()

                loss = F.cross_entropy(
                    out.reshape(-1, out.size(-1)),
                    targets.reshape(-1),
                    ignore_index=args.pad_idx
                )

                n_tok = int(targets.ne(args.pad_idx).sum().item())
                val_loss_sum += float(loss.detach().cpu()) * max(n_tok, 1)
                val_tok += max(n_tok, 1)

                if batch_i == 0:
                    sample = model(images[:4], mode="sample")
                    s = sample[0] if isinstance(sample, tuple) else sample
                    preds = decode_batch(tokenizer, s)
                    refs = decode_batch(tokenizer, reports_ids[:4])
                    for sid, pred, ref in zip(ids[:4], preds, refs):
                        preview_rows.append({"id": sid, "pred": pred, "ref": ref})

        val_loss = val_loss_sum / max(val_tok, 1)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        print("[preview]")
        for r in preview_rows[:4]:
            print("ID:", r["id"])
            print("PRED:", r["pred"][:300])
            print("REF :", r["ref"][:300])

        save_checkpoint(os.path.join(args.save_dir, "current_checkpoint.pth"), model, optimizer, epoch, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(os.path.join(args.save_dir, "model_best.pth"), model, optimizer, epoch, best_val_loss)

        with open(os.path.join(args.record_dir, f"preview_epoch_{epoch}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "preview": preview_rows,
            }, f, ensure_ascii=False, indent=2)

    print("[AR] training finished. best_val_loss:", best_val_loss)


if __name__ == "__main__":
    main()
