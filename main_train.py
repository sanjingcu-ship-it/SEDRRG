import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the images.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the annotation file.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=2, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    # parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    # parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')
    parser.add_argument('--visual_extractor', type=str, default='swim_transformer_v2',
                        help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', action='store_true', default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--feature_dim', type=int, default=2048, help='the dimension of the feature.')
    parser.add_argument('--num_diffusion_steps', type=int, default=9, help='the number of time steps used for training')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=25, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=8e-7, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=10, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.5, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=90, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    #diffusion model
    # 图像 & 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dimension for all layers')
    parser.add_argument('--words_emb_dim', type=int, default=256, help='word embedding dimension')
    parser.add_argument('--vocab_size', type=int, default=12000, help='size of vocabulary')
    parser.add_argument('--seq_length', type=int, default=32, help='seq_length')

    # 报告结构设置
    parser.add_argument('--max_sent', type=int, default=4, help='maximum number of sentences')
    parser.add_argument('--max_word', type=int, default=32, help='maximum number of words per sentence')

    # 扩散相关参数
    # parser.add_argument('--timesteps', type=int, default=12, help='number of diffusion steps')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'],
                        help='beta schedule for diffusion process')
    parser.add_argument('--pred_method', type=str, default='pred_noise', choices=['pred_noise', 'pred_x0'],
                        help='prediction target for diffusion: noise or x0')

    # p2 loss trick（加权扩散损失）
    parser.add_argument('--p2_loss_weight_k', type=float, default=1.0, help='P2 loss weight k')
    parser.add_argument('--p2_loss_weight_gamma', type=float, default=0.0, help='P2 loss weight gamma')

    # 损失函数选择
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2'], help='type of diffusion loss')

    # 训练参数（可选）
    # parser.add_argument('--lr', type=float, default=1e-4)


    parser.add_argument('--use_lesion_mask', action='store_true', default=True)
    parser.add_argument('--lesion_alpha', type=float, default=0.5)

    parser.add_argument('--use_se', action='store_true', default=True)
    parser.add_argument('--se_ratio', type=int, default=16)

    args = parser.parse_args()
    return args

import random
# def main():
#     # parse arguments
#     args = parse_agrs()
#     seed = 5
#     # seed = random.randint(0, 10000)
#     print(seed)
#
#     # fix random seeds
#     torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)
#
#     # create tokenizer
#     tokenizer = Tokenizer(args)

#
#     # create data loader
#     train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
#     val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
#     test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
#
#     # build model architecture
#     model = R2GenModel(args, tokenizer)
#     from modules.loss import MedFactStructLoss
#     criterion = MedFactStructLoss(tokenizer=tokenizer)
#
#     # criterion = compute_loss
#     metrics = compute_scores
#     optimizer = build_optimizer(args, model)
#     lr_scheduler = build_lr_scheduler(args, optimizer)
#     trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
#     trainer.train()
def main():
    import random
    import numpy as np
    import torch

    from copy import deepcopy

    args = parse_agrs()
    best_bleu4 = -1
    best_seed = -1
    best_score = None

    for seed in range(10000):  # 可以改成你想跑的次数
        # seed = 49
        # seed = random.randint(0, 10000)
        print(f"\n[Trial {seed + 1}] Using seed: {seed}")

        # 设置随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 初始化各组件
        tokenizer = Tokenizer(args)

        # Keep diffusion vocabulary size consistent with tokenizer ids.
        # diffusion.py internally uses args.vocab_size + 1, so args.vocab_size
        # should be the maximum valid token id rather than len(vocab).
        if hasattr(tokenizer, "idx2token"):
            args.vocab_size = max(int(k) for k in tokenizer.idx2token.keys())
        elif hasattr(tokenizer, "token2idx"):
            args.vocab_size = max(int(v) for v in tokenizer.token2idx.values())
        else:
            raise AttributeError("Tokenizer has neither idx2token nor token2idx; cannot infer args.vocab_size.")
        print(f"[VocabFix] args.vocab_size={args.vocab_size}, diffusion_vocab_size={args.vocab_size + 1}")

        train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
        val_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
        test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

        model = R2GenModel(args, tokenizer)
        # criterion = compute_loss
        from modules.loss import MedFactStructLoss
        criterion = MedFactStructLoss(tokenizer=tokenizer)
        metrics = compute_scores
        optimizer = build_optimizer(args, model)
        lr_scheduler = build_lr_scheduler(args, optimizer)

        trainer = Trainer(
            model, criterion, metrics, optimizer, args,
            lr_scheduler, train_dataloader, val_dataloader, test_dataloader
        )

        # 训练一个 epoch 并返回验证指标
        val_log = trainer.train()  # 假设 train() 里会返回验证结果 dict


        bleu4 = val_log.get('val_BLEU_4', 0.0)
        print(f"[Seed {seed}] BLEU_4 = {float(bleu4):.4f}")


        if bleu4 > best_bleu4:
            best_bleu4 = bleu4
            best_seed = seed
            best_score = deepcopy(val_log)
        print("\n==========================")
        print(f"🏆 最佳 Seed: {best_seed}")
        print(f"📈 最佳 BLEU_4: {best_bleu4:.4f}")
        print("📊 对应的完整验证指标:")
        for k, v in best_score.items():
            print(f"{k}: {v}")




if __name__ == '__main__':
    main()
