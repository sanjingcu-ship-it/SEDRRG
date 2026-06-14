import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.tester import Tester
from modules.loss import compute_loss
from models.r2gen import R2GenModel


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='swim_transformer_v2', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
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
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--num_diffusion_steps', type=int, default=9, help='the number of diffusion denoising steps used for sampling')
    parser.add_argument('--sample_diffusion_steps', type=int, default=None, help='number of reverse diffusion steps used only at sampling time')
    parser.add_argument('--sample_max_len', type=int, default=22, help='maximum token length used during diffusion sampling')
    parser.add_argument('--sample_alpha', type=float, default=1.2, help='length penalty alpha used during diffusion sampling')
    parser.add_argument('--sample_temperature', type=float, default=0.8, help='temperature used during diffusion sampling')
    parser.add_argument('--sample_top_k', type=int, default=4, help='top-k used during diffusion sampling')
    parser.add_argument('--sample_ngram_boost', type=float, default=1.5, help='n-gram boost used during diffusion sampling')

    # Denoising trajectory export for supplementary visualization.
    # This does not change training or normal testing unless --export_denoising_trace is specified.
    parser.add_argument('--export_denoising_trace', action='store_true',
                        help='export intermediate discrete denoising states during test-time sampling')
    parser.add_argument('--trace_output', type=str, default='',
                        help='JSONL file used to save exported denoising trajectories')
    parser.add_argument('--trace_num_cases', type=int, default=50,
                        help='maximum number of test cases for trajectory export')
    parser.add_argument('--trace_steps', type=str, default='',
                        help='comma-separated reverse timesteps to save, e.g., 8,6,4,2,0 or 5,4,3,2,1,0')

    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--load', type=str, help='whether to load a pre-trained model.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
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
    # [SYNC-TOKENIZER-SPECIAL-IDS-20260603]
    args.pad_idx = int(getattr(tokenizer, "pad_token_id", getattr(args, "pad_idx", 0)))
    args.eos_idx = int(getattr(tokenizer, "eos_token_id", getattr(args, "eos_idx", 0)))
    args.bos_idx = int(getattr(tokenizer, "cls_token_id", getattr(args, "bos_idx", 0)))
    print(f"[SpecialIDFix] pad_idx={args.pad_idx}, bos_idx={args.bos_idx}, eos_idx={args.eos_idx}")


    # create data loader
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = R2GenModel(args, tokenizer)

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build trainer and start to train
    tester = Tester(model, criterion, metrics, args, test_dataloader)
    tester.test()


if __name__ == '__main__':
    main()
