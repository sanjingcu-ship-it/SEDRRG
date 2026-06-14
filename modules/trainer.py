import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # Setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'test_' + args.monitor_metric  # use test metric for original-project reproduction
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        # 只保留 test 的记录
        self.best_recorder = {'test': {self.mnt_metric: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # Save logged information into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # Print logged information to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # Evaluate model performance based on test set
            best = False
            if self.mnt_mode != 'off':
                try:
                    # Check if model performance improved on test set
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print(f"Warning: Metric '{self.mnt_metric}' not found. Disabling monitoring.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print(f"Test performance didn't improve for {self.early_stop} epochs. Training stopped.")
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

        self._print_best()
        self._print_best_to_file()
        return self.best_recorder['test']

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)

        record_table = pd.concat([
            record_table,
            pd.DataFrame([self.best_recorder['test']])
        ], ignore_index=True)

        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: No GPU available, using CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(f"Warning: Only {n_gpu} GPUs available, using {n_gpu} instead of {n_gpu_use}.")
            n_gpu_use = n_gpu

        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        print(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")

    def _record_best(self, log):
        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['test'][self.mnt_metric]) or \
                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['test'][self.mnt_metric])
        if improved:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print(f'Best results (w.r.t {self.args.monitor_metric}) in test set:')
        for key, value in self.best_recorder['test'].items():
            print(f'\t{key:15s}: {value}')

from tqdm.auto import tqdm
class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.ep = 0

    def _train_epoch(self, epoch):
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.train_dataloader)):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
            # 前向传播
            outputs = self.model(images, reports_ids, mode='train')
            # 损失计算
            # loss = self.criterion(
            #     outputs,
            #     reports_ids[:, 1:],  # 去掉起始token
            #     reports_masks[:, 1:]  # 对应mask
            # )
            loss = self.criterion(
                outputs,
                reports_ids[:, 1:],
                reports_masks[:, 1:],
                images=images
            )

            # Optional MFSL diagnostics. Enable with SEDRRG_DEBUG_MFSL=1.
            if (
                os.environ.get("SEDRRG_DEBUG_MFSL", "0") == "1"
                and batch_idx < 3
                and hasattr(self.criterion, "last_components")
            ):
                print("[MFSL components]", self.criterion.last_components)

            train_loss += loss.item()
            self.optimizer.zero_grad()
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # self.model.eval()
        # with torch.no_grad():
        #     val_gts, val_res = [], []
        #     for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.val_dataloader)):
        #         images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
        #             self.device), reports_masks.to(self.device)
        #         output = self.model(images, mode='sample')
        #         reports = self.model.tokenizer.decode_batch(output)  # Directly pass the list of strings
        #         ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
        #         val_res.extend(reports)
        #         val_gts.extend(ground_truths)
        #     val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
        #                                {i: [re] for i, re in enumerate(val_res)})
        #     log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            self.ep += 1
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                # output = self.model(images, mode='sample')
                # reports = self.model.tokenizer.decode_batch(output)
                import re

                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output)

                # # 后处理：清洗重复句号和多余空格
                # cleaned_reports = []
                # for report in reports:
                #     report = re.sub(r'(\.\s*){2,}', '. ', report)  # 多个 ". " → 一个 ". "
                #     report = re.sub(r'\.{2,}', '.', report)  # 连续 "..." → "."
                #     report = re.sub(r'^[\.\s]+', '', report)  # 开头的 "." 或空格去掉
                #     report = re.sub(r'\s{2,}', ' ', report)  # 多个空格 → 一个空格
                #     report = report.strip()  # 去掉首尾空格
                #     cleaned_reports.append(report)
                #
                # reports = cleaned_reports



                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                # print('reports', reports)
                # print('ground_truths', ground_truths)
                # print('ground_truths',ground_truths)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            print('reports', reports)
            print('ground_truths', ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})



        self.lr_scheduler.step()

        return log
