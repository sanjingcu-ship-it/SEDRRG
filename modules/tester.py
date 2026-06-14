import logging
import os
import json
from abc import abstractmethod

import cv2
import pandas as pd
import torch

from modules.utils import generate_heatmap
from tqdm import tqdm
import numpy as np


class BaseTester(object):
    def __init__(self, model, criterion, metric_ftns, args):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._load_checkpoint(args.load)

    @abstractmethod
    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        tokenizer = model_ref.tokenizer

        trace_enabled = bool(getattr(self.args, 'export_denoising_trace', False))
        trace_limit = int(getattr(self.args, 'trace_num_cases', 0) or 0)
        trace_output = str(getattr(self.args, 'trace_output', '') or '')

        if trace_enabled and not trace_output:
            trace_output = os.path.join(self.save_dir, "denoising_trace.jsonl")

        trace_fp = None
        trace_count = 0

        if trace_enabled:
            os.makedirs(os.path.dirname(trace_output) or ".", exist_ok=True)
            trace_fp = open(trace_output, "w", encoding="utf-8")

        def _case_ids_to_list(x):
            if isinstance(x, (list, tuple)):
                return [str(v) for v in x]
            try:
                if hasattr(x, "detach"):
                    x = x.detach().cpu().numpy()
                if hasattr(x, "tolist"):
                    x = x.tolist()
                if isinstance(x, list):
                    return [str(v) for v in x]
            except Exception:
                pass
            return [str(x)]

        try:
            with torch.no_grad():
                test_gts, test_res = [], []

                for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                    images = images.to(self.device)
                    reports_ids = reports_ids.to(self.device)
                    reports_masks = reports_masks.to(self.device)

                    output = self.model(images, mode='sample')

                    batch_traces = None
                    if isinstance(output, (tuple, list)):
                        if len(output) == 2:
                            output, batch_traces = output[0], output[1]
                        else:
                            raise RuntimeError("Unexpected sample output length: {}".format(len(output)))

                    if hasattr(output, "detach"):
                        output_np = output.detach().cpu().numpy()
                    elif hasattr(output, "cpu"):
                        output_np = output.cpu().numpy()
                    else:
                        output_np = np.asarray(output)

                    reports = tokenizer.decode_batch(output_np)
                    ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)

                    if trace_enabled and batch_traces is not None and trace_fp is not None:
                        case_ids = _case_ids_to_list(images_id)

                        for local_i, trace in enumerate(batch_traces):
                            if trace_limit > 0 and trace_count >= trace_limit:
                                break

                            decoded_trace = []
                            for item in trace:
                                token_ids = np.asarray([item["token_ids"]])
                                decoded_text = tokenizer.decode_batch(token_ids)[0]

                                decoded_item = dict(item)
                                decoded_item["decoded_text"] = decoded_text
                                decoded_trace.append(decoded_item)

                            rec = {
                                "case_index": int(trace_count),
                                "batch_index": int(batch_idx),
                                "local_index": int(local_i),
                                "image_id": case_ids[local_i] if local_i < len(case_ids) else str(images_id),
                                "dataset": str(getattr(self.args, "dataset_name", "")),
                                "checkpoint": str(getattr(self.args, "load", "")),
                                "seed": int(getattr(self.args, "seed", -1)),
                                "num_diffusion_steps": int(getattr(self.args, "num_diffusion_steps", -1)),
                                "sample_diffusion_steps": getattr(self.args, "sample_diffusion_steps", None),
                                "sample_max_len": int(getattr(self.args, "sample_max_len", -1)),
                                "sample_top_k": int(getattr(self.args, "sample_top_k", -1)),
                                "reference_report": ground_truths[local_i] if local_i < len(ground_truths) else "",
                                "final_generated_report": reports[local_i] if local_i < len(reports) else "",
                                "trace": decoded_trace
                            }

                            trace_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            trace_count += 1

                    if trace_enabled and trace_limit > 0 and trace_count >= trace_limit:
                        break

                test_met = self.metric_ftns(
                    {i: [gt] for i, gt in enumerate(test_gts)},
                    {i: [re] for i, re in enumerate(test_res)}
                )
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                print(log)

                pd.DataFrame(test_res).to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
                pd.DataFrame(test_gts).to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)

                if trace_enabled:
                    print("[TraceExport] saved {} cases to {}".format(trace_count, trace_output))

        finally:
            if trace_fp is not None:
                trace_fp.close()

        return log

    def plot(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _load_checkpoint(self, load_path):
        load_path = str(load_path)
        self.logger.info("Loading checkpoint: {} ...".format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])


class Tester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(Tester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()

        model_ref = self.model.module if hasattr(self.model, 'module') else self.model
        tokenizer = model_ref.tokenizer

        trace_enabled = bool(getattr(self.args, 'export_denoising_trace', False))
        trace_limit = int(getattr(self.args, 'trace_num_cases', 0) or 0)
        trace_output = str(getattr(self.args, 'trace_output', '') or '')

        if trace_enabled and not trace_output:
            trace_output = os.path.join(self.save_dir, "denoising_trace.jsonl")

        trace_fp = None
        trace_count = 0

        if trace_enabled:
            os.makedirs(os.path.dirname(trace_output) or ".", exist_ok=True)
            trace_fp = open(trace_output, "w", encoding="utf-8")

        def _case_ids_to_list(x):
            if isinstance(x, (list, tuple)):
                return [str(v) for v in x]
            try:
                if hasattr(x, "detach"):
                    x = x.detach().cpu().numpy()
                if hasattr(x, "tolist"):
                    x = x.tolist()
                if isinstance(x, list):
                    return [str(v) for v in x]
            except Exception:
                pass
            return [str(x)]

        try:
            with torch.no_grad():
                test_gts, test_res = [], []

                for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                    images = images.to(self.device)
                    reports_ids = reports_ids.to(self.device)
                    reports_masks = reports_masks.to(self.device)

                    output = self.model(images, mode='sample')

                    batch_traces = None
                    if isinstance(output, (tuple, list)):
                        if len(output) == 2:
                            output, batch_traces = output[0], output[1]
                        else:
                            raise RuntimeError("Unexpected sample output length: {}".format(len(output)))

                    if hasattr(output, "detach"):
                        output_np = output.detach().cpu().numpy()
                    elif hasattr(output, "cpu"):
                        output_np = output.cpu().numpy()
                    else:
                        output_np = np.asarray(output)

                    reports = tokenizer.decode_batch(output_np)
                    ground_truths = tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)

                    if trace_enabled and batch_traces is not None and trace_fp is not None:
                        case_ids = _case_ids_to_list(images_id)

                        for local_i, trace in enumerate(batch_traces):
                            if trace_limit > 0 and trace_count >= trace_limit:
                                break

                            decoded_trace = []
                            for item in trace:
                                token_ids = np.asarray([item["token_ids"]])
                                decoded_text = tokenizer.decode_batch(token_ids)[0]

                                decoded_item = dict(item)
                                decoded_item["decoded_text"] = decoded_text
                                decoded_trace.append(decoded_item)

                            rec = {
                                "case_index": int(trace_count),
                                "batch_index": int(batch_idx),
                                "local_index": int(local_i),
                                "image_id": case_ids[local_i] if local_i < len(case_ids) else str(images_id),
                                "dataset": str(getattr(self.args, "dataset_name", "")),
                                "checkpoint": str(getattr(self.args, "load", "")),
                                "seed": int(getattr(self.args, "seed", -1)),
                                "num_diffusion_steps": int(getattr(self.args, "num_diffusion_steps", -1)),
                                "sample_diffusion_steps": getattr(self.args, "sample_diffusion_steps", None),
                                "sample_max_len": int(getattr(self.args, "sample_max_len", -1)),
                                "sample_top_k": int(getattr(self.args, "sample_top_k", -1)),
                                "reference_report": ground_truths[local_i] if local_i < len(ground_truths) else "",
                                "final_generated_report": reports[local_i] if local_i < len(reports) else "",
                                "trace": decoded_trace
                            }

                            trace_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            trace_count += 1

                    if trace_enabled and trace_limit > 0 and trace_count >= trace_limit:
                        break

                test_met = self.metric_ftns(
                    {i: [gt] for i, gt in enumerate(test_gts)},
                    {i: [re] for i, re in enumerate(test_res)}
                )
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                print(log)

                pd.DataFrame(test_res).to_csv(os.path.join(self.save_dir, "res.csv"), index=False, header=False)
                pd.DataFrame(test_gts).to_csv(os.path.join(self.save_dir, "gts.csv"), index=False, header=False)

                if trace_enabled:
                    print("[TraceExport] saved {} cases to {}".format(trace_count, trace_output))

        finally:
            if trace_fp is not None:
                trace_fp.close()

        return log

    def plot(self):
        assert self.args.batch_size == 1 and self.args.beam_size == 1
        self.logger.info('Start to plot attention weights in the test set.')
        os.makedirs(os.path.join(self.save_dir, "attentions"), exist_ok=True)
        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        mean = mean[:, None, None]
        std = std[:, None, None]

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(enumerate(self.test_dataloader)):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)
                output = self.model(images, mode='sample')
                if hasattr(output, "detach"):
                    output_np = output.detach().cpu().numpy()
                elif hasattr(output, "cpu"):
                    output_np = output.cpu().numpy()
                else:
                    output_np = np.asarray(output)
                reports = self.model.tokenizer.decode_batch(output_np)
                image = torch.clamp((images[0].cpu() * std + mean) * 255, 0, 255).int().cpu().numpy()
                report = self.model.tokenizer.decode_batch(output.cpu().numpy())[0].split()
                attention_weights = [layer.src_attn.attn.cpu().numpy()[:, :, :-1].mean(0).mean(0) for layer in
                                     self.model.encoder_decoder.model.decoder.layers]
                for layer_idx, attns in enumerate(attention_weights):
                    assert len(attns) == len(report)
                    for word_idx, (attn, word) in enumerate(zip(attns, report)):
                        os.makedirs(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx)), exist_ok=True)

                        heatmap = generate_heatmap(image, attn)
                        cv2.imwrite(os.path.join(self.save_dir, "attentions", "{:04d}".format(batch_idx),
                                                 "layer_{}".format(layer_idx), "{:04d}_{}.png".format(word_idx, word)),
                                    heatmap)
