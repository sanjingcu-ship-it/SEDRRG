import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)
class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]

        image_id = (
            example.get('dicom_id')
            or example.get('id')
            or example.get('uid')
            or example.get('image_id')
            or example.get('study_id')
            or str(idx)
        )

        image_path = (
            example.get('image_path')
            or example.get('image_paths')
            or example.get('images')
            or example.get('filename')
        )
        if image_path is None:
            raise KeyError(f"Cannot find image path key in example. Available keys: {list(example.keys())}")

        if isinstance(image_path, str):
            candidate_paths = [image_path]
        elif isinstance(image_path, (list, tuple)):
            candidate_paths = list(image_path)
        else:
            raise TypeError(f"Unsupported image_path type: {type(image_path)} for image_id={image_id}")

        image_1 = None
        last_error = None
        for p in candidate_paths:
            # Some annotations store absolute paths, while others store paths relative to image_dir.
            full_path = p if os.path.isabs(str(p)) else os.path.join(self.image_dir, str(p))
            try:
                image_1 = Image.open(full_path).convert('RGB')
                break
            except Exception as e:
                last_error = e
                continue

        if image_1 is None:
            raise FileNotFoundError(
                f"Failed to load any image for image_id={image_id}. "
                f"candidate_paths={candidate_paths}, image_dir={self.image_dir}, last_error={last_error}"
            )

        if self.transform is not None:
            image_1 = self.transform(image_1)

        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image_1, report_ids, report_masks, seq_length)
        return sample

###########################################mimic版#####################################
# import numpy as np
# class IuxrayMultiImageDataset(BaseDataset):
#     def __getitem__(self, idx):
#         # 创建默认的PIL图像（中灰色）
#         default_pil = Image.new('RGB', (256, 256), (128, 128, 128))
#
#         example = self.examples[idx]
#         image_id = example['id']  #如果是mimic数据集改成dicom_id
#         image_path = example['image_path']
#
#         try:
#             # 尝试加载并转换图像
#             image_1 = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
#             if hasattr(self, 'transform') and self.transform is not None:
#                 image_1 = self.transform(image_1)  # 此时image_1应该是PIL图像
#         except Exception as e:
#             # 打印前10个错误避免日志爆炸
#             if not hasattr(self, '_error_count'):
#                 self._error_count = 0
#             if self._error_count < 10:
#                 print(f"图像加载失败，使用默认图像: {image_path} | 错误: {str(e)}")
#                 self._error_count += 1
#             # 使用默认图像（确保应用相同的transform）
#             image_1 = self.transform(default_pil) if hasattr(self,
#                                                              'transform') and self.transform is not None else default_pil
#
#         report_ids = example['ids']
#         report_masks = example['mask']
#         seq_length = len(report_ids)
#
#         # 确保最终image_1是张量
#         if isinstance(image_1, Image.Image):
#             image_1 = torch.tensor(np.array(image_1)).permute(2, 0, 1).float() / 255.0
#
#         sample = (image_id, image_1, report_ids, report_masks, seq_length)
#         return sample
###################################################################################################

class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
