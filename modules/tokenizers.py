# import json
# import re
# from collections import Counter
#
#
# class Tokenizer(object):
#     def __init__(self, args):
#
#         self.ann_path = args.ann_path
#         self.threshold = args.threshold
#         self.dataset_name = args.dataset_name
#         if self.dataset_name == 'iu_xray':
#             self.clean_report = self.clean_report_iu_xray
#         else:
#             self.clean_report = self.clean_report_mimic_cxr
#         self.ann = json.loads(open(self.ann_path, 'r').read())
#         self.token2idx, self.idx2token = self.create_vocabulary()
#
#         self.unk_token = '<unk>'
#         self.unk_token_id = self.token2idx.get(self.unk_token, 0)
#
#         self.pad_token_id = 0
#         self.cls_token_id = len(self.token2idx) + 2
#         self.eos_token = '<eos>'
#         self.token2idx['<eos>'] = len(self.token2idx) + 3
#         self.idx2token[len(self.idx2token) + 3] = '<eos>'
#         self.eos_token_id = self.token2idx['<eos>']
#         self.token2idx['<pad>'] = 0
#         self.idx2token[0] = '<pad>'
#
#
#     def create_vocabulary(self):
#         total_tokens = []
#
#         for example in self.ann['train']:
#             tokens = self.clean_report(example['report']).split()
#             for token in tokens:
#                 total_tokens.append(token)
#
#         counter = Counter(total_tokens)
#         vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
#         vocab.sort()
#         token2idx, idx2token = {}, {}
#         for idx, token in enumerate(vocab):
#             token2idx[token] = idx + 1
#             idx2token[idx + 1] = token
#         return token2idx, idx2token
#
#     def clean_report_iu_xray(self, report):
#         report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
#             .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
#             .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
#                                         replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def clean_report_mimic_cxr(self, report):
#         report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#             .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#             .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#             .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#             .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#                                         .replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report
#
#     def get_token_by_id(self, id):
#         return self.idx2token[id]
#
#     def get_id_by_token(self, token):
#         if token not in self.token2idx:
#             return self.token2idx['<unk>']
#         return self.token2idx[token]
#
#     def get_vocab_size(self):
#         return len(self.token2idx)
#
#     def __call__(self, report):
#         tokens = self.clean_report(report).split()
#         ids = []
#         for token in tokens:
#             ids.append(self.get_id_by_token(token))
#         # ids = [0] + ids + [0]
#         ids = [self.cls_token_id] + [self.get_id_by_token(t) for t in tokens] + [self.eos_token_id]
#         return ids
#
#     def decode(self, ids):
#         txt = ''
#         for i, idx in enumerate(ids):
#             if idx > 0:
#                 if i >= 1:
#                     txt += ' '
#                 txt += self.idx2token.get(idx, self.unk_token)
#             else:
#                 break
#         return txt
#
#     def decode_batch(self, ids_batch):
#         out = []
#         for ids in ids_batch:
#             out.append(self.decode(ids))
#         return out
#
#     def is_valid_id(self, idx):
#         return idx in self.idx2token


import json
import re
from collections import Counter
import torch
import torch.nn as nn

class Tokenizer(object):
    def __init__(self, args):

        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.dataset_name = args.dataset_name
        if self.dataset_name == 'iu_xray':
            self.clean_report = self.clean_report_iu_xray
        else:
            self.clean_report = self.clean_report_mimic_cxr
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

        self.unk_token = '<unk>'
        self.unk_token_id = self.token2idx.get(self.unk_token, 0)

        self.pad_token_id = 0
        self.mask_token_id = len(self.token2idx) + 1
        self.cls_token_id = len(self.token2idx) + 2
        self.eos_token = '<eos>'
        self.token2idx['<eos>'] = len(self.token2idx) + 3
        self.idx2token[len(self.idx2token) + 3] = '<eos>'
        self.eos_token_id = self.token2idx['<eos>']
        self.token2idx['<pad>'] = 0
        self.idx2token[0] = '<pad>'
        self.high_ngram_ids = [idx for token, idx in self.token2idx.items() if len(token.split()) >= 2]

        print(f"Vocabulary size: {len(self.token2idx)}")
        print(f"Maximum ID: {max(self.idx2token.keys())}")
        print(f"Example tokens: {list(self.idx2token.items())[:80]}")

    def create_vocabulary(self):
        from nltk import word_tokenize, ngrams
        ngram_counter = Counter()

        for example in self.ann['train']:
            tokens = word_tokenize(self.clean_report(example['report']).lower())
            for n in range(1, 5):
                ngram_counter.update([' '.join(gram) for gram in ngrams(tokens, n)])
        # # [MIMIC-WORDLEVEL-VOCAB-20260603]
        if getattr(self, 'dataset_name', '') == 'mimic_cxr':
            # For MIMIC-CXR, use word-level vocabulary only. Phrase tokens caused
            # high-frequency template repetition during diffusion sampling.
            thresholds = {1: self.threshold, 2: 10**12, 3: 10**12, 4: 10**12}
        else:
            thresholds = {1: 4, 2: 1000, 3: 150, 4: 30}
        # thresholds = {1: 10, 2: 1000, 3: 150, 4: 50}

        # thresholds = {1: 30, 2: 1000, 3: 150, 4: 50}       0.354,0.076
        # thresholds = {1: 504, 2: 1000, 3: 150, 4: 70}  0.31,0.072

        # thresholds = {1: 11, 2: 1000, 3: 150, 4: 70}       0.359,0.082   22
        # thresholds = {1: 11, 2: 2000, 3: 150, 4: 70}      0.337,0.084
        # thresholds = {1: 4, 2: 1000, 3: 150, 4: 30}   # disabled; do not override dataset-specific thresholds
        # thresholds = {1: 6, 2: 1000, 3: 200, 4: 100}     0.365,0.082
        # thresholds = {1: 4, 2: 1000, 3: 200, 4: 200}   #0.312,0.084
        # thresholds = {1: 4, 2: 2000, 3: 100, 4: 200}   0.358,0.078
        # thresholds = {1: 4, 2: 1500, 3: 1000, 4: 200}   0.337,0.052
        # thresholds = {1: 4, 2: 1500, 3: 1000, 4: 400}   0.311
        # thresholds = {1: 3, 2: 2500, 3: 1000, 4: 400}   0.335,0.070
        # thresholds = {1: 3, 2: 1900, 3: 1000, 4: 500}   0.351,0.062
        # thresholds = {1: 4, 2: 2500, 3: 1500, 4: 300}   0.311,0.045
        exclude_punct = {'.', ',', ';', ':', '?', '!'}

        vocab = {
            phrase for phrase, freq in ngram_counter.items()
            if len((words := phrase.split())) in thresholds
               and (len(words) == 1 or not any(w in exclude_punct for w in words))
               and freq > thresholds[len(words)]
        }

        # [MIMIC-HARD-FILTER-PHRASE-TOKENS-20260603]
        if getattr(self, 'dataset_name', '') == 'mimic_cxr':
            # Hard filter phrase tokens for MIMIC-CXR.
            # The original n-gram vocabulary creates phrase tokens such as
            # 'no acute cardiopulmonary process', which caused template repetition
            # and unstable diffusion sampling on MIMIC.
            vocab = {tok for tok in vocab if ' ' not in str(tok)}

        vocab.add('<unk>')
        vocab = sorted(vocab)

        token2idx = {token: idx + 1 for idx, token in enumerate(vocab)}
        idx2token = {idx: token for token, idx in token2idx.items()}

        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        # Return embedding/output size, not the number of stored tokens.
        # Token ids are not guaranteed to be contiguous from 0 because pad=0
        # is reserved and idx2token starts from 1. Therefore the safe size is max_id + 1.
        return max(self.idx2token.keys()) + 1

    def __call__(self, report):
        text = self.clean_report(report).lower()
        words = text.split()
        i = 0
        tokens = []

        while i < len(words):
            matched = False
            for n in [4, 3, 2, 1]:
                if i + n <= len(words):
                    phrase = ' '.join(words[i:i + n])
                    if phrase in self.token2idx:
                        tokens.append(phrase)
                        i += n
                        matched = True
                        break
            if not matched:
                tokens.append('<unk>')
                i += 1

        ids = [self.cls_token_id] + [self.get_id_by_token(t) for t in tokens] + [self.eos_token_id]
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            idx = int(idx)

            # Stop at true sequence boundary.
            if idx == self.eos_token_id or idx == self.pad_token_id:
                break

            # Skip non-text special tokens.
            if idx in [
                getattr(self, "cls_token_id", -1),
                getattr(self, "mask_token_id", -1),
            ]:
                continue

            token = self.idx2token.get(idx, self.unk_token)

            # Do not surface unknown/special tokens in decoded reports.
            if token in ["<cls>", "<eos>", "<pad>", "<mask>", "<unk>"]:
                continue

            tokens.append(token)

        return ' '.join(tokens)

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

    def is_valid_id(self, idx):
        return idx in self.idx2token

# ---- Added for MIMIC-CXR report tokenization in revision CE experiments ----
# The original reproduced project only defined clean_report_iu_xray, while
# Tokenizer.__init__ expects clean_report_mimic_cxr when dataset_name=mimic_cxr.
def _sedrrg_clean_report_mimic_cxr(self, report):
    import re
    if report is None:
        report = ""
    report = str(report)
    report = report.replace("\n", " ")
    report = re.sub(r"_{2,}", " ", report)  # remove MIMIC de-identification blanks
    report = re.sub(r"\s+", " ", report)
    report = report.strip().lower()
    report = re.sub(r"([.,!?;:()])", r" \1 ", report)
    report = re.sub(r"\s+", " ", report).strip()
    return report

if not hasattr(Tokenizer, "clean_report_mimic_cxr"):
    Tokenizer.clean_report_mimic_cxr = _sedrrg_clean_report_mimic_cxr
# ---- End MIMIC-CXR tokenizer patch ----
