#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm

use self-dataset class
"""
import os
import sys
import numpy as np
from dataclasses import dataclass, field
import torch.utils.data as data
from typing import Optional
import torch
import transformers
from transformers import (
    AdamW,
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    get_scheduler,
    set_seed,
)
from transformers.models.auto.tokenization_auto import logger
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from logger import get_logger

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    per_device_batch_size: Optional[int] = field(
        default=32,
        metadata={
            "help": 'batchsize'
        },
    )
    # num_train_epochs: Optional[int] = field(default=6, metadata={"help": 'max training epochs'},)
    # weight_decay: Optional[float] = field(default=0.0001, metadata={"help": 'weight_decay'},)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )

def read_csv(filepath, delimiter, skip_rows=0):
    '''
    default: can't filter the csv head
    :param filepath:
    :param delimiter:
    :return:
    '''
    import csv
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = csv.reader(f, delimiter=delimiter)
        lines = [line for line in lines]
        return lines[skip_rows:]

class SelfPrompt_dataset(data.Dataset):
    def __init__(self, text_filetpath, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.instances = read_csv(text_filetpath, delimiter=',', skip_rows=1)
        self.cls_ = 101
        self.sep_ = 102
        self.mask_ = 103
        self.unk_  = 100
        self._label_pad = -100 

    def __len__(self):
        return len(self.instances)

    def bert_tokenize(self, text):
        ids = []
        for word in text.strip().split():
            ws = self.tokenizer.tokenize(word)
            ids.extend(self.tokenizer.convert_tokens_to_ids(ws))
        return ids

    def __getitem__(self, index):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep], 0s padded
        - attn_masks   : (L, ), ie., [1, 1, 1, 1, 0, 0]
        - txt_labels   : (L, ), [-100, -100, wid, -100]
        """
        label_name, text = self.instances[index]
        input_ids = self.bert_tokenize(text)
        target_id = self.bert_tokenize(label_name)
        # print(text, input_ids)
        # print(label_name, target_id)
        assert len(target_id) == 1
        # text input, get tokenized  
        input_ids, txt_labels = self.create_mlm_io(input_ids, target_id[0])
        attn_masks = torch.ones(len(input_ids), dtype=torch.long)
        example = {}
        example['input_ids'] = input_ids
        example['attn_masks'] = attn_masks
        example['txt_labels'] = txt_labels
        return example

    def create_mlm_io(self, input_ids, target_id):
        txt_labels = []
        for input_id in input_ids:
            if input_id == self.mask_:
                txt_labels.append(target_id)
            else:
                txt_labels.append(self._label_pad)
        input_ids = torch.tensor([self.cls_]
                                 + input_ids
                                 + [self.sep_], dtype=torch.long)
        txt_labels = torch.tensor([self._label_pad] + txt_labels + [self._label_pad], dtype=torch.long)
        assert len(input_ids) == len(txt_labels)
        return input_ids, txt_labels

def mlm_collate(inputs):
    """
    Jinming: modify to img_position_ids
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    _label_pad = -100
    input_ids = [sample['input_ids'] for sample in inputs]
    attn_masks = [sample['attn_masks'] for sample in inputs]
    txt_labels = [sample['txt_labels'] for sample in inputs]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    # 文档里要求是-100 https://huggingface.co/transformers/model_doc/bert.html#transformers.models.bert.modeling_bert.BertForMaskedLM
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=_label_pad).to(device)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0).to(device)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(device)
    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'attention_mask': attn_masks,
             'labels': txt_labels}
    return batch

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    log_dir = os.path.join(training_args.output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger(log_dir, suffix='none')

    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)   
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    print(tokenizer.vocab_size)

    logger.info("Training/evaluation dataloader %s", training_args)
    train_dataset = SelfPrompt_dataset(data_args.train_file, tokenizer)
    val_dataset = SelfPrompt_dataset(data_args.validation_file, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=mlm_collate, batch_size= model_args.per_device_batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=mlm_collate, batch_size=model_args.per_device_batch_size)

    logger.info("Modeling %s", training_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config)
    model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    max_train_steps = int(training_args.num_train_epochs * len(train_dataset) / model_args.per_device_batch_size)
    lr_scheduler = get_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {model_args.per_device_batch_size}")
    logger.info(f"  Num steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    select_metrix = 'uar'
    best_eval_metrix = 0
    best_eval_epoch = -1
    for epoch in range(int(training_args.num_train_epochs)):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        logger.info(f'\t[LR] current {epoch} learning rate {lr}')
        model.train()
        # https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForMaskedLM
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        # evaluation at every epoch
        model.eval()
        logger.info('[Evaluation]  on validation')
        eval_results = evaluation(model, eval_dataloader)
        logger.info('\t Epoch {}: {}'.format(epoch, eval_results))
        # choose the best epoch
        if eval_results[select_metrix] > best_eval_metrix:
            best_eval_epoch = epoch
            best_eval_metrix = eval_results[select_metrix]
            # 保存为 Hugging-face 的格式
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

    # print best eval result and clean the other models
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    model = AutoModelForMaskedLM.from_pretrained(
                training_args.output_dir,
                config=config)    
    model.to(device)
    model.eval()
    logger.info('[Final Evaluation]  on validation')
    eval_results = evaluation(model, eval_dataloader)
    logger.info('Epoch {}: {}'.format(best_eval_epoch, eval_results))

def compute_metrics(preds, label_ids):
    assert len(preds) == len(label_ids)
    acc = accuracy_score(label_ids, preds)
    uar = recall_score(label_ids, preds, average='macro')
    wf1 = f1_score(label_ids, preds, average='weighted')
    uwf1 = f1_score(label_ids, preds, average='macro')
    cm = confusion_matrix(label_ids, preds)
    return {'total':len(preds), "acc": acc, "uar": uar, "wf1": wf1, 'uwf1': uwf1, 'cm': cm}

def evaluation(model, set_dataloader, return_pred=False):
    total_preds = []
    total_labels = []
    for step, batch in enumerate(set_dataloader):
        outputs = model(**batch)
        scores = outputs.logits
        labels = batch['labels']
        scores = scores.view(-1, model.config.vocab_size)
        labels = labels.view(-1)
        scores = scores[labels != -100]
        labels = labels[labels != -100].detach().cpu().numpy()
        # print(scores.shape, labels.shape)
        temp_preds = scores.argmax(axis=1).detach().cpu().numpy()
        total_preds.append(temp_preds)
        total_labels.append(labels)
    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)
    # set early stop and save the best models
    eval_metric = compute_metrics(total_preds, total_labels)
    return eval_metric

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0))
    main()