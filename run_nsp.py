#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm

https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForNextSentencePrediction
use self-dataset class for Next Sentence Prediction
one sentence Vs 4 emotions
i love this movie [SEP] It was happy
i love this movie [SEP] It was neutral
i love this movie [SEP] It was sad
i love this movie [SEP] It was anger
"""
import os
import sys
import fcntl
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
    BertForNextSentencePrediction,
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

    def __getitem__(self, index):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, wd, wd, wd, wd], 0s padded
        - attn_masks   : (L, ), ie., [1, 1, 1, 1, 0, 0]
        - labels   : 有点奇怪
            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        """
        target, text1, text2 = self.instances[index]
        encode = self.tokenizer(text1, text2, return_tensors='pt')
        if int(target) == 0:
            target = 1
        else:
            target = 0
        # text input, get tokenized  
        example = {}
        example['label'] = torch.tensor(target, dtype=torch.long)
        example['input_ids'] =  encode['input_ids'][0]
        example['token_type_ids'] = encode['token_type_ids'][0]
        example['attention_mask'] = encode['attention_mask'][0]
        # print(text1, text2)
        # print(example['input_ids'])
        return example

def bert_id2token(tokenizer, batch_ids):
    batch_texts = []
    for ids in batch_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        new_tokens = []
        for token in tokens:
            if token == '[PAD]':
                continue
            else:
                new_tokens.append(token)
        batch_texts.append(' '.join(new_tokens))
    return batch_texts

def nsp_collate(inputs):
    """
    Jinming: modify to img_position_ids
    Return:
    """
    labels = [sample['label'] for sample in inputs]
    input_ids = [sample['input_ids'] for sample in inputs]
    attention_masks = [sample['attention_mask'] for sample in inputs]
    token_type_ids = [sample['token_type_ids'] for sample in inputs]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=1).to(device)
    attn_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0).to(device)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'token_type_ids': token_type_ids,
             'attention_mask': attn_masks,
             'labels': labels
             }
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
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=nsp_collate, batch_size= model_args.per_device_batch_size)
    eval_dataloader = DataLoader(val_dataset, collate_fn=nsp_collate, batch_size=model_args.per_device_batch_size)

    logger.info("Modeling %s", training_args)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = BertForNextSentencePrediction.from_pretrained(
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
        nsp_eval_results, eval_results = evaluation(model, eval_dataloader, tokenizer)
        logger.info('\t Epoch NSP {}: {}'.format(epoch, nsp_eval_results))
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
    model = BertForNextSentencePrediction.from_pretrained(
                training_args.output_dir,
                config=config)    
    model.to(device)
    model.eval()
    logger.info('[Final Evaluation]  on validation')
    output_path = os.path.join(training_args.output_dir, 'details.csv')
    nsp_eval_results, eval_results = evaluation(model, eval_dataloader, tokenizer, output_path)
    logger.info('Epoch {}: {}'.format(best_eval_epoch, eval_results))
    output_tsv = os.path.join(os.path.split(training_args.output_dir)[0], 'result.csv')
    if not os.path.exists(output_tsv):
        open(output_tsv, 'w').close()  # touch output_csv
    cvNo = int(training_args.output_dir.split('/')[-1])
    write_result_to_tsv(output_tsv, eval_results, cvNo)

def write_result_to_tsv(file_path, tst_log, cvNo):
    # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
    f_in = open(file_path)
    fcntl.flock(f_in.fileno(), fcntl.LOCK_EX) # 加锁
    content = f_in.readlines()
    if len(content) != 12:
        content += ['\n'] * (12-len(content))
    content[cvNo-1] = 'CV{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(cvNo, tst_log['wa'], tst_log['wf1'], tst_log['uar'])
    f_out = open(file_path, 'w')
    f_out.writelines(content)
    f_out.close()
    f_in.close()     

def compute_metrics(preds, label_ids):
    assert len(preds) == len(label_ids)
    acc = accuracy_score(label_ids, preds)
    uar = recall_score(label_ids, preds, average='macro')
    wf1 = f1_score(label_ids, preds, average='weighted')
    uwf1 = f1_score(label_ids, preds, average='macro')
    cm = confusion_matrix(label_ids, preds)
    return {'total':len(preds), "wa": acc, "uar": uar, "wf1": wf1, 'uwf1': uwf1, 'cm': cm}

def write_csv(filepath, data, delimiter):
    '''
    TSV is Tab-separated values and CSV, Comma-separated values
    :param data, is list
    '''
    import csv
    with open(filepath, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=delimiter)
        csv_writer.writerows(data)

def compute_argmax_metrics(total_scores, total_labels, total_texts, output_path):
    # 每四个是一组, 将每个样本属于‘是’的概率拿出来，4个类别候选中取概率最大的作为最终的判别
    all_instances = []
    total_preds = []
    total_targets = []
    assert len(total_scores) % 4 == 0
    print(len(total_scores), len(total_labels), len(total_texts))
    assert len(total_scores) == len(total_labels) == len(total_texts)
    for index in range(0, len(total_scores)):
        all_instances.append([total_texts[index], total_scores[index], total_labels[index]])
    for index in range(0, len(total_scores), 4):
        sub_scores = total_scores[index*4 : (index+1)*4]
        sub_labels = total_labels[index*4 : (index+1)*4]
        sub_texts = total_texts[index*4 : (index+1)*4]
        sum_emos = []
        sum_probs = []
        if len(sub_scores) == len(sub_labels) == len(sub_texts) == 4:
            for j in range(len(sub_scores)):
                emo_name = sub_texts[j].split('[SEP]')[1].split(' ')[3]
                if sub_labels[j] == 0:
                    total_targets.append(emo_name)
                sum_probs.append(sub_scores[j][0])
                sum_emos.append(emo_name)
            total_preds.append(sum_emos[np.argmax(sum_probs)])
    assert len(total_preds) == len(total_targets)
    total_preds_ids, total_targets_ids = [], []
    label_map = {'anger':0, 'happy':1, 'neutral':2, 'sad':3}
    for pred, target in zip(total_preds, total_targets):
        total_preds_ids.append(label_map[pred])
        total_targets_ids.append(label_map[target])
    val_results = compute_metrics(total_preds_ids, total_targets_ids)
    if output_path is not None:
        print(output_path)
        write_csv(output_path, all_instances, delimiter=';')
    return val_results

def evaluation(model, set_dataloader, tokenizer, output_path=None):
    # 需要整理一下结果，决定最后的结果
    total_preds = []
    total_scores = []
    total_labels = []
    total_texts = []
    for step, batch in enumerate(set_dataloader):
        outputs = model(**batch)
        scores =  torch.softmax(outputs.logits, dim=1)
        labels = batch['labels'].detach().cpu().numpy()
        temp_preds = scores.argmax(axis=1).detach().cpu().numpy()
        total_preds.append(temp_preds)
        total_labels.append(labels)
        total_scores.append(scores.detach().cpu().numpy())
        input_ids = batch['input_ids'].detach().cpu().numpy()
        batch_texts = bert_id2token(tokenizer, input_ids)
        total_texts.append(batch_texts)
    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)
    total_texts = np.concatenate(total_texts)
    total_scores = np.concatenate(total_scores)
    nsp_eval_metric = compute_metrics(total_preds, total_labels)
    emo_eval_metric = compute_argmax_metrics(total_scores, total_labels, total_texts, output_path)
    return nsp_eval_metric, emo_eval_metric

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(0))
    main()