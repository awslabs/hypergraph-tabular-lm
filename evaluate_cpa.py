import sys
import re
import json
import logging
import numpy as np
import os.path as osp
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import transformers
from transformers.optimization import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from typing import Optional
from model import Encoder
from data import BipartiteData
from dataclasses import dataclass, field, fields
from torchmetrics import Precision, Recall, F1Score, AveragePrecision



def vocab2lbl(label_file):
    lbl_dict = {}
    with open(label_file, 'r') as f:
        for line in f.readlines():
            id, name = line.strip().split()
            lbl_dict[name] = int(id)
    return lbl_dict




class CPAHyperGraphDataset(InMemoryDataset):
    def __init__(self, data_args, tokenizer, type):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.type = type
        self.raw_data = json.load(open(osp.join(self.data_args.data_path,'preprocessed_{}.json'.format(self.type, 'r'))))
        super().__init__(self.data_args.data_path)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['{}_data.pt'.format(self.type)]

    def process(self):
        # Read data into huge `Data` list.
        data_list = self._table2graph(self.raw_data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



    def _tokenize_word(self, word):
        # refer to numBERT: https://github.com/google-research/google-research/tree/master/numbert
        number_pattern = re.compile(
            r"(\d+)\.?(\d*)")  # Matches numbers in decimal form.
        def number_repl(matchobj):
            """Given a matchobj from number_pattern, it returns a string writing the corresponding number in scientific notation."""
            pre = matchobj.group(1).lstrip("0")
            post = matchobj.group(2)
            if pre and int(pre):
                # number is >= 1
                exponent = len(pre) - 1
            else:
                # find number of leading zeros to offset.
                exponent = -re.search("(?!0)", post).start() - 1
                post = post.lstrip("0")
            return (pre + post).rstrip("0") + " scinotexp " + str(exponent)
        
        def apply_scientific_notation(line):
            """Convert all numbers in a line to scientific notation."""
            res = re.sub(number_pattern, number_repl, line)
            return res
        
        word = apply_scientific_notation(word)        
        wordpieces = self.tokenizer.tokenize(word)[:self.data_args.max_token_length]

        mask = [1 for _ in range(len(wordpieces))]
        while len(wordpieces)<self.data_args.max_token_length:
            wordpieces.append('[PAD]')
            mask.append(0)
        return wordpieces, mask

    def _table2graph(self, examples):
        data_list = []
        pos_count = torch.zeros(self.data_args.label_type_num, dtype=torch.float32)
        for exm in tqdm(examples):
            try:
                tb = exm['table']
            except:
                tb = exm
            cap = '' if  tb['caption'] is None else tb['caption']
            cap = ' '.join(cap.split()[:self.data_args.max_token_length]) # filter too long caption
            header = [' '.join(h['name'].split()[:self.data_args.max_token_length]) for h in tb['header']][:self.data_args.max_column_length]
            data = [row[:self.data_args.max_column_length] for row in tb['data'][:self.data_args.max_row_length]]
            labels = tb['label_ids']
            assert len(data[0]) == len(header)

            wordpieces_xs_all, mask_xs_all = [], []
            wordpieces_xt_all, mask_xt_all = [], []
            nodes, edge_index = [], []


            # caption to hyper-edge (t node)
            if not cap:
                wordpieces = ['[TAB]'] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
                mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)
            else:
                wordpieces, mask = self._tokenize_word(cap)
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)


            # header to hyper-edge (t node)
            for head in header:
                if not head:
                    wordpieces = ['[HEAD]'] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
                    mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                    wordpieces_xt_all.append(wordpieces)
                    mask_xt_all.append(mask)
                else:
                    wordpieces, mask = self._tokenize_word(head)
                    wordpieces_xt_all.append(wordpieces)
                    mask_xt_all.append(mask)


            # row to hyper edge (t node)
            for i in range(len(data)):
                wordpieces = ['[ROW]'] + ['[PAD]' for _ in range(self.data_args.max_token_length- 1)]
                mask = [1] + [0 for _ in range(self.data_args.max_token_length- 1)]
                wordpieces_xt_all.append(wordpieces)
                mask_xt_all.append(mask)

            # cell to nodes (s node)
            for row_i, row in enumerate(data):
                for col_i, word in enumerate(row):
                    if not word:
                        wordpieces = ['[CELL]'] + ['[PAD]' for _ in range(self.data_args.max_token_length - 1)]
                        mask = [1] + [0 for _ in range(self.data_args.max_token_length - 1)]
                    else:
                        word = ' '.join(word.split()[:self.data_args.max_token_length])
                        wordpieces, mask = self._tokenize_word(word)
                    wordpieces_xs_all.append(wordpieces)
                    mask_xs_all.append(mask)
                    node_id = len(nodes)
                    nodes.append(node_id)
                    edge_index.append([node_id, 0]) # connect to table-level hyper-edge
                    edge_index.append([node_id, col_i+1]) # # connect to col-level hyper-edge
                    edge_index.append([node_id, row_i + 1 + len(header)])  # connect to row-level hyper-edge

            # add label
            label_ids = torch.zeros((len(header)-1, self.data_args.label_type_num), dtype=torch.float32)
            assert len(label_ids) == len(labels) == len(header) -1
            col_mask = [0 for i in range(len(wordpieces_xt_all))]
            col_mask[1:1+len(header)] = [1]*len(header)

            for col_i, lbl in enumerate(labels):
                for lbl_i in lbl:
                    label_ids[col_i, lbl_i] = 1.0
                    pos_count[lbl_i] += 1

            xs_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xs_all], dtype=torch.long)
            xt_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(x) for x in wordpieces_xt_all], dtype=torch.long)


            # check all 0 input
            xs_tem = torch.count_nonzero(xs_ids, dim =1)
            xt_tem = torch.count_nonzero(xt_ids, dim=1)
            assert torch.count_nonzero(xs_tem) == len(xs_tem)
            assert torch.count_nonzero(xt_tem) == len(xt_tem)
            edge_index = torch.tensor(edge_index, dtype=torch.long).T
            bigraph = BipartiteData(edge_index=edge_index, x_s=xs_ids, x_t=xt_ids, y=label_ids, col_mask = col_mask, num_nodes=len(xs_ids), num_hyperedges= len(xt_ids))
            data_list.append(bigraph)
        return data_list



class CPADataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer,
            data_args,
            seed,
            batch_size,
            py_logger):

        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.seed = seed
        self.batch_size = batch_size
        self.py_logger = py_logger
        self.py_logger.info("Using tabular data collator")


    def json2table(self, data_dir, lbl_dict):
        samples = []
        with open(data_dir, 'r') as f:
            data = json.load(f)
        rows_count, columns_count = [], []
        for tb in tqdm(data):
            id = tb[0]
            cap = ' '.join([tb[4], tb[1], tb[3]]).strip()
            heads = tb[5]
            content = tb[6]
            labels = tb[-1]
            assert len(labels) == len(heads)-1
            label_ids = []
            for lbl in labels:
                label_ids.append([lbl_dict[l] for l in lbl])
            row_count = max([cell[0][0] for col in content for cell in col]) + 1
            rows_count.append(row_count)
            columns_count.append(len(heads))

            data = [['' for _ in range(len(heads))] for _ in range(row_count)]
            for col in content:
                for cell in col:
                    i, j = cell[0][0], cell[0][1]
                    data[i][j] = cell[1][1]

            # cut long headers
            if len(heads) > self.data_args.max_column_length:
                def divide_chunks(l, n):
                    # looping till length l
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                heads_list = list(divide_chunks(heads[1:], self.data_args.max_column_length-1))
                for l in heads_list:
                    l.insert(0, heads[0])
                labels_list = list(divide_chunks(labels, self.data_args.max_column_length-1))
                label_ids_list = list(divide_chunks(label_ids, self.data_args.max_column_length-1))
                
                data_list = [[] for _ in range(len(heads_list))]
                for row in data:
                    row_list = list(divide_chunks(row[1:], self.data_args.max_column_length-1))
                    for l in row_list:
                        l.insert(0, row[0])
                    for i in range(len(row_list)):
                        data_list[i].append(row_list[i])
            else:
                heads_list = [heads]
                labels_list = [labels]
                label_ids_list = [label_ids]
                data_list = [data]

            for i in range(len(heads_list)):
                heads = [{'name': h} for h in heads_list[i]]
                sample = {'uuid':id, 'table':{'caption':cap, 'header': heads, 'data': data_list[i], 'labels': labels_list[i], 'label_ids': label_ids_list[i]}}
                samples.append(sample)
        return samples

    def prepare_data(self) -> None:
        self.py_logger.info(f"Preparing data... \n")

        if not osp.exists(osp.join(self.data_args.data_path,'preprocessed_train.json')):

            label_file = self.data_args.data_path + 'relation_vocab.txt'
            train_data_dir = self.data_args.data_path + 'train.table_rel_extraction.json'
            valid_data_dir = self.data_args.data_path + 'dev.table_rel_extraction.json'
            test_data_dir = self.data_args.data_path + 'test.table_rel_extraction.json'

            lbl_dict = vocab2lbl(label_file)
            train_samples = self.json2table(train_data_dir, lbl_dict)
            valid_samples = self.json2table(valid_data_dir, lbl_dict)
            test_samples = self.json2table(test_data_dir, lbl_dict)

            json.dump(train_samples, open(osp.join(self.data_args.data_path,'preprocessed_train.json'),'w'))
            json.dump(valid_samples, open(osp.join(self.data_args.data_path, 'preprocessed_valid.json'), 'w'))
            json.dump(test_samples, open(osp.join(self.data_args.data_path, 'preprocessed_test.json'), 'w'))


    def setup(self, stage):
        self.py_logger.info(f"Setting up... \n")
        self.train_dataset = CPAHyperGraphDataset(self.data_args, self.tokenizer, 'train')
        self.valid_dataset = CPAHyperGraphDataset(self.data_args, self.tokenizer, 'valid')
        self.test_dataset = CPAHyperGraphDataset(self.data_args, self.tokenizer, 'test')
        self.py_logger.info(
            "Training dataset size {}, validating dataset size {}, testing dataset size {},.".format(len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset)))

    def train_dataloader(self):
        """This will be run every episode."""

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.data_args.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.data_args.num_workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.data_args.num_workers)





class CPAClassifier(pl.LightningModule):

    def __init__(self,model_config, optimizer_cfg):
        super().__init__()
        self.model_config = model_config
        self.enc = Encoder(self.model_config)
        print('check point',optimizer_cfg.checkpoint_path)
        # for non-deepseepd
        # state_dict = torch.load(open(checkpoint_path, 'rb'))['state_dict']
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     if 'model' in k:
        #         name = k[6:] # remove `model.`
        #         new_state_dict[name] = v
        # self.enc.load_state_dict(new_state_dict, strict=True)
        
        # for deepspeed
        state_dict = torch.load(open(optimizer_cfg.checkpoint_path, 'rb'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['module'].items():
            if 'model' in k:
                name = k[13:] # remove `module.model.`
                new_state_dict[name] = v
        self.enc.load_state_dict(new_state_dict, strict=True)
        

        self.optimizer_cfg = optimizer_cfg
        self.fc1 = nn.Linear(768*2, 384)
        self.fc2 = nn.Linear(384, 121)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout()
        self.init_weights()
        self.loss_fct = BCEWithLogitsLoss()
        self.pre, self.rec, self.f1, self.avg_pre = Precision(threshold=0.5, average='micro'), Recall(threshold=0.5, average='micro'), F1Score(threshold=0.5, average='micro'), AveragePrecision()
        

    def init_weights(self):
        nn.init.xavier_uniform(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_uniform(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

    def training_step(self, batch, batch_idx):
        batch_size = len(batch.col_mask)
        device = batch.edge_index.device
        outputs= self.enc(batch)
        hyperedge_outputs = outputs[1]
        
        all_hyperedge_embeds = torch.split(hyperedge_outputs, batch.num_hyperedges.cpu().tolist())
        pairs = []
        for batch_i in range(batch_size):
            hyperedge_embeds = all_hyperedge_embeds[batch_i]
            col_embeds = torch.index_select(hyperedge_embeds, 0, torch.nonzero(torch.LongTensor(batch.col_mask[batch_i])).squeeze().to(device))
            obj_col_embeds = col_embeds[1:]
            sub_col_embeds = col_embeds[0].repeat(obj_col_embeds.shape[0], 1)
            pair_embeds = torch.cat((sub_col_embeds, obj_col_embeds), dim=1)
            pairs.append(pair_embeds)
        all_pair_embeds = torch.cat(pairs, dim=0)    
        labels = batch.y
        assert all_pair_embeds.size(0) == labels.size(0)
        logits = self.fc2(self.dropout(self.relu(self.fc1(all_pair_embeds))))
        loss = self.loss_fct(logits, labels)
        self.log_dict({'train_loss': loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = len(batch.col_mask)
        device = batch.edge_index.device
        outputs= self.enc(batch)
        hyperedge_outputs = outputs[1]
        
        all_hyperedge_embeds = torch.split(hyperedge_outputs, batch.num_hyperedges.cpu().tolist())
        pairs = []
        for batch_i in range(batch_size):
            hyperedge_embeds = all_hyperedge_embeds[batch_i]
            col_embeds = torch.index_select(hyperedge_embeds, 0, torch.nonzero(torch.LongTensor(batch.col_mask[batch_i])).squeeze().to(device))
            obj_col_embeds = col_embeds[1:]
            sub_col_embeds = col_embeds[0].repeat(obj_col_embeds.shape[0], 1)
            pair_embeds = torch.cat((sub_col_embeds, obj_col_embeds), dim=1)
            pairs.append(pair_embeds)
        all_pair_embeds = torch.cat(pairs, dim=0)    
        labels = batch.y
        assert all_pair_embeds.size(0) == labels.size(0)
        logits = self.fc2(self.dropout(self.relu(self.fc1(all_pair_embeds))))
        loss = self.loss_fct(logits, labels)

        self.log("validation_loss", loss, prog_bar=True)

        return {"logits": logits, "labels": labels}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([out["logits"] for out in outputs], dim=0)
        labels = torch.cat([out["labels"] for out in outputs], dim=0).long()
        probs = torch.sigmoid(logits)
        precision, recall, f1_score, avg_precision = self.pre(probs, labels), self.rec(probs, labels), self.f1(probs, labels), self.avg_pre(probs, labels)
        self.log_dict({'val_f1': f1_score, 'val_avg_accuracy': avg_precision,
                       'val_precision': precision, 'val_recall': recall}, prog_bar=True)


    def test_step(self, batch, batch_ix):
        batch_size = len(batch.col_mask)
        device = batch.edge_index.device
        outputs= self.enc(batch)
        hyperedge_outputs = outputs[1]
        
        all_hyperedge_embeds = torch.split(hyperedge_outputs, batch.num_hyperedges.cpu().tolist())
        pairs = []
        for batch_i in range(batch_size):
            hyperedge_embeds = all_hyperedge_embeds[batch_i]
            col_embeds = torch.index_select(hyperedge_embeds, 0, torch.nonzero(torch.LongTensor(batch.col_mask[batch_i])).squeeze().to(device))
            obj_col_embeds = col_embeds[1:]
            sub_col_embeds = col_embeds[0].repeat(obj_col_embeds.shape[0], 1)
            pair_embeds = torch.cat((sub_col_embeds, obj_col_embeds), dim=1)
            pairs.append(pair_embeds)
        all_pair_embeds = torch.cat(pairs, dim=0)    
        labels = batch.y
        assert all_pair_embeds.size(0) == labels.size(0)
        logits = self.fc2(self.dropout(self.relu(self.fc1(all_pair_embeds))))
        loss = self.loss_fct(logits, labels)

        self.log("test_loss", loss, prog_bar=True)

        return {"logits": logits, "labels": labels}


    def test_epoch_end(self, outputs):
        logits = torch.cat([out["logits"] for out in outputs], dim=0)
        labels = torch.cat([out["labels"] for out in outputs], dim=0).long()
        probs = torch.sigmoid(logits)
        precision, recall, f1_score, avg_precision = self.pre(probs, labels), self.rec(probs, labels), self.f1(probs, labels), self.avg_pre(probs, labels)
        self.log_dict({'test_f1': f1_score, 'test_avg_accuracy': avg_precision,
                       'test_precision': precision, 'test_recall': recall}, prog_bar=True)



    # ---------------------
    # TRAINING SETUP
    # ---------------------

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.trainer.datamodule.train_dataloader()
        if self.trainer.max_steps!=-1:
            return self.trainer.max_steps
        dataset_size = len(dataset)*dataset.batch_size
        num_devices = max(1, self.trainer.num_gpus)
        effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
        return (dataset_size // effective_batch_size) * self.trainer.max_epochs

    def configure_optimizers(self):
        learning_rate = self.optimizer_cfg.base_learning_rate
        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {
                "params": params_decay,
                "weight_decay": self.optimizer_cfg.weight_decay,
            },
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = self.optimizer_cfg.get_optimizer(optim_groups, learning_rate)
        num_training_steps = self.num_training_steps()
        print('num steps:', num_training_steps)
        scheduler = get_scheduler(
            self.optimizer_cfg.lr_scheduler_type,
            optimizer,
            num_warmup_steps=int(self.optimizer_cfg.warmup_step_ratio*num_training_steps),
            num_training_steps=num_training_steps,
        )
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "validation_loss",
                }
            ],
        )





#********************************* set up arguments *********************************

@dataclass
class DataArguments:
    """
    Arguments pertaining to which config/tokenizer we are going use.
    """
    tokenizer_config_type: str = field(
        default='bert-base-uncased',
        metadata={
            "help": "bert-base-cased, bert-base-uncased etc"
        },
    )
    data_path: str = field(default='../table_graph/data/col_rel/', metadata={"help": "data path"})
    max_token_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input token length for cell/caption/header after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    max_row_length: int = field(
        default=30,
        metadata={
            "help": "The maximum total input rows for a table"
        },
    )
    max_column_length: int = field(
        default=20,
        metadata={
            "help": "The maximum total input columns for a table"

        },
    )
    label_type_num: int = field(
        default=121,
        metadata={
            "help": "The total label types"

        },
    )

    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"},
    )

    valid_ratio: float = field(
        default=0.3,
        metadata={"help": "Number of workers for dataloader"},
    )


    def __post_init__(self):
        if self.tokenizer_config_type not in ["bert-base-cased", "bert-base-uncased"]:
            raise ValueError(
                f"The model type should be bert-base-(un)cased. The current value is {self.tokenizer_config_type}."
            )



@dataclass
class OptimizerConfig:
    batch_size: int = 256
    base_learning_rate: float = 1e-3
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: transformers.SchedulerType = "linear"
    warmup_step_ratio: float = 0.1
    seed: int = 42
    optimizer: str = "Adam"
    adam_w_mode: bool = True
    save_every_n_epochs: int=1
    save_top_k: int=1
    checkpoint_path: str=''


    def __post_init__(self):
        if self.optimizer.lower() not in {
            "adam",
            "fusedadam",
            "fusedlamb",
            "fusednovograd",
        }:
            raise KeyError(
                f"The optimizer type should be one of: Adam, FusedAdam, FusedLAMB, FusedNovoGrad. The current value is {self.optimizer}."
            )

    def get_optimizer(self, optim_groups, learning_rate):
        optimizer = self.optimizer.lower()
        optim_cls = {
            "adam": AdamW if self.adam_w_mode else Adam,
        }[optimizer]

        args = [optim_groups]
        kwargs = {
            "lr": learning_rate,
            "eps": self.adam_epsilon,
            "betas": (self.adam_beta1, self.adam_beta2),
        }
        if optimizer in {"fusedadam", "fusedlamb"}:
            kwargs["adam_w_mode"] = self.adam_w_mode

        optimizer = optim_cls(*args, **kwargs)
        return optimizer


def evaluate():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    py_logger = logging.getLogger(__name__)
    py_logger.setLevel(logging.INFO)
    tb_logger = TensorBoardLogger("logs", name="evaluate_cpa")

    # ********************************* parse arguments *********************************
    parser = HfArgumentParser((DataArguments, OptimizerConfig))
    parser = pl.Trainer.add_argparse_args(parser)

    (
        data_args,
        optimizer_cfg,
        trainer_args,
    ) = parser.parse_args_into_dataclasses()
    py_logger.info(f"data_args: {data_args}\n")
    py_logger.info(f"optimizer_cfg: {optimizer_cfg}\n")
    py_logger.info(f"trainer_args: {trainer_args}\n")


    # ********************************* set up tokenizer and model config*********************************
    # custom BERT tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_config_type)  
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    py_logger.info(f"new tokens added: {new_tokens}\n")
    tokenizer.add_tokens(new_tokens)
    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    model_config.update({'vocab_size': len(tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})
    py_logger.info(f"model config: {model_config}\n")

    data_module = CPADataModule(tokenizer=tokenizer,
                                  data_args = data_args,
                                  seed=optimizer_cfg.seed,
                                  batch_size=optimizer_cfg.batch_size,
                                  py_logger=py_logger
                                  )

    # ********************************* set up model module *********************************
    model_module = CPAClassifier(model_config, optimizer_cfg)

    # ********************************* set up trainer *********************************
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_epochs=optimizer_cfg.save_every_n_epochs,
            save_top_k=optimizer_cfg.save_top_k,
            monitor="val_f1",
            mode = "max"
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    trainer = pl.Trainer.from_argparse_args(
        trainer_args,
        callbacks=callbacks,
        logger = tb_logger,
    )

    trainer.fit(model_module, data_module)
    trainer.validate(ckpt_path="best", datamodule = data_module)
    trainer.test(ckpt_path="best", datamodule = data_module)



if __name__ == '__main__':
    print(111111)
    import warnings
    from pytorch_lightning import  seed_everything

    warnings.filterwarnings("ignore")
    seed = 42
    seed_everything(seed, workers=True)
    evaluate()
