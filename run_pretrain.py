import os
import sys
import logging

import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy

import transformers
from transformers import AutoTokenizer, AutoConfig,HfArgumentParser
from transformers.optimization import AdamW, get_scheduler

from dataclasses import dataclass, field, fields
from typing import Optional

from model import Encoder, ContrastiveLoss
from data import TableDataModule



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
    data_path: str = field(default='./data/pretrain/', metadata={"help": "data path"})
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

    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"},
    )

    valid_ratio: float = field(
        default=0.01,
        metadata={"help": "Number of workers for dataloader"},
    )

    seed: int = 42
    max_epoch: int = 5
    electra: bool = False
    mask_ratio: float = 0.15
    contrast_bipartite_edge: bool = False 
    bipartite_edge_corrupt_ratio: float=0.3



@dataclass
class OptimizerConfig:
    
    batch_size: int = 128
    base_learning_rate: float = 1e-4  
    weight_decay: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: transformers.SchedulerType = "linear"
    warmup_step_ratio: float = 0.05
    optimizer: str = "Adam"
    adam_w_mode: bool = True
    save_every_n_epochs: int=1
    save_top_k: int=3
    @classmethod
    def dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

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


class PlModel(pl.LightningModule):
  def __init__(
          self,
          model_config,
          optimizer_cfg,
  ):
    super().__init__()
    self.model = Encoder(model_config)
    self.model_config = model_config
    self.optimizer_cfg = optimizer_cfg
    self.save_hyperparameters()

    # electra head
    if self.model_config.electra:
        self.dense = nn.Linear(self.model_config.hidden_size, self.model_config.hidden_size)
        self.act = nn.GELU()
        self.dense_prediction = nn.Linear(self.model_config.hidden_size, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.pre, self.rec, self.f1, self.acc = BinaryPrecision(threshold=0.5), BinaryRecall(threshold=0.5), BinaryF1Score(threshold=0.5), BinaryAccuracy(threshold=0.5)

    # contrast loss
    elif self.model_config.contrast_bipartite_edge:
        self.con_loss = ContrastiveLoss(temperature=0.07)


  def training_step(self, batch, batch_idex):

    # electra objective
    if self.model_config.electra:
        outputs = self.model(batch)
        cell_embeds = outputs[0]
        hyperedge_outputs = outputs[1]
        col_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(batch.col_mask).squeeze())
        all_embeds = torch.cat([cell_embeds, col_embeds], axis =0)
        hidden_states = self.dense(all_embeds)
        hidden_states = self.act(hidden_states)
        logits = self.dense_prediction(hidden_states).view(-1)
        c_lbls = batch.electra_c
        h_lbls = batch.electra_h
        lbls = torch.cat([c_lbls, h_lbls])
        loss_pos = self.criterion(logits[lbls==1.], lbls[lbls==1.])
        loss_neg = self.criterion(logits[lbls==0.], lbls[lbls==0.])
        electra_loss = loss_pos + loss_neg
        loss = electra_loss
        
    # contrast_bipartite_edge objective
    elif self.model_config.contrast_bipartite_edge:
        self.model_config.update({'edge_neg_view':1})
        outputs1 = self.model(batch)
        hyperedge_outputs1 = outputs1[1]
        hyper_embeds1 = torch.index_select(hyperedge_outputs1, 0, torch.nonzero(batch.hyper_mask).squeeze())
        self.model_config.update({'edge_neg_view':2})
        outputs2 = self.model(batch)
        hyperedge_outputs2 = outputs2[1]
        hyper_embeds2 = torch.index_select(hyperedge_outputs2, 0, torch.nonzero(batch.hyper_mask).squeeze())
        con_edge_loss = self.con_loss(hyper_embeds1, hyper_embeds2)
        loss = con_edge_loss

    self.log("training_loss", loss, prog_bar=True)
    return loss


  def validation_step(self, batch, batch_idx):

    # electra objective
    if self.model_config.electra:
        outputs = self.model(batch)
        cell_embeds = outputs[0]
        hyperedge_outputs = outputs[1]
        col_embeds = torch.index_select(hyperedge_outputs, 0, torch.nonzero(batch.col_mask).squeeze())
        all_embeds = torch.cat([cell_embeds, col_embeds], axis=0)
        hidden_states = self.dense(all_embeds)
        hidden_states = self.act(hidden_states)
        electra_logits = self.dense_prediction(hidden_states).view(-1)
        c_lbls = batch.electra_c
        h_lbls = batch.electra_h
        electra_lbls = torch.cat([c_lbls, h_lbls])
        # calculate loss individually for pos/neg cases balance
        loss_pos = self.criterion(electra_logits[electra_lbls==1.], electra_lbls[electra_lbls==1.])
        loss_neg = self.criterion(electra_logits[electra_lbls==0.], electra_lbls[electra_lbls==0.])
        electra_loss = loss_pos + loss_neg
        self.log("validation_loss", electra_loss, prog_bar=True)
        return {"logits": electra_logits, "labels": electra_lbls}
    
    
    # contrast_bipartite_edge objective
    elif self.model_config.contrast_bipartite_edge:
        self.model_config.update({'edge_neg_view':1})
        outputs1 = self.model(batch)
        hyperedge_outputs1 = outputs1[1]
        hyper_embeds1 = torch.index_select(hyperedge_outputs1, 0, torch.nonzero(batch.hyper_mask).squeeze())
        self.model_config.update({'edge_neg_view':2})
        outputs2 = self.model(batch)
        hyperedge_outputs2 = outputs2[1]
        hyper_embeds2 = torch.index_select(hyperedge_outputs2, 0, torch.nonzero(batch.hyper_mask).squeeze())
        con_edge_loss = self.con_loss(hyper_embeds1, hyper_embeds2)
        self.log("validation_loss", con_edge_loss, prog_bar=True) 
        return con_edge_loss

        
        
  def validation_epoch_end(self, outputs):
        if self.model_config.electra:
            logits = torch.cat([out["logits"] for out in outputs], dim=0)
            labels = torch.cat([out["labels"] for out in outputs], dim=0).long()
            probs = torch.sigmoid(logits)
            precision, recall, f1_score, acc = self.pre(probs, labels), self.rec(probs, labels), self.f1(probs, labels), self.acc(probs, labels)
            self.log_dict({'val_f1': f1_score, 'acc':acc,
                        'val_precision': precision, 'val_recall': recall}, prog_bar=True)

  def num_training_steps(self) -> int:
    """Total training steps inferred from datamodule and devices."""
    dataset = self.trainer.datamodule.train_dataloader()
    if self.trainer.max_steps != -1:
      return self.trainer.max_steps
    num_devices = max(1, self.trainer.num_gpus)
    dataset_size = len(dataset) * dataset.batch_size * num_devices  
    effective_batch_size = dataset.batch_size * self.trainer.accumulate_grad_batches * num_devices
    return (dataset_size // effective_batch_size) * self.trainer.max_epochs

  def configure_optimizers(self):
    from dataclasses import asdict
    self.logger.log_hyperparams(asdict(self.optimizer_cfg))
    # Infer learning rate
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
    scheduler = get_scheduler(
      self.optimizer_cfg.lr_scheduler_type,
      optimizer,
      num_warmup_steps=int(self.optimizer_cfg.warmup_step_ratio * num_training_steps),
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




def main():
    # ********************************* set up logger *********************************

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    py_logger = logging.getLogger(__name__)
    py_logger.setLevel(logging.INFO)

    tb_logger = TensorBoardLogger("logs", name="pretrain", default_hp_metric=True)

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

    # Set env variable for using tokenizer in multiple process dataloader
    if data_args.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Set seed before initializing model.
    pl.utilities.seed.seed_everything(data_args.seed)

    # ********************************* set up tokenizer and model config*********************************
    # custom tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_config_type) 
    new_tokens = ['[TAB]', '[HEAD]', '[CELL]', '[ROW]', "scinotexp"]
    py_logger.info(f"new tokens added: {new_tokens}\n")
    tokenizer.add_tokens(new_tokens)
    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    model_config.update({'vocab_size': len(tokenizer), "pre_norm": False, "activation_dropout":0.1, "gated_proj": False})
    model_config.update({"electra": data_args.electra, "contrast_bipartite_edge": data_args.contrast_bipartite_edge})
    py_logger.info(f"model config: {model_config}\n")

    # ********************************* set up data module *********************************
    data_module = TableDataModule(tokenizer=tokenizer,
                                  data_args = data_args,
                                  seed=data_args.seed,
                                  batch_size=optimizer_cfg.batch_size,
                                  py_logger=py_logger, 
                                  objective = 'electra' if model_config.electra else 'contrast'
                                  )
    # ********************************* set up model module *********************************
    model_module = PlModel(model_config, optimizer_cfg)

    # ********************************* set up trainer callbacks and plugins *********************************
    if model_config.electra: 
        callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_epochs=optimizer_cfg.save_every_n_epochs,
            save_top_k=optimizer_cfg.save_top_k,
            monitor="val_f1",
            mode = 'max'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]
    elif model_config.contrast_bipartite_edge:  
        callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_epochs=optimizer_cfg.save_every_n_epochs,
            save_top_k=optimizer_cfg.save_top_k,
            monitor="validation_loss",
            mode = 'min'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ]

    plugins = []
    if trainer_args.gpus == -1:
        trainer_args.gpus = torch.cuda.device_count()
    assert trainer_args.replace_sampler_ddp == False # We need to set replace_sampler_ddp False to use SequentialSampler, so that each GPU only pass part of the dataset
    # ********************************* set up trainer *********************************
    trainer = pl.Trainer.from_argparse_args(
        trainer_args,
        strategy="deepspeed_stage_1",
        callbacks=callbacks,
        plugins=plugins,
        logger=tb_logger,
        max_epochs = data_args.max_epoch,
        precision='bf16'
    )
    trainer.fit(model_module, data_module) 



if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()































