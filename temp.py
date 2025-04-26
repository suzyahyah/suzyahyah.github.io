#!/usr/bin/python3
# Author: Suzanna Sia

import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Sampler
from torch.utils.data import BatchSampler
from tqdm import tqdm
import os
import logging
import torch
from transformers import GPT2Config, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from transformers import Trainer
from datasets import load_dataset, load_from_disk
# the goal is to write our own dataloader for huggingface trainer
# https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
# line 1017 shows that you can provide your own dataloader for dynamic
# batching.

#https://claude.ai/chat/ee7e03ae-eddd-4bf3-84af-767a58113ebf


# Instead of getting the tokenized sequence length exactly, its sufficient to
# just calculate the length as an approximation
#  train_ds_tok_lens = [len(train_ds['train'][i]['text'].split()) for i in
# tqdm(range(len(train_ds['train'])))] is still super inefficient, we need to
# find better ways to do it
# sampler = DynamicBatchSampler(train_ds_tok_lens)


#https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py

class DynamicBatchSampler(BatchSampler):
    def __init__(self, seq_lengths, max_seq_len=1024, med_batchsize=64):
        self.med_batchsize = med_batchsize
        self.seq_lengths = seq_lengths
        self.bins = {"small": [], "med": [], "large": [], 'xl':[]}
        self.max_seq_len = max_seq_len
        self.construct_bins()
        self.current_index = {"small": 0, "med": 0, "large": 0, 'xl':0}

    def construct_bins(self):
        print("Constructing bins", len(self.seq_lengths))
        for i, val in tqdm(enumerate(self.seq_lengths)):
            if val < (self.max_seq_len / 4):
                self.bins['small'].append(i)

            elif val < (self.max_seq_len / 2):
                self.bins['med'].append(i)
            elif val < ((self.max_seq_len / 4) * 3):
                self.bins['large'].append(i)
            else:
                self.bins['xl'].append(i)


    def __iter__(self):
        #while True:
          bin_type, size = random.choices([('small', self.med_batchsize*2),
                                          ('med', self.med_batchsize),
                                          ('large', (self.med_batchsize/2)*3),
                                          ('xl', self.med_batchsize//2)],
                                        weights=[len(self.bins['small']),
                                                  len(self.bins['med']),
                                                  len(self.bins['large']),
                                                  len(self.bins['xl'])],
                                        k=1)[0]

          print("SAMPLER ITER CALLED", bin_type, size)
          cur_index = self.current_index[bin_type]

          self.current_index[bin_type] += size
          if self.current_index[bin_type] > len(self.bins[bin_type]):
              self.current_index[bin_type] = 0
              random.shuffle(self.bins[bin_type])


          yield self.bins[bin_type][cur_index:cur_index + size]

    def __len__(self):
        # Return an estimate of the number of batches
        return sum([
            len(self.bins['small']) // (self.med_batchsize*2 + 1),
            len(self.bins['med']) // (self.med_batchsize + 1),
            len(self.bins['large']) // (self.med_batchsize//2 * 3 + 1),
            len(self.bins['xl']) // (self.med_batchsize//2 + 1)
        ]) // 4
        #return len(self.seq_lengths)


from transformers.trainer import Trainer
from tqdm import tqdm
from torch.utils.data import DataLoader
#from custom_sampler import DynamicBatchSampler
import numpy as np
from  transformers.trainer_utils import seed_worker
import datasets
# import DataLoader
 
class MyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def torch_call(self, examples):
        #import pdb; pdb.set_trace()
        #examples['input_ids'].pad
        #examples['attention_mask'].pad
        examples = self.tokenizer.pad(examples, padding=True, pad_to_multiple_of=None, return_tensors='pt')
        del examples['lengths']

        return super().torch_call(examples)


class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_train_sampler(self):
        # process lengths on dataset
        dynamic_sampler = DynamicBatchSampler((self.train_dataset['lengths']), max_seq_len=1024, med_batchsize=64)
        return dynamic_sampler

    def get_train_dataloader(self):
        collate_fn = MyDataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=False)
        data_collator = self.data_collator
        # batch_size argument empty, default to 1
        dataloader_params = {"collate_fn": collate_fn,
                            "num_workers": self.args.dataloader_num_workers,
                            "pin_memory": self.args.dataloader_pin_memory,
                            "shuffle": False,
                            "sampler": None,
                            "drop_last": False,
                            "persistent_workers": self.args.dataloader_persistent_workers}
        dataloader_params['batch_sampler'] = self._get_train_sampler()
        dataloader = DataLoader(self.train_dataset, **dataloader_params)
        print("TEST TEMP DATALOADER")
        #for batch in dataloader:
        #    print(batch)
        return self.accelerator.prepare(DataLoader(self.train_dataset, 
                                                    **dataloader_params))


import os
import logging
import torch
from transformers import GPT2Config, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments
from transformers import Trainer
from datasets import load_dataset, load_from_disk
#from custom_trainer import MyTrainer
#import gpt2configd

os.environ['HF_DATASETS_OFFLINE'] = '0'
logging.getLogger("datasets").setLevel(logging.DEBUG)

NGPUS = 1
block_size = 1024
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
LOGGING_STEPS = 500
DATASET = "openwebtext"
SAVE_STRATEGY = "steps"
eval_steps = 500
save_steps = 500
num_epochs = 2
nproc = 60
max_steps = 275000 * num_epochs # learning rate scheduler
size = "small"
learning_rate = 5e-4
weight_decay = 0.1
lr_scheduler_type = "linear"
#warmup_steps = 5000
OUTPUT_DIR = f"./training_dir/data-{DATASET}/model-gpt2-125M_lr-{learning_rate}_maxsteps-{max_steps}_lrscheduler-{lr_scheduler_type}_batchsize-dynamic_weightdecay-{weight_decay}_blocksize-{block_size}"
  
def get_training_args():
    training_args = TrainingArguments(output_dir=OUTPUT_DIR,
            max_steps=max_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            weight_decay=weight_decay,
            bf16_full_eval=True,
            bf16=True,
            num_train_epochs=num_epochs,
            #per_device_train_batch_size=BATCH_SIZE,
            evaluation_strategy="steps",
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            ddp_find_unused_parameters=False,
            eval_steps=eval_steps,
           # report_to='wandb',
            logging_steps=LOGGING_STEPS,
            save_strategy=SAVE_STRATEGY,
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True)
    return training_args


    #
def load_ds2(tokenizer):
    block_size = 8
    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], truncation=True,
                max_length=block_size) #, padding='longest',
                #return_overflowing_tokens=True)
        lengths = [len(ids) for ids in tokenized['input_ids']]
        return {
            "input_ids": tokenized['input_ids'],
            "attention_mask": tokenized['attention_mask'],
            "lengths": lengths
        }


    dev_ds = load_dataset("wikitext", "wikitext-103-raw-v1",
            split='validation')
    dev_ds = dev_ds.filter(lambda example: example['text'] != '')
    #try:
    #    train_ds = load_from_disk(f'tokenized_openwebtext_lengths')
    #except:
    #    train_ds = load_dataset("openwebtext", trust_remote_code=True)
    #    train_ds = train_ds.map(tokenize_function, batched=True,
    #        remove_columns=train_ds['train'].column_names, num_proc=nproc)
    #    train_ds.save_to_disk(f"tokenized_openwebtext_lengths")
    dev_ds = dev_ds.map(tokenize_function, batched=True,
        remove_columns=dev_ds.column_names, num_proc=60)
    return dev_ds, dev_ds

model = AutoModelForCausalLM.from_config(GPT2Config(), torch_dtype=torch.bfloat16)
                #                            torch_dtype=torch.bfloat16,
                                            #attn_implementation="flash_attention_2")
device = 'cuda'

    # train tokenizer from scratch
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
train_ds, dev_ds = load_ds2(tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
            mlm=False) #

#def custom_causal_lm_collate_fn_dict(batch):
#    input_ids = torch.stack([item["input_ids"] for item in batch])
#    labels = input_ids.clone()
#    return {"input_ids": input_ids, "labels": labels}

training_args = get_training_args()
trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator)

trainer.train()
