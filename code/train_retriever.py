#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from mytools.tool_utils import FileUtils, TorchUtils
from dataloader import create_dataloader
from model import create_model, get_model_class
from collections import defaultdict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from common_utils import override


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--override_args",
        type=str,
        default=None,
        help="use the command line to override args",
    )

    parser.add_argument(
        "--project_config",
        default="./config/retriver_base.yaml",
        help=(
            "path to the config file (.yaml) for training the phrase retriever"
        ),
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args = vars(args)
    retrieval_args = FileUtils.load_from_disk(args['project_config'])
    for k, v in retrieval_args.items():
        args[k] = v
    args = override(args)
    FileUtils.save_file(args, args['output_dir'] + "/project_config.yaml", "yaml")
    args = defaultdict(lambda: None, args)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    find_unused_parameters = args.get('find_unused_parameters', False)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(gradient_accumulation_steps=args['gradient_accumulation_steps'], kwargs_handlers=[ddp_kwargs])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("Args: \n: {}".format(args))
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args['seed'] is not None:
        set_seed(args['seed'])

    # Handle the repository creation
    if accelerator.is_main_process:
        FileUtils.check_dirs(args['output_dir'])
    accelerator.wait_for_everyone()
    tokenizer = AutoTokenizer.from_pretrained(args['model_config_dir'])
    train_dataloader = create_dataloader(args['ds_name'], args)
    if 'model_pretrained_dir' in args:
        unwrapped_model = get_model_class(args['model_name']).from_pretrained(args)
    else:
        unwrapped_model = create_model(args['model_name'], args)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in unwrapped_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in unwrapped_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'])

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args['gradient_accumulation_steps'])
    if args['max_train_steps'] is None:
        args['max_train_steps'] = args['num_train_epochs'] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=args['num_warmup_steps'] * args['gradient_accumulation_steps'],
        num_training_steps=args['max_train_steps'] * args['gradient_accumulation_steps'],
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unwrapped_model, optimizer, train_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args['checkpointing_steps']
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit() :
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args['batch_size'] * accelerator.num_processes * args['gradient_accumulation_steps']

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {args['batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {args['max_train_steps']}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args['max_train_steps']), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, args['num_train_epochs']):
        model.train()
        total_loss = []
        step_until_now = 0
        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            step_until_now += 1
            with accelerator.accumulate(model):
                loss = model(batch)
                # We keep track of the loss at each epoch
                total_loss.append(loss.detach().float())
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if step_until_now % args['logging_steps'] == 0 and accelerator.is_local_main_process:
                logging.info("Averaged loss at {}-{} (epoch-step) is {:.3f}".format(epoch, step_until_now, sum(total_loss) / len(total_loss)))
                total_loss = []
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args['output_dir'] is not None:
                        output_dir = os.path.join(args['output_dir'], output_dir)
                    accelerator.save_state(output_dir)
                    if (accelerator.is_local_main_process):
                        FileUtils.save_to_disk(dict(args), output_dir + "/retriver_config.yaml", 'yaml')

            if completed_steps >= args['max_train_steps']:
                break

        if args['checkpointing_steps'] == "epoch":
            output_dir = f"epoch_{epoch}"
            if args['output_dir'] is not None:
                output_dir = os.path.join(args['output_dir'], output_dir)
            accelerator.save_state(output_dir)
            if (accelerator.is_local_main_process):
                FileUtils.save_to_disk(dict(args), output_dir + "/retriver_config.yaml", 'yaml')

    if accelerator.is_main_process:
        logging.info("Finished training")


if __name__ == "__main__":
    main()
