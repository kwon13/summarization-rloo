import multiprocessing
import shutil
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    AutoModel, AutoConfig
)
import torch
import torch.nn as nn
from trl import ModelConfig
from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE


"""
# https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L877
accelerate launch --config_file deepspeed_zero3.yaml \
    rloo.py \
    --output_dir models/rloo_tldr_t=0.1_ppo=1 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --total_episodes 450 \
    --model_name_or_path fiveflow/exa-base \
    --sft_model_path fiveflow/exa-base \
    --reward_model_path exaone_reward_modelv2 \
    --local_rollout_forward_batch_size 1 \
    --non_eos_penalty True \
    --response_length 512 \
    --stop_token eos \
    --temperature 0.1 \
    --rloo_k 2
"""


if __name__ == "__main__":
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    # shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    
    print("# Load Reward Model")
    reward_model = AutoModelForSequenceClassification.from_pretrained(config.reward_model_path, num_labels=1)
    print("# Load Ref & Policy Model")
    ref_policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=torch.float16)
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, torch_dtype=torch.float16)
    ################
    # Dataset
    ################
    raw_datasets = load_dataset('fiveflow/summary-rlhf-v4')
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1000))
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["chosen"][:-1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not config.sanity_check,
        )

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    # filtering
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 3000 and x["lengths"] > 1500)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 3000 and x["lengths"] > 1500)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################
    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.generate_completions()
