# KoLEG: On-the-Fly Korean Legal Knowledge Editing with Continuous Retrieval

**Official Implementation** of *“KoLEG: On-the-Fly Korean Legal Knowledge Editing with Continuous Retrieval”* (Findings of EMNLP 2025)


## Requirements

To set up the environment, run:

```bash
git clone https://github.com/ekgus9/KoLEG.git
cd KoLEG

conda create -n KoLEG python=3.10
conda activate KoLEG

# Install PyTorch with CUDA 12.1 support
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt
```

## Korean Legislative Amendment Dataset Structure

The Korean Legislative Amendment Dataset is designed to evaluate a model’s capability in legal knowledge editing and temporal reasoning within dynamically updated law texts.
This dataset captures both *single updates* (latest amendments) and *sequential revisions* (multiple accumulated changes with temporal cues).

```
./dataset
├──  train_data/
│   ├── train_rel.json             # Queries with relevant knowledge and the corresponding edited answers
│   ├── train_por.json             # Edited answers + relevant knowledge to reconstruct the original queries
│   ├── train_loc.json             # Non-editing queries (no knowledge injected), for stability
│   └── train_loc+.json            # Non-editing queries with irrelevant knowledge injected, for robustness
├── test_data/
│   ├── test_book.json             # Legal knowledge base
│   └── test.json                  # Evaluation dataset
└── sequential_data/
    ├── sequential_edit_1.json     # 1st-stage edited legal provisions
    ├── sequential_edit_2.json     # 2nd-stage edited legal provisions
    ├── sequential_edit_3.json     # 3rd-stage edited legal provisions
    ├── sequential_edit_4.json     # 4th-stage edited legal provisions 
    ├── sequential_test_1.json     # 1st-stage test set
    ├── sequential_test_2.json     # 2nd-stage test set
    ├── sequential_test_3.json     # 3rd-stage test set
    └── sequential_test_4.json     # 4th-stage test set
```

## Editing-Aware Learning Strategy

### 1. Traning LawEdit Retriever

This phase fine-tunes the `BAAI/bge-m3` embedding model to create the **LawEdit Retriever**. The training uses an Editing-Aware Learning dataset within the FlagEmbedding framework. Before training, you must first download the dataset from HuggingFace and convert it to json:

```bash
mkdir -p dataset/train_data && python -c "from datasets import load_dataset; ds = load_dataset('yongchanskii/editing_aware_retrieval_dataset'); ds['train'].to_json('dataset/train_data/editing_aware_retrieval_dataset.jsonl')"
```

Then, you can start training:

```bash
cd train && \
torchrun --nproc_per_node 2 \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    --model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data ../dataset/train_data/editing_aware_retrieval_dataset.jsonl \
    --cache_path ./cache/data \
    --train_group_size 128 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
    --output_dir ./results/test_encoder_only_m3_bge-m3_sd_deepspeed \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 True \
    --num_train_epochs 0.067 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing False \
    --logging_steps 1 \
    --save_steps 5 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    --unified_finetuning True \
    --use_self_distill True \
    --fix_encoder False \
    --self_distill_start_step 0 \
&& cd ../
```

#### Arguments

* `--model_name_or_path`: Base pretrained model to be fine-tuned.
  Here, the `BAAI/bge-m3` model is used as the backbone.
* `--train_data`: Path to the training dataset.
* `--output_dir`: Directory to save checkpoints and logs.
* `--num_train_epochs`: Total number of fine-tuning epochs.
* `--learning_rate`, `--warmup_ratio`: Standard optimization hyperparameters.
* `--deepspeed`: Path to the DeepSpeed configuration file for distributed training.


### 2. Training (Alignment Phrase)

This phase fine-tunes the **`meta-llama/Llama-3.1-8B-Instruct`** model with an editing-aware alignment strategy. Training is conducted using LoRA and DeepSpeed for efficient multi-GPU optimization.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed train/fastchat/train/train_lora_koleg.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ./data/train_rel.json \
    --fp16 True \
    --output_dir checkpoint_koleg_llama \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 2 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --q_lora False \
    --deepspeed train/playground/deepspeed_config_s2.json
```

#### Arguments

* `--model_name_or_path`: Base pretrained model to be fine-tuned.
  Here, the Llama-3.1-8B-Instruct model is used as the backbone.
* `--data_path`: Path to the training dataset.
* `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA-specific hyperparameters controlling the rank, scaling, and dropout rate of low-rank adapters.
* `--output_dir`: Directory to save checkpoints and logs.
* `--num_train_epochs`: Total number of fine-tuning epochs.
* `--gradient_accumulation_steps`: Accumulates gradients to simulate larger batch sizes for memory efficiency.
* `--learning_rate`, `--weight_decay`, `--warmup_ratio`, `--lr_scheduler_type`: Standard optimization hyperparameters.
* `--deepspeed`: Path to the DeepSpeed configuration file for distributed training.


### 2. Evaluation (Inference Phrase)

The evaluation is based on [**EasyEdit**](https://github.com/zjunlp/EasyEdit).

```bash
python test/run.py \
    --hparams_fname=test/hparams/llama \
    --ds_name=data/test.json \
    --batch_size=1 \
    --retriever=[retriever_path]
```

#### Arguments

* `--hparams_fname`: Path to the hyperparameter configuration file.
* `--ds_name`: Evaluation dataset containing test samples.
* `--batch_size`: Number of samples processed per batch.
* `--retriever`: Retriever model or checkpoint path used to locate related knowledge.


## Citation

If you use **KoLEG** or the dataset in your research, please cite our paper:

```
@inproceedings{seo2025koleg,
  title     = {KoLEG: On-the-Fly Korean Legal Knowledge Editing with Continuous Retrieval},
  author    = {Jaehyung Seo and Dahyun Jung and Jaewook Lee and Yongchan Chun and Dongjun Kim 
               and Hwijung Ryu and Donghoon Shin and Heuiseok Lim},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  year      = {2025}
}
```
