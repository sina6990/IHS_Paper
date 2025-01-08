import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import bitsandbytes as bnb
import argparse
import logging
import os
from datetime import datetime
import argparse
import logging
import json
from utils import parse_args
from model_llama import *
from dataPreprocessing import data_preprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    args = parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log all arguments
    logging.info("Training arguments:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    train_dataset, test_dataset = data_preprocessing(args.file_name, args.train_ratio)
    trainer, model, tokenizer = train_model(train_dataset, args)

    logging.info("Getting predictions and computing metrics...")
    predictions, texts, true_labels = predict(model, tokenizer, test_dataset)
    metrics = compute_metrics(predictions, true_labels)
    
    # Save model and metrics
    model_save_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save test metrics
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    logging.info("Training completed successfully")