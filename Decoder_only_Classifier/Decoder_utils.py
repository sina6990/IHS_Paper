import argparse
import logging
from datetime import datetime
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Training an LLM with LoRA for classification')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels')
    parser.add_argument('--file_name', type=str, required=True, help='Name of the input CSV file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='Base model name')
    parser.add_argument('--output_dir', type=str, default='./results/llama', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logging')
    return parser.parse_args()

def setup_logging(args):
    # Create logs directory if it doesn't exist
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create unique run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split('/')[-1]
    run_id = f"{model_short_name}_{timestamp}"
    
    # Set up logging configuration
    log_filename = os.path.join(args.log_dir, f"{run_id}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Log all arguments
    logging.info("Training Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
        
    return run_id

def file_path(file_name):
    if file_name == 'mentalmanip_con':
        return '../datasets/MentalManip/mentalmanip_con.csv'
    elif file_name == 'mentalmanip_maj':
        return '../datasets/MentalManip/mentalmanip_maj.csv'