import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training an LLM with LoRA for classification')
    parser.add_argument('--file_name', type=str, required=True, help='Name of the input CSV file')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-3B-Instruct", help='Base model name')
    parser.add_argument('--output_dir', type=str, default='./results/llama', help='Output directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA attention dimension')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    return parser.parse_args()

