import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from DP_Decoder import *
from Decoder_train_eval_fn import *
from Decoder_utils import *

if __name__ == "__main__":
    args = parse_args()
    run_id = setup_logging(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set memory growth
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logging.info("Set pad token to eos token")
    
    model = CustomCausalModel(args.model_name, args.num_labels)
    model.model.config.pad_token_id = tokenizer.pad_token_id
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader, test_loader = custom_dataloader(
        file_path(args.file_name),
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    logging.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Training
    logging.info("Starting training...")
    metrics_history = train_model(model, train_loader, optimizer, criterion, device, args, run_id)

    # Final evaluation
    logging.info("Performing final evaluation...")
    criterion = criterion.to(dtype=model.model.dtype)
    final_loss, final_accuracy, eval_metrics = evaluate_model(model, test_loader, criterion, device)
    logging.info(f"Final Test Loss: {final_loss:.4f}")
    logging.info(f"Final Test Accuracy: {final_accuracy:.2f}%")

        # Save final results
    final_results = {
        'args': vars(args),
        'metrics_history': metrics_history,
        'final_test_loss': final_loss,
        'final_test_accuracy': final_accuracy,
        'eval_metrics': eval_metrics,
        'model_info': {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    }
    
    # Save results
    results_file = os.path.join(args.log_dir, f"{run_id}_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    logging.info(f"Saved final results to {results_file}")
    
    # Save model
    model_save_path = os.path.join(args.output_dir, f"{run_id}_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Saved model to {model_save_path}")
    
