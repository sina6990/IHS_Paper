import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from DP_Decoder import *
from Decoder_train_eval_fn import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    num_labels = 2
    path = '../datasets/MentalManip/mentalmanip_con.csv'

    # Set memory growth
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = CustomCausalModel(model_name, num_labels)
    model.model.config.pad_token_id = tokenizer.pad_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader, test_loader = custom_dataloader(
        path,
        batch_size=2,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Starting training...")
    train_model(model, train_loader, optimizer, criterion, device, 1, 4)

    # Final evaluation
    print("\nPerforming final evaluation...")
    criterion = criterion.to(dtype=model.model.dtype)
    final_loss, final_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    
