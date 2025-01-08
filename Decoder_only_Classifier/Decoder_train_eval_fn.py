import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import json
import logging
import os

class CustomCausalModel(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  
            torch_dtype=torch.float16,  
            use_cache=False 
        )

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()

        # Freeze the pre-trained parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            inference_mode=False
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Replace the LM head with a classification head
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # Take last token hidden state
        logits = self.classifier(hidden_states)
        return logits

def train_model(model, train_loader, optimizer, criterion, device, args, run_id):
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'learning_rate': [],
        'epoch': []
    }

    model.to(device)
    scaler = torch.amp.GradScaler(device.type)
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        # Create progress bar for current epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.num_epochs}', total=len(train_loader))
        
        for i, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast(device.type):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                train_loss += loss.item() * args.gradient_accumulation_steps
            
            if i % 10 == 0:
                current_loss = train_loss / (i + 1)
                current_acc = 100 * correct / total
                logging.info(f'Epoch {epoch + 1}, Batch {i}/{len(train_loader)}: '
                           f'Loss = {current_loss:.4f}, Acc = {current_acc:.2f}%')
            
            # Update progress bar with current metrics
            current_loss = train_loss / (i + 1)
            current_acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Close progress bar for current epoch
        pbar.close()
        
        # Print epoch summary
        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100 * correct / total

        # Save metrics for this epoch
        metrics_history['train_loss'].append(epoch_loss)
        metrics_history['train_acc'].append(epoch_acc)
        metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        metrics_history['epoch'].append(epoch + 1)
        
        logging.info(f'Epoch {epoch + 1} Complete - '
                    f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Save metrics after each epoch
        metrics_file = os.path.join(args.log_dir, f"{run_id}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
    
    return metrics_history

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    eval_metrics = {
        'batch_losses': [],
        'batch_accuracies': []
    }

    progress_bar = tqdm(test_loader, desc='Evaluating', leave=True)
    
    with torch.amp.autocast(device.type), torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Ensure classifier weights match model dtype
            if hasattr(model, 'classifier'):
                model.classifier.to(dtype=model.model.dtype)
            logits = model(input_ids, attention_mask)
            
            # Get predictions first
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Calculate loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Record batch metrics
            batch_acc = 100 * (predictions == labels).sum().item() / labels.size(0)
            eval_metrics['batch_losses'].append(loss.item())
            eval_metrics['batch_accuracies'].append(batch_acc)
            
            progress_bar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'accuracy': f'{100*correct/total:.2f}%'
            })
    
    progress_bar.close()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, eval_metrics