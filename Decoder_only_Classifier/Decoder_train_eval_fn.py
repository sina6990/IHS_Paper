import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm

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

# def train_model(model, train_loader, optimizer, criterion, device, num_epochs=3):
#     model.to(device)
    
#     # Create progress bar for epochs
#     epoch_pbar = tqdm(range(num_epochs), desc='Epochs', position=0)
    
#     for epoch in epoch_pbar:
#         model.train()
#         train_loss = 0
#         correct = 0
#         total = 0
        
#         # Create progress bar for batches
#         batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', 
#                          leave=False, position=1)
        
#         for batch in batch_pbar:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
            
#             optimizer.zero_grad()
#             logits = model(input_ids, attention_mask)
            
#             # Get predictions
#             predictions = torch.argmax(logits, dim=1)
#             correct += (predictions == labels).sum().item()
#             total += labels.size(0)
            
#             # Calculate loss and backpropagate
#             loss = criterion(logits, labels)
#             loss.backward()
#             optimizer.step()
            
#             # Update batch progress bar
#             train_loss += loss.item()
#             current_loss = train_loss / (batch_pbar.n + 1)
#             current_acc = 100 * correct / total
#             batch_pbar.set_postfix({
#                 'loss': f'{current_loss:.4f}',
#                 'acc': f'{current_acc:.2f}%'
#             })
        
#         # Calculate final metrics for the epoch
#         train_loss = train_loss / len(train_loader)
#         train_acc = 100 * correct / total
        
#         # Update epoch progress bar
#         epoch_pbar.set_postfix({
#             'train_loss': f'{train_loss:.4f}',
#             'train_acc': f'{train_acc:.2f}%'
#         })
        
#         batch_pbar.close()
    
#     epoch_pbar.close()

def train_model(model, train_loader, optimizer, criterion, device, num_epochs=3, gradient_accumulation_steps=4):
    model.to(device)
    
    scaler = torch.amp.GradScaler('cuda')
    
    epoch_pbar = tqdm(range(num_epochs), desc='Epochs', position=0)
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer.zero_grad()
        
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', 
                         leave=False, position=1)
        
        for i, batch in enumerate(batch_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Updated autocast syntax
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                train_loss += loss.item() * gradient_accumulation_steps
            
            current_loss = train_loss / (batch_pbar.n + 1)
            current_acc = 100 * correct / total
            batch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%'
        })
        
        batch_pbar.close()
    
    epoch_pbar.close()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.amp.autocast('cuda'), torch.no_grad():
        for batch in test_loader:
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
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy