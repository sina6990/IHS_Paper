import torch
import numpy as np
from transformers import get_scheduler
from tqdm.auto import tqdm
from evaluate import load, combine

def train_model(model, optimizer, train_dataloader, num_epochs, device):
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer= optimizer,
        num_warmup_steps= 0,
        num_training_steps= num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

def eval_model(model, test_dataloader, device):
    metric = load('accuracy')
    #metric = combine(["accuracy", "recall", "precision", "f1"])

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # Add predictions and references to the metric
        metric.add_batch(predictions=predictions.cpu().numpy(), references=batch["labels"].cpu().numpy())

    result = metric.compute()
    print(result)
    