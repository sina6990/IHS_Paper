import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import bitsandbytes as bnb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import logging

def prepare_model_and_tokenizer(args):
    logging.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, config)
    logging.info("Model prepared with LoRA configuration")
    return model, tokenizer

def get_max_length(texts, tokenizer):
    # Tokenize all texts without padding or truncation to get their lengths
    lengths = [len(tokenizer(text, truncation=False)['input_ids']) for text in texts]
    return max(lengths)

def tokenize_function(examples, tokenizer):
    prompt_template = "I will provide you with a dialogue. Please determine whether or not it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n {}\nAnswer:"
    texts = [prompt_template.format(text) for text in examples['Dialogue']]

    batch_max_length = 1024

    model_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=batch_max_length,
        return_tensors=None
    )

    # Create labels
    labels = list()
    for idx, label in enumerate(examples['label']):
        # Copy input_ids
        label_sequence = model_inputs['input_ids'][idx].copy()
        # Add the true/false token at the end
        response = " true" if label == 1 else " false"
        response_ids = tokenizer.encode(response)[1:]  # Skip the first token which would be the start token
        label_sequence.extend(response_ids)
        # Pad if necessary
        if len(label_sequence) < batch_max_length:
            label_sequence.extend([tokenizer.pad_token_id] * (batch_max_length - len(label_sequence)))
        # Trim if too long
        label_sequence = label_sequence[:batch_max_length]
        labels.append(label_sequence)
    
    model_inputs['labels'] = labels

    return model_inputs

def train_model(train_dataset, args):
    model, tokenizer = prepare_model_and_tokenizer(args)
    
    processed_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=train_dataset.column_names,
        batched=True,
        batch_size=args.batch_size
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=data_collator
    )
    
    logging.info("Starting training")
    #trainer.train()

    return trainer, model, tokenizer

def evaluate_model(text, model, tokenizer, device="cuda"):
    prompt = f"I will provide you with a dialogue. Please determine whether or not it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n{text}\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            max_new_tokens=5,
            num_return_sequences=1,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    print(outputs)
    # Get just the generated answer part
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    # Extract the last word and clean it
    prediction = response[len(prompt):].strip().lower()
    print(prediction)
    return prediction

# def evaluate_model(text, model, tokenizer, device="cuda"):
#    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
#    messages = [
#        {"role": "system", "content": "You are a helpful assistant that answers with only 'Yes' or 'No'."},
#        {"role": "user", "content": f"I will provide you with a dialogue. Please determine whether or not it contains elements of mental manipulation. Just answer with 'Yes' or 'No', and don't add anything else.\n{text}"}
#    ]
   
#    model.eval()
#    with torch.no_grad():
#        outputs = pipe(messages, max_new_tokens=5)
#        prediction = outputs[0]["generated_text"][-1]
#        print(prediction)
#        #prediction = prediction.split()[0] if prediction.split() else ''
       
#     #    if prediction.lower() not in ['yes', 'no']:
#     #        return None
           
#        return prediction

# def predict(model, tokenizer, test_dataset):
#    predictions, texts, true_labels = [], [], []
#    invalid_count = 0
   
#    for item in test_dataset:
#        pred = evaluate_model(item['Dialogue'], model, tokenizer)
#        if pred is None:
#            invalid_count += 1
#            continue
           
#        predictions.append(1 if pred == 'yes' else 0)
#        texts.append(item['Dialogue'])
#        true_labels.append(int(item['label']))
   
#    print(f"Invalid predictions: {invalid_count}")
#    return predictions, texts, true_labels

def predict(model, tokenizer, test_dataset):
    logging.info("Starting model evaluation")
    predictions = list()
    texts = list()
    true_labels = list()
    invalid = list()
    invalid_count = 0
    
    for item in test_dataset:
        pred = evaluate_model(item['Dialogue'], model, tokenizer)
        # Check if prediction is valid
        if pred.lower() not in ['yes', 'no']:
            invalid.append(pred)
            invalid_count += 1
            continue

        pred_num = 1 if pred.lower == 'yes' else 0 
        predictions.append(pred_num)
        texts.append(item['Dialogue'])
        true_labels.append(int(item['label']))
    
    print(f"Invalid predictions: {invalid}")
    print(f"Number of Invalid predictions: {invalid_count}")
    return predictions, texts, true_labels

def compute_metrics(predictions, true_labels):
    predictions = np.array(predictions, dtype=int)
    true_labels = np.array(true_labels, dtype=int)
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    metrics = {
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return metrics


