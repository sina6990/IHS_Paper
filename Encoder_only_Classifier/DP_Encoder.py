from datasets import load_dataset
from torch.utils.data import DataLoader

def tokenize_function(dataset, tokenizer):
    return tokenizer(dataset['Dialogue'], truncation=True, max_length=tokenizer.model_max_length)

def data_loader(dataset, batch_size, data_collator):
    return DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator)

def custom_dataloader(path, batch_size, tokenizer, data_collator):
    dataset = load_dataset("csv", data_files=path)

    tokenized_datasets = dataset.map(lambda dataset: tokenize_function(dataset, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["ID", "Dialogue", "Technique", "Vulnerability"])
    tokenized_datasets = tokenized_datasets.rename_column("Manipulative", "labels")
    tokenized_datasets.set_format("torch")

    split_dataset = tokenized_datasets['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    train_loader = data_loader(train_dataset, batch_size, data_collator)
    test_loader = data_loader(test_dataset, batch_size, data_collator)

    return train_loader, test_loader

