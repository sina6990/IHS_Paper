from Implicit_Manipulation.IHS_Paper.Encoder_only_Classifier.DP_Encoder import *
from Implicit_Manipulation.IHS_Paper.Encoder_only_Classifier.Encoder_train_eval_fn import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, AdamW
from transformers import RobertaTokenizer, RobertaModel

if __name__ == '__main__':
    path = '../datasets/MentalManip/mentalmanip_con.csv'
    model_name = 'distilbert-base-uncased'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer)
    optimizer = AdamW(model.parameters(), lr=5e-3)

    train, test = custom_dataloader(path, 6, tokenizer, data_collator)

    train_model(model, optimizer, train, 5, device)
    eval_model(model, test, device)