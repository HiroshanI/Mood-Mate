import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    DistilBertForSequenceClassification, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    hamming_loss
)
from tqdm.notebook import tqdm_notebook as tqdm
from statistics import mean

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(  
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EmotionClassifier():
    def __init__(self, model_name='distilbert-base-multilingual-cased', n_labels=6, tok_path=None):
        
        # Check device availability
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        # Init Model 
        if model_name.startswith("distil"):
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=n_labels
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=n_labels
            )
            
        # Init Tokenizer
        if tok_path == None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            
        # Move model to device
        self.model.to(device)
        
    def create_dataset(self, df, max_len):
        return EmotionDataset(
            texts=df['text'].tolist(),
            labels=df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_len=max_len
        )
    
    def train(self, train, test, 
              max_len=128, n_epochs=5, learning_rate=5e-5,
              steps=50, save=False, save_dir="./saved/"):
        
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='macro'
            )
            acc = accuracy_score(labels, preds)
            ham_score = hamming_loss(labels, preds)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'hamming_loss':ham_score
            } 
        
        train_dataset = self.create_dataset(train, max_len)
        test_dataset = self.create_dataset(test, max_len)
        
        training_args = TrainingArguments(
            output_dir='./logs',
            learning_rate=learning_rate,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_steps=-1,
            save_steps=-1,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        ) 

        trainer.train() 

        if save:
            tok_name = f'tokenizer_{n_epochs}_{learning_rate}_{max_len}_{train.shape[0]}'
            mod_name = f'model_{n_epochs}_{learning_rate}_{max_len}_{train.shape[0]}'

            trainer.save_model(
                save_dir + mod_name
            )
            self.tokenizer.save_pretrained(
                save_dir + tok_name
            )
            results = trainer.evaluate()
            return results, tok_name, mod_name
        
        results = trainer.evaluate()
        
        return results
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).item()
        confidences = torch.softmax(outputs.logits, dim=1)
        # print(predictions, confidences, outputs.logits)
        return predictions, confidences
        
def evaluate_dataset(train, test, n_exp=3, n_epochs=5, model_name='distilbert-base-multilingual-cased', learning_rate=5e-5, max_len=128, steps=10, save=False, save_dir="./saved/"):

    results = []

    for _ in range(n_exp):
        model = EmotionClassifier(model_name=model_name)
        results.append(model.train(
            train, 
            test, 
            max_len, 
            n_epochs, 
            learning_rate, 
            steps, 
            save, 
            save_dir
        ))
        
    f1_mean = mean([r['eval_f1'] for r in results])
    acc_mean = mean([r['eval_accuracy'] for r in results])
    recall_mean = mean([r['eval_recall'] for r in results])
    precision_mean = mean([r['eval_precision'] for r in results])
    hamming_loss_mean = mean([r['eval_hamming_loss'] for r in results])
    
    
    return f1_mean, acc_mean, recall_mean, precision_mean, hamming_loss_mean

def make_prediction(input, model='original', lang='en'):
    print(f"Selected model: {model} | {lang}")
    model_dir = "prediction/text/models/"
    if lang == 'en':
        if model == "augmented":
            model_path = model_dir + 'mod_aug'
            tok_path = model_dir+'tok_aug'
        elif model == 'original':
            model_path = model_dir + 'mod_en_6000'
            tok_path = model_dir+'tok_en_6000'
    elif lang == 'si':
        if model == "augmented":
            model_path = model_dir+'mod_aug_si'
            tok_path = model_dir+'tok_aug_si'
        elif model == 'original':
            model_path = model_dir+'mod_og_si'
            tok_path = model_dir+'tok_og_si'
    elif lang == 'tm':
        if model == "augmented":
            model_path = model_dir+'mod_aug_tm'
            tok_path = model_dir+'tok_aug_tm'
        elif model == 'original':
            model_path = model_dir+'mod_og_tm'
            tok_path = model_dir+'tok_og_tm'
        

    model = EmotionClassifier(model_path, tok_path=tok_path)
    return model.predict(input)
