import shap
from prediction.text.evaluate import EmotionClassifier, EmotionDataset
from transformers import (
    DistilBertForSequenceClassification, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
import pickle
import torch
import pandas as pd
import shutil

def train_and_save_model(train, test, epochs, learning_rate):
    # Train and save model
    model = EmotionClassifier()
    _, tok, mod = model.train(train, test, n_epochs=epochs, save=True, steps=100, learning_rate=learning_rate)
    return mod, tok

def load_model(model_path, tok_path):
    # Load saved model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    return model, tokenizer

def get_shap_values(train, model, tokenizer):
    preds = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        batch_size=16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # Extract shap values
    dataset = EmotionDataset(train.text, train.label, tokenizer, 128)
    masker = shap.maskers.Text(tokenizer=r"\s")
    explainer = shap.Explainer(preds, masker)
    shap_values = explainer(dataset.texts)

    return shap_values

def extract_importance(train, model_path, tok_path, load=True, test=None, epochs=10):
    if load:
        model, tokenizer = load_model(model_path, tok_path)
        return get_shap_values(train, model, tokenizer)
    else:
        mod, tok = train_and_save_model(train, test, epochs)
        saved_dir = 'saved/'
        model, tokenizer = load_model(saved_dir+mod, saved_dir+tok)
        return get_shap_values(train, model, tokenizer)
    
def load_shaps_per_sent(shap_path):
    with open(shap_path, 'rb') as file:
        return pickle.load(file)

def create_shaps_for_samples(train_files, test_file):
    print('Creating Shapley values for train samples:')
    print(*train_files, sep='\n')
    print()
    # Read data
    train_ge = {}
    for filename in train_files:
        # train_ge[ partition_size ] = partition_data
        train_ge[filename.split('.')[0].split('_')[-1]] = pd.read_csv(filename)
    test = pd.read_csv(test_file)

    # Train model and save
    model_path, tokenizer_path = {}, {}
    filenames = []
    for train_size, train_data in train_ge.items():

        # Train and save model
        print("Sample size:",train_data.shape)
        model_path, tokenizer_path = train_and_save_model(
            train=train_data, 
            test=test, 
            epochs=5
        )

        print(model_path, tokenizer_path)

        # Extract shap values for train
        shap_values_per_sent = extract_importance(
            train=train_data, 
            load=True, 
            model_path='saved/'+model_path,
            tok_path='saved/'+tokenizer_path,
        )
        f = f'shap_values/shap_values_per_sent_{train_size}.pkl'
        filenames.append(f)
        with open(f, 'wb') as file:
            pickle.dump(shap_values_per_sent, file)
        shutil.rmtree('saved/'+model_path)
        shutil.rmtree('saved/'+tokenizer_path)
    return filenames

def get_shap_values_as_list(train, mod_name=None, tok_name=None):
    shaps = extract_importance(train, 'prediction/text/models/'+mod_name, 'prediction/text/models/'+tok_name)
    shaps = [x[:,0] for x in shaps]
    shaps_s = [x for x in shaps]
    # print(len(shaps_s))
    return shaps_s
