import pickle
import spacy
import os

def load_ner_model(model_path):
    with open(model_path, 'rb') as model_file:
        loaded_nlp = pickle.load(model_file)
    return loaded_nlp

def predict_labels(ner_model, text):
    try:
        doc = ner_model(text)
        labels = [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
        print(f"Input text: {text}")
        print(f"Predicted labels: {labels}")
        return labels
    except Exception as e:
        print(f"Error predicting entities: {str(e)}")
        raise