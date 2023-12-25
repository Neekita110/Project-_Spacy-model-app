import os
import spacy
import yaml
import pickle
import csv
import pandas as pd
from ast import literal_eval

def load_config(config_path=None):
    if config_path is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(script_dir, "config.yaml")

    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        config = {"models_dir": "models", "identifier": "ner_model"}  # Default identifier

    return config

def load_ner_model(model_path):
    try:
        with open(model_path, 'rb') as model_file:
            loaded_nlp = pickle.load(model_file)
        return loaded_nlp
    except Exception as e:
        print(f"Error loading NER model: {str(e)}")
        return None

def save_ner_model(ner_model, output_path):
    with open(output_path, 'wb') as model_file:
        pickle.dump(ner_model, model_file)

def check_label_examples(label_counts):
    for label, count in label_counts.items():
        if count < 5:
            return f"Some labels have less than 5 examples. Please check label: {label}"
    return None

def load_csv_data(file_path):
    # Load CSV data using pandas or your preferred method
    # Replace this with your actual data loading logic
    data = pd.read_csv(file_path)

    # Assuming your CSV has 'sentence' and 'entities' columns
    sentences = data['sentence'].tolist()
    entities_str = data['entities'].tolist()

    # Convert string representation of entities to Python objects
    entities = [literal_eval(entities_str_item) for entities_str_item in entities_str]

    # Combine sentences and entities into a list of tuples
    loaded_data = list(zip(sentences, entities))

    return loaded_data

def load_training_data(file_path):
    training_data = []
    label_counts = {}

    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')  # Assuming tab-separated values
        for row in csv_reader:
            if len(row) >= 2:
                text, label_list = row[0], row[1]
                label_list = eval(label_list)
                entities = [(start, end, label) for (start, end, label) in label_list]

                training_data.append((text, {"entities": entities}))

                for (_, _, label) in label_list:
                    label_counts[label] = label_counts.get(label, 0) + 1

    return training_data, label_counts


def train_ner_model_with_data(training_data, output_identifier, models_dir):
    nlp = spacy.blank("en")

    # Increase the number of training iterations
    for _ in range(20):  # You can adjust the number of iterations as needed
        for example in training_data:
            print(f"Training Example: {example}")
            nlp.update([example], drop=0.5, losses={})

    output_path = os.path.join(models_dir, f'{output_identifier}.pickle')
    save_ner_model(nlp, output_path)
    print(f"Model saved at: {output_path}")


def train_ner_model_with_file(file_path, output_identifier, models_dir):
    training_data, label_counts = load_training_data(file_path)

    error_message = check_label_examples(label_counts)
    if error_message:
        return error_message

    train_ner_model_with_data(training_data, output_identifier, models_dir)
    return f"NER model trained successfully. Saved as {output_identifier}.pickle in the '{models_dir}' directory."

def main():
    csv_file_path = os.path.join(os.getcwd(), 'training_data.csv')
    result_message = train_ner_model_with_file(csv_file_path, "ner_model", "models")
    print(result_message)

if __name__ == "__main__":
    main()