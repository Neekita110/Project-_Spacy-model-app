import os
from fastapi import FastAPI, HTTPException, Form, File, UploadFile, Query
from controller import load_ner_model, train_ner_model_with_file, load_config, save_ner_model
from predict import predict_labels
from fastapi import Path
# Neekita code

app = FastAPI()

# Load configuration
config = load_config()

# Update models directory and model identifier
MODELS_DIR = config.get("models_dir", "/content/ner_model")  # Update the models directory path
MODEL_IDENTIFIER = config.get("identifier", "ner_model")  # Default identifier


# Load the NER model (this is a placeholder, load your actual model)
ner_model = None


def load_ner_model_global():
    global ner_model
    ner_model_path = os.path.join(MODELS_DIR, f"{MODEL_IDENTIFIER}.pickle")
    print(f"Attempting to load NER model from: {ner_model_path}")

    # Check if the file exists
    if os.path.exists(ner_model_path):
        ner_model = load_ner_model(ner_model_path)
        print(f"NER model loaded successfully.")
    else:
        print(f"NER model file not found at: {ner_model_path}")



def contains_special_characters(text, allow_special_chars=False):
    # Check if the text contains non-alphanumeric characters (excluding spaces)
    if allow_special_chars:
        return False
    return any(not char.isalnum() and not char.isspace() for char in text)

def is_valid_identifier(identifier):
    # Check if the identifier contains special characters
    return True  # You can customize this validation as needed

def is_valid_text_input(text_input):
    # Check if the text input is valid
    return not text_input.isspace()

@app.post("/train")
def train_ner_model(file: UploadFile = File(...), identifier: str = Form(...)):
    try:
        # Check if the identifier is valid
        if not is_valid_identifier(identifier):
            raise ValueError("Invalid identifier. Use alphanumeric characters and spaces only.")

        # Handle training with the uploaded file
        content = file.file.read().decode("utf-8")

        # Save the training data to a temporary file
        temp_file_path = os.path.join("temp_training_data", f"{identifier}.csv")
        os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(content)

        # Train the NER model
        result_message = train_ner_model_with_file(temp_file_path, identifier, MODELS_DIR)

        # Clean up the temporary file
        os.remove(temp_file_path)

        return {"message": result_message}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_message = f"Error during training: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)





@app.post("/save_model")
def save_model(identifier: str = Query(..., description="Model identifier")):
    try:
        global ner_model  # Ensure the global variable is referenced

        # Debug prints/logs to trace the state
        print(f"Current state of ner_model: {ner_model}")

        # Check if the model is loaded
        if ner_model is None:
            raise HTTPException(status_code=500, detail="NER model is not loaded. Check the model loading process.")

        # Update the path to save the model
        output_path = os.path.join(MODELS_DIR, f'{identifier}.pickle')

        # Further code for saving the model...
        save_ner_model(ner_model, output_path)

        return {"message": f"NER model saved successfully as {identifier}.pickle"}
    except Exception as e:
        error_message = f"Error saving model: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/get_labels_endpoint")
def get_labels_endpoint(text_input: str = Query(..., description="Input text")):
    try:
        # Check if the input text is valid
        if not is_valid_text_input(text_input):
            raise ValueError("Invalid text input. Check your text again.")

        # Load the NER model if it is not already loaded
        global ner_model
        if ner_model is None:
            load_ner_model_global()

        # Ensure the model is loaded successfully
        if ner_model is None:
            raise ValueError("NER model not loaded successfully. Check the model loading process.")

        # Predict entities using the specified model
        labels = predict_labels(ner_model, text_input)

        return {"text": text_input, "labels": labels}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        error_message = "There is something wrong. Please check your text again."
        print(f"Error in get_labels_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)