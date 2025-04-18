# Libraries
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn as nn

# Main function
def classifier (tokenizer, model, device: str, input_text: str) -> str:
    ## SDG label prediction
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        output = model(**inputs)
        # NOTE:
        # output.logits = output[0] = output.last_hidden_state
        # print(f"logits: {output.logits}") => ERROR, because AutoModel was used instead of AutoModelForSequenceClassification
        logits = output.logits

        if device == "gpu":
            logits = logits.detach().cpu().numpy()

        # NOTE:
        # understanding bert logit output: https://discuss.huggingface.co/t/decoding-the-predicted-output-array-in-distilbertbase-uncased-model-for-ner/10673/2
        # logit shape = [batch size x num_labels]
        logits = logits.numpy()
        # print(np.shape(logits))
        predicted_label = np.argmax(logits, axis=1)
    # print(predicted_label)

    ## Map the int sdg label into sdg name
    sdg_label_map = {
        0: 'sdg_0',
        1: 'sdg_1',
        2: 'sdg_2',
        3: 'sdg_3',
        4: 'sdg_4',
        5: 'sdg_5',
        6: 'seg_6',
        7: 'sdg_7',
        8: 'sdg_8',
        9: 'sdg_9',
        10: 'sdg_10',
        11: 'sdg_11',
        12: 'sdg_12',
        13: 'sdg_13',
        14: 'sdg_14',
        15: 'sdg_15',
        16: 'sdg_16',
        17: 'sdg_17'
    }
    # print(sdg_label_map[predicted_label[0]])

    return sdg_label_map[predicted_label[0]]

# Test this file
if __name__ == "__main__":
    # Load model
    token_ckpt = "sadickam/sdg-classification-bert"
    model_ckpt = "../current_batch" 
                # this is possible because Arnab saved the model using Huggingface trainer.save_model()
    tokenizer = AutoTokenizer.from_pretrained(token_ckpt)
    
    # in order to make the model doing classification, we must import the model with
    # AutoModelForSequenceClassification. Otherwise, the model output is uninterpretable.
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

    result = classifier(tokenizer, model, "cpu", "solar pannels installed on a milk fame")
    print(result, type(result))