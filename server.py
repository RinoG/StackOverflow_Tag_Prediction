from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pickle

from torch.quantization import quantize_dynamic


app = Flask(__name__)

# Initialize your model, tokenizer, etc.
model = AutoModelForSequenceClassification.from_pretrained('fine-tuned-model')
model = quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

tokenizer = AutoTokenizer.from_pretrained('bert-tokenizer')
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

max_length = 33
device = torch.device("cpu")

def predict_tag(title, model, tokenizer, label_encoder, max_length, device):
    # Tokenize and prepare the input text
    inputs = tokenizer.encode_plus(
        title,
        None,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True
    )

    ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)

    # Get the model prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=mask)
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    # Decode the prediction
    predicted_tag = label_encoder.inverse_transform(prediction)[0]

    return predicted_tag

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    title = data['title']
    predicted_tag = predict_tag(title, model, tokenizer, label_encoder, max_length, device)
    return jsonify({'predicted_tag': predicted_tag})

if __name__ == '__main__':
    app.run(debug=True)
