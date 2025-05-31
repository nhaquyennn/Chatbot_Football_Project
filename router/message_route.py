import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

# Tải mô hình và tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('./question_classifier_model')
model = DistilBertForSequenceClassification.from_pretrained('./question_classifier_model')
label_encoder = np.load('./question_classifier_model/label_encoder.npy', allow_pickle=True)

def route_message(message):
    try:
        encoding = tokenizer(
            message,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
        return label_encoder[predicted_class]
    except Exception as e:
        print(f"❌ Lỗi khi phân loại câu hỏi: {e}")
        return "unknown"