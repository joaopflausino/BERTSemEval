import torch
import argparse
import json
from transformers import BertTokenizer
from model import BertSentimentClassifier

class SentimentPredictor:
    
    
    def __init__(self, model_path, model_name="bert-base-uncased", max_length=128, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        
        self.model = BertSentimentClassifier(
            bert_model_name=model_name,
            num_labels=3
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        
        self.id_to_label = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
    
    def predict(self, text):
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        
        
        sentiment = self.id_to_label[pred_label]
        confidence_scores = {
            "negative": float(probs[0, 0]),
            "neutral": float(probs[0, 1]),
            "positive": float(probs[0, 2])
        }
        
        result = {
            "text": text,
            "sentiment": sentiment,
            "confidence_scores": confidence_scores,
            "label_id": pred_label
        }
        
        return result
    
    def predict_batch(self, texts):
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results


def main():
    # exemplo de uso
    # python train.py --train_dir dataset/train --eval_file dataset/test/SemEval2017-task4-test.subtask-A.english.txt --output_dir output
    parser = argparse.ArgumentParser(description="Predict sentiment using BERT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pre-trained BERT model name")
    parser.add_argument("--input_file", type=str, help="Path to input file with one text per line")
    parser.add_argument("--output_file", type=str, help="Path to output file for predictions")
    parser.add_argument("--text", type=str, help="Text to predict sentiment for")
    
    args = parser.parse_args()
    
    
    predictor = SentimentPredictor(
        model_path=args.model_path,
        model_name=args.model_name
    )
    
    
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch(texts)
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        else:
            for result in results:
                print(f"Text: {result['text']}")
                print(f"Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence_scores']}")
                print("-" * 50)
    
    elif args.text:
        result = predictor.predict(args.text)
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence_scores']}")
    
    else:
        print("Please provide either --input_file or --text")


if __name__ == "__main__":
    main()