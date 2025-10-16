import torch
import argparse
import json
import os
from transformers import AutoTokenizer
from model_bertweet import BertTweetSentimentClassifier

class BertTweetSentimentPredictor:
    
    def __init__(self, model_path, model_name="vinai/bertweet-base", max_length=128, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        print(f"Loading BERTweet model: {model_name}")
        
        # Load tokenizer (try from model directory first, then from pretrained)
        model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
        try:
            if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
                print(f"Loading tokenizer from {model_dir}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
            else:
                print(f"Loading tokenizer from {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e:
            print(f"Warning: Error loading tokenizer: {e}")
            print(f"Fallback to default tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        self.max_length = max_length
        
        # Load model configuration
        config_path = os.path.join(model_dir, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_name = config.get("model_name", model_name)
                dropout_prob = config.get("dropout_prob", 0.1)
                print(f"Loaded model config: {config}")
        else:
            dropout_prob = 0.1
            print("No model config found, using defaults")
        
        # Initialize model
        self.model = BertTweetSentimentClassifier(
            bertweet_model_name=model_name,
            num_labels=3,
            dropout_prob=dropout_prob
        )
        
        # Load model weights
        if os.path.isfile(model_path):
            print(f"Loading model weights from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.id_to_label = {
            0: "negative",
            1: "neutral", 
            2: "positive"
        }
        
        print("BERTweet Sentiment Predictor initialized successfully!")
    
    def _preprocess_tweet(self, text):
        """
        Preprocess tweet text for BERTweet (same as in dataset reader)
        """
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text
    
    def predict(self, text, return_all_scores=True):
        """
        Predict sentiment for a single text
        
        Args:
            text (str): Input text
            return_all_scores (bool): Whether to return confidence scores for all classes
            
        Returns:
            dict: Prediction results
        """
        # Preprocess text
        processed_text = self._preprocess_tweet(text)
        
        # Tokenize
        encoding = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get probabilities and prediction
        probs = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        max_confidence = float(probs[0, pred_label])
        
        # Prepare result
        sentiment = self.id_to_label[pred_label]
        
        result = {
            "text": text,
            "processed_text": processed_text,
            "sentiment": sentiment,
            "label_id": pred_label,
            "confidence": max_confidence
        }
        
        if return_all_scores:
            confidence_scores = {
                "negative": float(probs[0, 0]),
                "neutral": float(probs[0, 1]),
                "positive": float(probs[0, 2])
            }
            result["confidence_scores"] = confidence_scores
        
        return result
    
    def predict_batch(self, texts, batch_size=32, return_all_scores=True):
        """
        Predict sentiment for multiple texts efficiently
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            return_all_scores (bool): Whether to return confidence scores for all classes
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch
            processed_texts = [self._preprocess_tweet(text) for text in batch_texts]
            
            # Tokenize batch
            encodings = self.tokenizer(
                processed_texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Process results
            probs = torch.softmax(logits, dim=1)
            pred_labels = torch.argmax(probs, dim=1).cpu().numpy()
            
            for j, (text, processed_text, pred_label) in enumerate(zip(batch_texts, processed_texts, pred_labels)):
                sentiment = self.id_to_label[pred_label]
                confidence = float(probs[j, pred_label])
                
                result = {
                    "text": text,
                    "processed_text": processed_text,
                    "sentiment": sentiment,
                    "label_id": int(pred_label),
                    "confidence": confidence
                }
                
                if return_all_scores:
                    confidence_scores = {
                        "negative": float(probs[j, 0]),
                        "neutral": float(probs[j, 1]),
                        "positive": float(probs[j, 2])
                    }
                    result["confidence_scores"] = confidence_scores
                
                results.append(result)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment using BERTweet model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--model_name", type=str, default="vinai/bertweet-base", help="Pre-trained BERTweet model name")
    parser.add_argument("--input_file", type=str, help="Path to input file with one text per line")
    parser.add_argument("--output_file", type=str, help="Path to output file for predictions")
    parser.add_argument("--text", type=str, help="Text to predict sentiment for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batch prediction")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BertTweetSentimentPredictor(
        model_path=args.model_path,
        model_name=args.model_name
    )
    
    # Process input
    if args.input_file:
        print(f"Processing file: {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(texts)} texts to process")
        results = predictor.predict_batch(texts, batch_size=args.batch_size)
        
        if args.output_file:
            print(f"Saving results to: {args.output_file}")
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        else:
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Text: {result['text']}")
                print(f"Sentiment: {result['sentiment']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"All scores: {result['confidence_scores']}")
    
    elif args.text:
        print(f"Processing single text: {args.text}")
        result = predictor.predict(args.text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"All scores: {result['confidence_scores']}")
    
    else:
        print("Please provide either --input_file or --text")


if __name__ == "__main__":
    main()