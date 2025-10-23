# BERTSemEval - Multi-Model Sentiment Analysis Framework

A comprehensive framework for comparing transformer models on sentiment analysis tasks, optimized for AMD GPUs with ROCm support.

## 🏗️ Project Structure

```
BERTSemEval/
├── src/
│   ├── models/          # Model implementations (BERT, RoBERTa, DistilBERT, ELECTRA, BERTweet)
│   ├── data/            # Data loading and preprocessing
│   ├── training/        # Training utilities with ROCm support
│   ├── evaluation/      # Model evaluation and comparison
│   └── utils/           # Configuration, visualization, and utilities
├── configs/             # Configuration files for different models
├── experiments/         # Experiment results and saved models
├── notebooks/           # Jupyter notebooks for analysis
└── dataset/            # SemEval dataset (train/test splits)
```
## To Do List
- make a better preprocessing
- make a data analysis in all of the data
- validate results with other projects and check the reliability of the model

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd BERTSemEval

# Install dependencies
pip install -r requirements.txt
```

### ROCm Setup for AMD GPUs

If using AMD GPUs, ensure ROCm is properly installed:

```bash
# Set environment variable (optional)
export ROC_VISIBLE_DEVICES=0

# Verify PyTorch can see your AMD GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

### Running Single Experiments

Train individual models using configuration files:

```bash
# Train BERT model
python train_model.py --config configs/bert_base.yaml

# Train RoBERTa model  
python train_model.py --config configs/roberta_base.yaml

# Train DistilBERT model
python train_model.py --config configs/distilbert_base.yaml

# Train ELECTRA model
python train_model.py --config configs/electra_base.yaml
```

### Running Batch Experiments

Run all model comparisons automatically:

```bash
python run_experiments.py
```

This will:
1. Train all configured models sequentially
2. Generate comparison results automatically
3. Create visualizations and HTML report

### Comparing Models

Compare trained models manually:

```bash
python compare_models.py --experiments bert_base roberta_base distilbert_base --report
```

## 📊 Supported Models

| Model | Configuration | Description |
|-------|--------------|-------------|
| BERT Base | `configs/bert_base.yaml` | Standard BERT-base-uncased |
| RoBERTa Base | `configs/roberta_base.yaml` | RoBERTa-base model |
| DistilBERT | `configs/distilbert_base.yaml` | Distilled BERT (faster, smaller) |
| ELECTRA Base | `configs/electra_base.yaml` | ELECTRA discriminator model |
| BERTweet | `configs/bertweet_base.yaml` | specific tweet BERT model |

## ⚙️ Configuration

Each model has its own YAML configuration file in the `configs/` directory. You can customize:

- **Model parameters**: architecture, dropout, etc.
- **Training hyperparameters**: learning rate, batch size, epochs
- **Data parameters**: max sequence length, dataset paths
- **Output settings**: save location, logging options

Example configuration structure:

```yaml
experiment:
  name: "model_experiment"
  seed: 42

model:
  type: "bert"
  name: "bert-base-uncased" 
  num_labels: 3
  dropout_prob: 0.1

training:
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 10
```

## 📈 Model Comparison Features

The framework provides comprehensive model comparison:

- **Performance Metrics**: Accuracy, F1-score, Precision, Recall
- **Training Curves**: Loss and accuracy over epochs
- **Confusion Matrices**: Per-class performance analysis  
- **Resource Usage**: Memory consumption and training time
- **Statistical Analysis**: Detailed per-class metrics

## 🎯 Results and Analysis

Results are automatically saved to the `experiments/` directory:

```
experiments/
├── bert_base/
│   ├── model.pt          # Trained model weights
│   ├── metrics.json      # Training metrics
│   └── config.yaml       # Used configuration
├── roberta_base/
└── comparison_results/
    ├── model_comparison.csv
    ├── training_curves.png
    ├── performance_comparison.png
    └── comparison_report.html
```

## 🔧 Development

### Adding New Models

1. Create a new model class in `src/models/`
2. Update the model factory in `src/utils/model_factory.py`
3. Create a configuration file in `configs/`
4. Add the import in `src/models/__init__.py`

### Custom Evaluation

Use the evaluation module for custom analysis:

```python
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.comparison import ModelComparison

# Evaluate single model
evaluator = ModelEvaluator(model, device)
metrics = evaluator.evaluate(test_dataloader)

# Compare multiple models
comparator = ModelComparison()
results = comparator.load_experiment_results(['bert_base', 'roberta_base'])
```

## 🐛 Troubleshooting

### AMD GPU Issues
- Ensure ROCm is properly installed and compatible with your GPU
- Set `torch.backends.cudnn.benchmark = False` for ROCm stability
- Monitor GPU memory usage with `rocm-smi`

### Memory Issues
- Reduce batch size in configuration files
- Use gradient accumulation for effective larger batch sizes
- Monitor memory usage in training logs

### Performance Optimization
- Enable mixed precision training for supported models
- Use DataLoader num_workers for faster data loading
- Adjust max_sequence_length based on your data distribution

## 📝 License

None (for now)

## 📊 Experimental Results

### Model Performance Comparison

| Model | Parameters | Best F1-Score | Validation Accuracy | Training Duration | GPU Memory (GB) |
|-------|------------|---------------|-------------------|------------------|----------------|
| **BERT Base (Regularized)** | 109.8M | 0.7317 | 73.17% | 00:34:06 | 1.77 |
| **RoBERTa Base (Regularized)** | 124.9M | 0.7313 | 73.13% | 00:41:30 | 2.00 |
| **DistilBERT Base (Regularized)** | 66.7M | 0.7190 | 71.90% | 00:17:49 | 1.09 |
| **ELECTRA Base (Regularized)** | 109.2M | 0.7406 | 74.06% | 00:33:07 | 1.76 |
| **BERTweet Base (Regularized)** | 135.2M | 0.7483 | 74.83% | 00:35:11 | 2.15 |

### Key Findings

**🏆 Top Performing Models:**
1. **BERTweet Base**: 74.83% F1-score - Best overall performance, specialized for Twitter data
2. **ELECTRA Base**: 74.06% F1-score - Strong performance with efficient architecture
3. **BERT Base**: 73.17% F1-score - Solid baseline performance
4. **RoBERTa Base**: 73.13% F1-score - Competitive with BERT

**⚡ Efficiency Leaders:**
- **DistilBERT Base**: Fastest training (17:49) with lowest memory usage (1.09 GB)
- **ELECTRA Base**: Best performance-to-speed ratio (74.06% F1 in 33:07)

**🎯 Model Insights:**
- BERTweet's Twitter-specific pretraining provides 0.77% advantage over ELECTRA
- ELECTRA significantly outperforms BERT/RoBERTa despite similar parameter counts
- DistilBERT achieves 71.90% F1 with only 66.7M parameters (best efficiency)
- GPU memory usage scales predictably with model size (1.09 GB to 2.15 GB)

**💾 Resource Analysis:**
- Training times range from 17:49 (DistilBERT) to 41:30 (RoBERTa)
- All models trained with regularization techniques for optimal performance
- Memory-efficient training possible on consumer GPUs (< 2.2 GB VRAM)

### Per-Class Performance (Best Models)

**BERTweet Base - Best Overall (Epoch 1):**
- Negative: 68.61% F1, 67.66% Recall, 69.59% Precision
- Neutral: 74.18% F1, 76.88% Recall, 71.66% Precision
- Positive: 78.02% F1, 75.24% Recall, 81.02% Precision

**ELECTRA Base - Runner-up (Epoch 1):**
- Negative: 68.60% F1, 71.93% Recall, 65.57% Precision
- Neutral: 73.23% F1, 74.44% Recall, 72.05% Precision
- Positive: 77.16% F1, 74.24% Recall, 80.32% Precision

**BERT Base - Strong Baseline (Epoch 1):**
- Negative: 65.48% F1, 65.39% Recall, 65.56% Precision
- Neutral: 73.52% F1, 77.78% Recall, 69.70% Precision
- Positive: 75.80% F1, 70.88% Recall, 81.46% Precision

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{bertsemeval,
  title={BERTSemEval: Multi-Model Sentiment Analysis Framework},
  author={João Pedro Flausino de Lima},
  year={2024}
}
```
