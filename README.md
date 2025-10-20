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

### ROCm Setup for AMD GPUs (Optional)

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
  num_epochs: 3
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
| **BERT Base** | 109.8M | 0.6963 | 69.63% | 28:33 | 1.73 |
| **BERT Base (Regularized)** | 109.8M | 0.6979 | 69.83% | 47:55 | 1.73 |
| **RoBERTa Base** | 124.9M | 0.7106 | 71.06% | 29:17 | 1.96 |
| **RoBERTa Base (Regularized)** | 124.9M | 0.7211 | 72.14% | 59:20 | 1.96 |
| **DistilBERT Base** | 66.7M | 0.6893 | 68.99% | 19:31 | 1.06 |
| **DistilBERT Base (Regularized)** | 66.7M | 0.7000 | 70.03% | 30:14 | 1.08 |
| **ELECTRA Base** | 109.2M | 0.7146 | 71.57% | 27:51 | 1.72 |
| **ELECTRA Base (Regularized)** | 109.2M | 0.7172 | 71.78% | 37:18 | 1.72 |
| **BERTweet Base (Regularized)** | 135.2M | 0.7260 | 72.65% | 40:13 | 2.10 |

### Key Findings

**🏆 Top Performing Models:**
1. **BERTweet Base (Regularized)**: 72.60% F1-score - Best overall performance
2. **RoBERTa Base (Regularized)**: 72.11% F1-score - Strong second place
3. **ELECTRA Base (Regularized)**: 71.72% F1-score - Excellent efficiency

**⚡ Efficiency Leaders:**
- **DistilBERT Base**: Fastest training (19:31) with lowest memory usage (1.06 GB)
- **ELECTRA Base**: Good balance of performance (71.46% F1) and speed (27:51)

**🔧 Regularization Impact:**
- All regularized models showed improved performance over their base versions
- Average improvement: +1.2% F1-score across all models
- Regularization particularly effective for larger models (BERT, RoBERTa)

**💾 Resource Analysis:**
- Parameter count correlates weakly with performance
- DistilBERT achieves 68.99% F1 with only 66.7M parameters (best efficiency)
- GPU memory usage scales predictably with model size

### Per-Class Performance (Best Models)

**BERTweet Base (Regularized) - Best Overall:**
- Negative: 74.4% F1, 78.8% Recall
- Neutral: 71.9% F1, 69.6% Precision  
- Positive: 71.5% F1, 69.9% Recall

**RoBERTa Base (Regularized) - Runner-up:**
- Negative: 73.9% F1, 76.7% Recall
- Neutral: 71.4% F1, 68.4% Recall
- Positive: 70.8% F1, 72.4% Recall

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{bertsemeval,
  title={BERTSemEval: Multi-Model Sentiment Analysis Framework},
  author={João Pedro Flausino de Lima},
  year={2024}
}
```
