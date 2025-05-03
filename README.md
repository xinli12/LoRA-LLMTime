# LoRA Fine-Tuning of Qwen2.5 for Token-Based Time Series Forecasting

This project explores the application of large language models (LLMs), specifically Qwen2.5-0.5B-Instruct, to time series forecasting. It outlines a complete pipeline from preprocessing using the LLMTIME framework to tokenisation and encoding of time series data. The study evaluates both the zero-shot performance of the untrained model and the improvements achieved through Low-Rank Adaptation (LoRA) fine-tuning under computational constraints.

## 📋 Features

- Convert numerical time series data to a text-based format for language models
- Fine-tune LLMs with LoRA for efficient adaptation
- Evaluate forecasting performance against baseline methods
- Hyperparameter optimisation for model tuning
- Comprehensive evaluation metrics

## 🔧 Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 📁 Project Structure

```
.
├── data/                       # Data files
├── logs/                       # Execution logs
├── notebooks/                  # Jupyter notebooks
├── results/                    # Results from various runs
│   ├── baseline_evaluation/    # Baseline model evaluation results
│   ├── lora/                   # Initial model checkpoints
│   ├── hyperparam_search/      # Hyperparameter search results
│   ├── model_evaluation/       # Initial model evaluation results
│   ├── final_model/            # Final model checkpoints
│   ├── final_model_evaluation/ # Final model evaluation results
│   └── notebook_figs/          # Figures from notebooks
├── scripts/                    # Shell scripts for experiments
└── src/                        # Source code
    ├── preprocessor.py         # LLMTIME preprocessing
    ├── evaluate_baseline.py    # Baseline evaluation
    ├── train_lora.py           # LoRA fine-tuning
    ├── hyperparameter_search.py # Hyperparameter optimization
    └── utils/                  # Utility functions
```

## 🚀 Quick Start

### Data Preprocessing

The LLMTIME approach converts numerical time series into text format with comma-separated variables and semicolon-separated timesteps:

```python
from src.preprocessor import LLMTIMEPreprocessor

# Example: Format numerical data as text
series = np.array([[1.5, 2.3], [1.6, 2.2]])
formatted = LLMTIMEPreprocessor.format_timeseries(series)
# Output: "1.50,2.30;1.60,2.20"
```

### Training

Run basic training with default parameters:

```bash
bash scripts/train.sh
```

Or specify custom parameters:

```bash
python src/train_lora.py --rank 4 --lr 1e-5 --max_steps 10000 --batch_size 4 --ctx_length 512
```

### Evaluation

Evaluate baseline model performance:

```bash
bash scripts/evaluate_baseline.sh
```

Evaluate a specific (trained) model:
```bash
bash scripts/evaluate_model.sh
```

### Hyperparameter Optimization

Run hyperparameter search:

```bash
bash scripts/hyperparam_search.sh
```


## 💼 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Accelerate
- Wandb
- h5py
- scikit-learn
- matplotlib
- seaborn
- plotly


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🛠️ AI Assistance Statement

This project utilized AI tools to enhance development efficiency and quality. AI contributions include:

### Development Assistance
- Initial code scaffolding and project structure design
- Template generation for configuration files (`pyproject.toml`, `.gitignore`, etc)
- Script prototyping for data preprocessing and model training (`.py`, `.sh`)
- Debugging assistance and performance optimization

### Documentation
- README generation and documentation structuring
- Code commenting and function documentation

### Code Quality
- Refactoring suggestions for improved readability
- Optimization of implementations

**Tools used**: Claude (Anthropic), ChatGPT (OpenAI), Grok (xAI)
