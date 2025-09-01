# Neel Nanda MATS 9.0 Application Code

This repository contains the code for a mechanistic interpretability investigation into how a GPT-2 model, trained from scratch on random walks over 4x4 grids, develops internal representations of space.

Using a suite of causal experiments, we trace the emergence of a "cognitive map" through the model's layers, identifying the specific circuits responsible for directional updates, coordinate representation, and functional reasoning.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model checkpoint is available at the specified path in each script.

## Experiment Structure

### Core Experiments

1. **Directional Update Circuit Analysis** (`exp1_activation_patching.ipynb`)
   - Comprehensive suite of causal experiments
   - Component path patching, block output patching
   - Attention pattern visualization and redundancy analysis

2. **PCA & Linear Probing Analysis** (`exp2_pca_linear_probing.ipynb`)
   - PCA of node token hidden states for early, middle and late layers
   - Tests coordinate decodability across transformer layers

4. **Direction Swapping Experiments** (`exp3_direction_swapping.ipynb`)
   - Tests functional map validation by swapping directions in walk history
   - Compares unconstrained vs. corner node scenarios
   - Measures cosine similarity between original and swapped representations

5. **Direction Ablation Experiments** (`exp4_direction_ablation.ipynb`)
   - Tests when direction information becomes redundant
   - Layer-swept ablation to identify transition points


## File Organization

- `exp1_activation_patching.ipynb`: Directional update circuit analysis
- `exp2_pca_linear_probing.ipynb`: PCA and linear probing analysis
- `exp3_direction_swapping.ipynb`: Direction swapping experiments
- `exp4_direction_ablation.ipynb`: Direction ablation experiments
- `utils/`: Utility functions for grid generation and model interaction

## Dependencies

- PyTorch
- Transformers
- NetworkX
- NumPy
- Matplotlib
- Pandas
- Scikit-learn
- tqdm

## Model Requirements

- GPT-2 based model checkpoint.
- 12-layer transformer architecture.
- Trained on the grid navigation task as described in the project write-up.
