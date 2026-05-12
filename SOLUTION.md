# Solution

## Reproducibility

### Google Colab

Open the terminal in Colab and run:

```bash
git clone https://github.com/DashaRudas/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
pip install -r requirements.txt
python solution.py
```

### Local Setup

```bash
git clone https://github.com/DashaRudas/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
python solution.py
```

- response pooling is done on layers 13 and 16
- a small set of geometric and lexical features is appended
- the probe is a regularized logistic regression model

## Feature extraction

for each prompt-response pair:

1. hidden states are pooled from layers 13 and 16 --- the main layers for conept making are usually in the middle, I've tried different combinations of layers, this is the best version
2. six extra features are added:
   - L2 norm of the pooled response embedding --- it could be correlated with confidence and representational sharpnes
   - average token-wise standard deviation on layer 13 --- signal for instability or uneven token dynamics
   - cosine similarity between prompt and response embeddings on layer 24 --- shows how well the answer stays anchored to the prompt
   - lexical overlap between prompt and response --- reuse of prompt wording could show factological mistakes
   - type-token ratio of the response --- for detecting cycled or generic answers
   - response length

this produces a 1798-dimensional vector

## Probe

the final probe uses:

- `StandardScaler` on pooled features
- `PCA(n_components=20)` on pooled features
- `StandardScaler` on geometric features
- `LogisticRegression(class_weight="balanced", solver="lbfgs")`

the prediction threshold is tuned on validation accuracy, with a preference for thresholds close to `0.5` when scores tie

## Results

- Train AUROC: `0.8216`
- Validation AUROC: `0.7739`
- Test AUROC: `0.8473`
- Test accuracy: `0.7788`
- Test F1: `0.8527`

## Additional experiments

I also tried several other directions --- tested different hidden-state layers. The idea was to check the middle or middle-late layers, where the model has already built a semantic representation of the answer. I tried several layer windows, including `15..21`, `16..22`, and `17..23`

These experiments showed that simply moving deeper into later layers does not help. The best signal was usually around the middle-late range, especially near layers `15..20`. Later layers such as `22` and `23` performed worse.

I also tried multi-layer features, where response hidden states from several layers were not averaged together, but passed to the probe as separate blocks. This gave the classifier more information, but it also increased dimensionality too much. Since the dataset is small, the wider representation made the probe more likely to overfit. Because of that, I rejected the larger multi-layer setup.

Another tested direction was to use richer pooling, such as combining `mean`, `last token`, and `std` representations across several layers. This looked reasonable because the end of the answer and token-level variation could contain useful information. However, in practice it made the feature vector too large and reduced accuracy. Simple mean pooling was more stable.

I also tested token-dynamics and layer-delta features. Some of them were useful in diagnostics, especially around middle layers, but they were not strong enough to justify a much wider final feature set. The final solution keeps only a small number of extra features that improved stability.

Prompt-response contrast features were also tested. The motivation was that hallucinations may appear as a mismatch between the prompt representation and the response representation. I tried features based on cosine distance and boundary jumps between prompt and response states. These features were interpretable, but they did not improve local accuracy, so they were removed from the final version
