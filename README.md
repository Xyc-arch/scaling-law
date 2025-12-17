## Main Experiments

This section describes how to reproduce the main experimental results. All commands should be executed from the codebase root directory.

### Step 0. Generate simulated Craft data (Craft only)

The Craft dataset is synthetically generated using a fully specified data-generating process.

    python craft.py

This produces:
    data/craft/craft.csv

### Step 1. Train–test split

For each dataset (Craft, Gender, Diabetes, Adult), split the full dataset into a training (“use”) set and a held-out test set. The test set is never used for model training or synthetic data generation.

    python use_test_split.py

This produces:
    data/{dataset}/use.csv
    data/{dataset}/test.csv

### Step 2. Seed–train split

Split the training (“use”) data into a balanced seed dataset and a disjoint raw training dataset. The seed dataset is balanced across spurious–label groups and is used to prompt and fine-tune language models.

    python train_seed_split.py

This produces:
    data/{dataset}/seed.csv
    data/{dataset}/train.csv
    data/{dataset}/train0.csv
    data/{dataset}/train1.csv
    data/{dataset}/train_pos.csv
    data/{dataset}/train_neg.csv

### Step 3. Synthetic data generation

LLM-based synthetic data generation

Fine-tune a GPT-2–based tabular generator on the seed data and sample synthetic records.

    python finetune_inference.py

This produces:
    data/{dataset}/synthetic_allseed.csv

Baseline oversampling methods

Generate synthetic data using SMOTE, ADASYN, or Random Over-Sampling.

    python baselines.py

This produces:
    data/{dataset}/smote.csv
    data/{dataset}/adasyn.csv
    data/{dataset}/ros.csv

### Step 4. Split synthetic data by label and spurious groups

All synthetic datasets (LLM-based or baseline) are further split into label-based and spurious-correlation-based subsets for downstream evaluation.

    python split_synthetic.py

This produces:
    data/{dataset}/{method}1.csv
    data/{dataset}/{method}0.csv
    data/{dataset}/{method}_pos.csv
    data/{dataset}/{method}_neg.csv

where {method} is one of synthetic_allseed, smote, adasyn, or ros.

### Step 5. Downstream evaluation

Imbalanced classification experiments

    python imbalanced_class_vary_ratio.py
    python imbalanced_class_vary_additional.py

Spurious correlation experiments

    python spurious_corr_vary_ratio.py
    python spurious_corr_vary_additional.py

Results are saved as JSON files under each dataset directory and can be visualized using the provided plotting scripts.




## Ablation Studies

We conduct a series of ablation experiments to isolate the sources of performance gains and to rule out alternative explanations such as data leakage or seed size effects. The ablations are designed to be orthogonal and target different potential failure modes.

### Fixed Seed Size Ablation (Small-Seed)

In this ablation, we fix the size of the seed dataset to a small constant (200 samples, balanced across groups) and generate synthetic data using the same procedure as in the main experiments.

The seed dataset is still drawn from the training pool, and downstream classifiers are trained on the remaining training data. This setting tests the data efficiency of the LLM-based generator and evaluates whether high-quality synthetic data can be produced from a limited number of examples.

Scripts:
    ablate_great.py
    ablate_imb.py
    ablate_spurious.py

Synthetic data:
    synthetic_seed.csv

Purpose:
    Evaluate robustness to limited seed size.

### Independent Synthetic Data Ablation (IND)

This ablation enforces sample-level independence between synthetic data and downstream training data.

We first construct an independent seed pool by removing all original seed samples from the training set:
    seed_ind = use − seed

Synthetic data are generated exclusively from this independent seed pool. Downstream classifiers are trained only on raw training data that never appear in the seed used for generation. This guarantees that synthetic samples are statistically independent of the training set at the sample level.

Scripts:
    create_seed_ind.py
    ablate_great.py
    ablate_ind_syn_split.py
    ablate_imb_ind.py
    ablate_spurious_ind.py

Synthetic data:
    synthetic_seed_ind.csv

Purpose:
    Rule out performance gains due to overlap, memorization, or data leakage between synthetic and training data.

### Summary of Ablations

    Fixed seed size tests how much data the generator needs.
    IND tests whether gains persist without any sample-level dependence between synthetic and training data.

Together, these ablations demonstrate that the observed improvements are driven by distribution learning rather than seed size artifacts or data leakage.

