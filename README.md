# CLAE & DCLAE: Causal Linear AutoEncoder Framework

This project is a framework for Linear AutoEncoders for Recommendation, including **RLAE**, **RDLAE**, and the newly added **CLAE** (Causal Linear AutoEncoder) and **DCLAE** (Dropout CLAE).

## Setup with `uv` (Recommended)

This project now supports [uv](https://github.com/astral-sh/uv) for faster and more reliable dependency management.

### Install dependencies
```bash
uv sync
```

### Run experiments
```bash
# Example: Run CLAE on yelp2018
uv run strong/code/main.py --model CLAE --dataset yelp2018 --gpu 0
```

---

## Grid Search

You can perform grid search for hyperparameters using the provided `grid_search.py` script.

### Usage
```bash
# Grid search for CLAE
uv run grid_search.py --model CLAE --dataset yelp2018 --gpu 0

# Grid search for DCLAE
uv run grid_search.py --model DCLAE --dataset yelp2018 --gpu 0
```
Results will be saved as CSV files in the `results/` directory.

---

## Datasets
- Strong generalization: [Link](https://drive.google.com/file/d/1qRDWRMp5U86jwInnWT6OirsjT4UKhNE2/view?usp=sharing)
- Weak generalization: [Link](https://drive.google.com/file/d/1Yo5roKrJ3mkKTOSHxFNz9RoOjEueQnpS/view?usp=sharing)

---

## Arguments (see more arguments in `parse.py`)
- **model**: EASE, EDLAE, RLAE, RDLAE, **CLAE**, **DCLAE**
- **reg_lambda** (CLAE/DCLAE): Ridge regularization parameter (default: 10.0)
- **alpha** (CLAE/DCLAE): Item-side normalization parameter (default: 0.5)
- **beta** (CLAE/DCLAE): User-side IPW parameter (default: 0.5)
- **dropout_p** (DCLAE): Dropout regularization probability (default: 0.3)

---

## Original Citation (SIGIR 2023)

```
@inproceedings{MoonKL23RDLAE,
  author    = {Jaewan Moon and
               Hye{-}young Kim and
               Jongwuk Lee},
  title     = {It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation},
  booktitle = {SIGIR},
  pages     = {1639--1648},
  year      = {2023},
}
```
