# Multi-Label Fake News Detection on LUN Dataset

<!-- Add buttons here -->
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/navendu-pottekkat/awesome-readme?include_prereleases)  ![GitHub pull requests](https://img.shields.io/github/issues-pr/navendu-pottekkat/awesome-readme)  ![GitHub All Releases](https://img.shields.io/github/downloads/navendu-pottekkat/awesome-readme/total)

# Table of contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Directory Structure](#directory_structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

# Introduction

This is the final project of CS4248 for Group 17. 

**Motivation**: The spread of fake news is a big challenge in this digital world owing to the negative influence and difficulty of identifying them. There has been effort made to solve this problem, but we note that most solutions are black box models and thus lack interpretability. 

**Description**: In this project, we constructed several fake news classifiers by applying machine learning techniques on the LUN dataset and attempt to interpret their results. Our best model (Logistic regression), achieved a macro-F1 score of 0.76. 

**Novelty:**  (1) construction of a new CNN model with a macro-F1 score of 0.65 (2) in-depth analysis and interpretation of the logistic regression model (3)  verification of categorization based on 2 dimensions (intention of author and information quality).

**Results Summary (Best)**:
| Method |Feature Engineering | Macro-F1|Accuracy|Precision|Recall
|--|--|--|--|--|--|
|Logistic Regression | TF-IDF |0.7559|0.7643|0.7793|0.7643|
|LSTM | GloVe |0.67|0.67|0.68|0.68|
|CNN | None |0.65|0.65|0.64|0.64|

[(Back to top)](#table-of-contents)

# Dataset

**1. Dataset Source:**  [Download link](https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset) (dataset originally constructed by [Rashkin et al.](https://aclanthology.org/D17-1317.pdf))

**2. Dataset Introduction:**

 - Sources of the dataset. (GN refers to Gigawords News)

|**Dataset**  |**Satire (#Docs)**  |**Hoax (#Docs)**|**Propaganda(#Docs)**|**Trusted (#Docs)**|
|--|--|--|--|--|
| Training data | The Onion(14,047) |American News (6,942)|Activist Report (17,870)|GN except “APW” and “WPB” (9,995)|
| Test data | The Borowitz Report Clickhole (750)|DC Gazette (750)|The Natural News (750)|GN only “APW” and “WPB” (750)|

 - Size: 48K news articles for training, 3K for testing.
 - Labels: 1-"Satire", 2-"Hoax", 3-"Propaganda", 4-"Reliable News"

[(Back to top)](#table-of-contents)

# Directory_Structure

```
FakeNewsDetection
├─ BERT.ipynb                          
├─ LIWC
│  ├─ README.md
│  ├─ __pycache__
│  │  └─ util.cpython-310.pyc
│  ├─ features.py
│  ├─ to_structure.py
│  └─ util.py
├─ LSTM.ipynb
├─ Logistic_Regression.ipynb
├─ README.md
├─ XGBoost.ipynb
├─ cnn.py
├─ extract_useful_n_grams.ipynb
├─ mlp.py
├─ paper
│  ├─ 1910.12203.pdf
│  ├─ 2021.acl-long.62.pdf
│  └─ D17-1317.pdf
└─ validate_2d_representation_random_forest.ipynb

```
[(Back to top)](#table-of-contents)

# Installation
[(Back to top)](#table-of-contents)

# Dependencies
[(Back to top)](#table-of-contents)
# Usage
[(Back to top)](#table-of-contents)

# License
[(Back to top)](#table-of-contents)
