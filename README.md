# 📈 Predicting 30-Day Video Views from Early Engagement-Kaggle

> An end-to-end predictive modelling project that forecasts a short video's **cumulative views at day 30** using only its metadata, its creator's profile, and its **early engagement signals**.

Built for an in-class machine learning kaggle-competition.

---

## Table of Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Author](#author)

---

## Overview

Can we tell how popular a video will become a month after posting, by looking only at how it performed in its first few days?

This project answers that question by building a regression pipeline that combines three data sources:

1. **Video metadata**: duration, resolution ratio, language, topic, music, text and emotion features.
2. **Creator statistics**: followers, following, total likes received and video count on the day of posting.
3. **Early daily engagement**: plays, likes, comments, shares, saves, downloads and WhatsApp shares.

**Target:** `target_day30_views` (cumulative views at day 30)
**Evaluation metric:** RMSE (Root Mean Squared Error)

---

## Results at a Glance

Three models were compared on a held-out validation set (80/20 split, `random_state=42`): 

| Model | Validation RMSE |
|---|---:|
| **Random Forest** (tuned) | **70,719** |
| XGBoost | 176,615 |
| Linear Regression | 266,524 |

The tuned **Random Forest** was selected to generate the final competition submission.

**Random Forest hyperparameter search** (9 configurations tried):

| n_estimators | max_depth | min_samples_leaf | max_features | RMSE |
|---:|---|---:|---:|---:|
| **500** | **None** | **3** | **1.0** | **70,719** |
| 700 | None | 1 | 1.0 | 79,379 |
| 500 | None | 1 | 1.0 | 80,253 |
| 500 | 30 | 1 | 1.0 | 80,838 |
| 500 | 20 | 1 | 1.0 | 82,442 |
| 300 | None | 1 | 1.0 | 83,111 |
| 500 | None | 2 | 1.0 | 85,183 |
| 500 | None | 1 | 0.7 | 87,811 |
| 500 | None | 1 | 0.5 | 96,071 |

---

## Dataset

| File | Rows | Description |
|---|---:|---|
| `train_videos.csv` | 12,000 | Labelled videos with metadata and the day-30 target |
| `test_videos.csv` | 3,001 | Videos to predict (no target) |
| `creators_daily.csv` | 252,166 | Daily snapshot of creator statistics |
| `engagement_daily.csv` | 79,489 | Daily early-engagement counts per video |

> ⚠️ The data files are **not included** in this repository. Place them in the project root before running the notebook.

**Feature groups**

- **Video:** `duration`, `ratio`, `desc_language`, `is_english`, `created_by_ai`, `is_ads`, `topic`, `create_time` (hour of day)
- **Music:** `music_selected_from`, `music_author`, `music_owner_id`, `music_id`
- **Text:** `word_count`, `emoji_count`, `question_count`, `hashtag_count`, `speaking_rate`
- **Emotion scores:** `anger`, `joy`, `surprise`, `sadness`, `disgust`, `fear`
- **Creator:** `follower_count`, `following_count`, `total_favorited`, `video_count`
- **Early engagement:** `play_count`, `like_count`, `comment_count`, `share_count`, `collect_count`, `download_count`, `whatsapp_share_count`
- **Engineered rates:** `like_rate`, `share_rate`, `comment_rate`, `collect_rate`, `download_rate`, `whatsapp_share_rate` (each metric divided by `play_count`)

---

## Methodology

### 1. Data integration
- Aggregated `engagement_daily` per `video_id` (sum of each engagement metric).
- Joined creator statistics on `author_id` + the video's creation date.
- Joined the aggregated engagement onto the video table.

### 2. Cleaning & missing values
- Dropped `enterprise_verified` (empty).
- Filled text and emotion features with the **median**.
- Filled missing music fields and `topic` with `-1` / `"Unknown"`.
- Replaced infinite values from rate calculations and imputed them with the median.
- Removed training rows with no engagement data at all, and rows with a missing target.
- Extracted the **posting hour** from `create_time`.

### 3. Feature engineering
Created per-play engagement rates (likes, comments, shares, saves, downloads, WhatsApp shares) to capture *how* viewers interact, not just how many.

### 4. Exploratory analysis
- The target is **heavily right-skewed**, so a log-scale view was used for visualisation.
- Correlation analysis and scatter plots confirmed that early plays and likes are strongly related to day-30 views.

### 5. Modelling
A scikit-learn `Pipeline` with a `ColumnTransformer` (one-hot encoding for categorical columns, numeric columns passed through) feeding into each model:
- Linear Regression (baseline)
- Random Forest (manual hyperparameter search)
- XGBoost

### 6. Prediction
The same cleaning and feature steps were applied to the test set (using training medians where imputation was needed), and predictions were exported to `submission.csv`.

---

## Key Findings

- **Early plays dominate.** `play_count` has a correlation of ~0.94 with the target, and accounts for ~76% of the Random Forest's feature importance.
- **Other engagement counts help.** Likes (~0.86), saves (~0.63), shares (~0.61) and WhatsApp shares (~0.56) are also strongly correlated with day-30 views.
- **Creator reach matters, but less.** `follower_count` and `total_favorited` show moderate correlation (~0.26).
- **Content attributes add little.** Emotion scores, hashtags, emojis and duration have weak individual correlation with the target.
- **Tree ensembles beat the linear baseline** by a wide margin, which suggests a non-linear relationship between early engagement and long-term views.

---

## Getting Started

### Prerequisites
- Python 3.9+
- scikit-learn **1.4 or newer** (for `root_mean_squared_error`)

### Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install pandas numpy matplotlib scikit-learn xgboost jupyter
```

### Run

1. Put the four CSV files in the project root.
2. Launch the notebook:
   ```bash
   jupyter notebook Competition_Maryam_Mohammed.ipynb
   ```
3. Run all cells. The final cell writes `submission.csv` with two columns: `video_id`, `target_day30_views`.

---

## Project Structure

```
.
├── Competition_Maryam_Mohammed.ipynb   # Full pipeline: EDA → modelling → submission
├── train_videos.csv                    # (not included)
├── test_videos.csv                     # (not included)
├── creators_daily.csv                  # (not included)
├── engagement_daily.csv                # (not included)
├── submission.csv                      # Generated predictions
└── README.md
```

---

## Limitations & Future Work

Ideas for improving the current solution:

- **Model the log of the target** (e.g. `log1p`) to handle the heavy skew and reduce the influence of extreme videos on RMSE.
- **Use cross-validation** instead of a single train/validation split for more reliable model selection.
- **Remove identifier columns** (`video_id`, `author_id`, `music_id`, `music_owner_id`) from the feature set, since they carry no generalisable meaning and may add noise.
- **Engineer growth features** from the daily engagement (e.g. day-over-day growth in plays) rather than only summed totals.
- **Tune XGBoost** properly and try other gradient boosting libraries such as LightGBM or CatBoost.
- **Reduce dimensionality** of high-cardinality categoricals like `music_author`, which produce thousands of one-hot columns.

---

## Author

**Maryam Mohammed**

Feel free to open an issue or reach out with questions and suggestions.
