# DATATHON 26–27 — Round 1

## Dataset Description
Predict one of seven target classes using 174 numerical features. Feature values contain noise and missing data.

### Files
- **train.csv** — Training features with known target labels.
- **test.csv** — Test features; predict a target label for every row.
- **sample_submission.csv** — Submission template containing test IDs and placeholder predictions.

### Columns
- **id** — Unique observation identifier. Preserve it in your submission; it is not a predictive feature.
- **f1–f174** — Numerical input features. Blank values represent missing data.
- **target** — Integer class label from 1 to 7, available only in train.csv.

### Submission
Replace the placeholder targets in sample_submission.csv with your predictions. Submit exactly two columns: id and target, with one row per test ID and no additional index column.

---

## Competition Overview
**Build your model. Climb the leaderboard. Make every prediction count.**

Welcome to the online qualifier of DATATHON 26–27.

Compete individually or in a team of two in a 30+ hours, seven-class classification challenge. The top 45 eligible teams will advance to the on-campus finale.

### Your Goal
Build a machine learning model that predicts the correct target class, from 1 to 7, for each observation in the test dataset.

You will work with 100+ numerical features containing noise and missing values. Explore the data, develop a preprocessing strategy and validate your models before submitting predictions.

### How to Participate
1. Download the competition files from the Data tab.
2. Use train.csv to explore the features and develop your model.
3. Generate predictions for every observation in test.csv.
4. Submit a CSV with exactly two columns: id and target, following sample_submission.csv.
5. Track your public leaderboard score and refine your approach.
6. Select up to two final submissions before the deadline.
7. Use your registered team name on Kaggle. Both members of a two-person team must join the same Kaggle team.

### Evaluation
Submissions are ranked using **Macro F1**, which gives equal weight to each of the seven classes. Higher scores are better.

- **Public leaderboard**: 30% of the test observations.
- **Private leaderboard**: The remaining 70%, used for final ranking.
- **Submission allowance**: 25 submissions per team per UTC day, resetting at 5:30 AM IST.

Public rankings are provisional. Qualification depends on final private leaderboard ranking, eligibility, rule compliance and code verification.

### Schedule
- **Starts**: 3 October 2026 at 10:00 AM IST.
- **Submission deadline**: 4 October 2026 at 7:00 PM IST.
- **Scheduled private leaderboard release**: 4 October 2026 at 7:30 PM IST.
- **On-campus finale**: 9 October 2026 at PCCOE, Pune.

### Before You Begin
Read the Rules and Evaluation sections carefully. Keep your solution reproducible and retain the source code used to generate your final predictions.

Follow the official WhatsApp group and Kaggle announcements for instructions and updates.

Explore. Validate. Improve. See you on the leaderboard!

---

## Detailed Description

### The Problem
Real-world classification requires more than fitting a model to clean data. Missing measurements, noisy features and complex relationships can make reliable predictions challenging.

Your task is to predict one of seven target classes for each observation using numerical features. The dataset contains deliberately introduced noise and missing values, challenging you to build a model that generalizes beyond the training data.

### Understanding the Data
Each observation contains:
- `id`: A unique identifier used to match observations with predictions.
- `f1`–`f174`: Numerical input features.
- `target`: The class label, ranging from 1 to 7, provided only in the training dataset.

The test dataset contains the same input columns but excludes the target. Missing feature values are intentional and must be handled as part of your solution.

The `id` column is an identifier, not a meaningful predictive measurement.

### Your Approach
Explore the feature distributions, class balance and missing-value patterns before building your solution. Consider how preprocessing, feature selection and model choice affect performance.

Use a local validation strategy to compare approaches. A strong public leaderboard score may not translate into equally strong performance on the private test subset.

You may investigate:
- Missing-value handling and feature scaling.
- Feature selection and feature engineering.
- Methods for handling class imbalance.
- Model selection, hyperparameter tuning and ensembling.

All approaches must comply with the competition rules.

### What Makes a Strong Solution?
A successful solution should perform consistently across all seven classes. Evaluation uses Macro F1, which gives each class equal weight, so strong performance on common classes alone is insufficient.

Aim for a reproducible pipeline that transforms the provided data into reliable predictions. Keep the code, dependencies and configuration needed to reproduce your selected final submissions.

### Expected Output
Submit a CSV containing exactly two columns: `id` and `target`.

Include every test ID exactly once, with an integer prediction from 1 to 7. Follow sample_submission.csv for the required format and save your file without an additional index column.

### Evaluation
Submissions are evaluated using Macro F1 across the seven target classes. The F1 score is calculated separately for each class, then averaged with equal weight. Undefined per-class F1 scores are treated as zero.

Scores range from 0 to 1, with higher scores indicating better performance. Submit predicted class labels, not probabilities.

### Public and Private Leaderboards
The public leaderboard uses 30% of the test observations and provides feedback during the competition.
The private leaderboard uses the remaining 70% and determines the final ranking.
Public leaderboard positions are provisional and may change in the final results. Qualification is subject to eligibility, rule compliance and code verification.

### Submission File
For each `id` in `test.csv`, predict a target class from 1 to 7.

Your submission must:
- Be a CSV file with exactly two columns: `id` and `target`.
- Include every test ID exactly once, without missing or additional IDs.
- Contain integer target labels from 1 to 7, with no missing predictions.
- Include a header and no additional index column.

Example format:
```csv
id,target
1000001,3
1000002,7
1000003,1
```

Use `sample_submission.csv` as your template, preserving its IDs and replacing the placeholder targets with your predictions.

### Submission Limits
Each team may submit 25 files per UTC day. The allowance resets at 12:00 AM UTC / 5:30 AM IST and is shared by all team members.

Select up to two final submissions for private leaderboard evaluation before 4 October 2026 at 7:00 PM IST.
