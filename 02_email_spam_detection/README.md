# Project 02: Email & SMS Spam Detection (Binary NLP Classification)

## 🎯 Goal
Classify text messages and emails into **HAM** (legitimate) or **SPAM** (unsolicited / phishing / scam) using Natural Language Processing (NLP) and supervised classification algorithms.

---

## 📊 Dataset
- **SMS & Email Spam Collection**: 5,572 labeled text samples.
- **Classes**:
  - `ham` (~86.6%): Legitimate personal, work, and notification messages.
  - `spam` (~13.4%): Unsolicited promotions, prize giveaways, urgent scams, and phishing attempts.

---

## 🧠 Machine Learning & NLP Concepts Covered

### 1. Text Feature Extraction: TF-IDF
Computers cannot directly process raw strings. We convert text into numerical vectors using **Term Frequency - Inverse Document Frequency (TF-IDF)**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- **TF (Term Frequency)**: How often word $t$ appears in document $d$.
- **IDF (Inverse Document Frequency)**: Diminishes weights of common English words (e.g., *"the"*, *"is"*, *"and"*) and boosts words unique to spam (e.g., *"urgent"*, *"prize"*, *"claim"*, *"free"*).
- Uses **N-Grams** `(1, 2)` to capture combinations like *"free trial"*, *"act now"*, *"claim prize"*.

### 2. Algorithms Benchmarked
- **Multinomial Naive Bayes (`MultinomialNB`)**: Probabilistic model applying Bayes' theorem with feature independence assumption. The historical gold standard for email spam filtering.
- **Complement Naive Bayes (`ComplementNB`)**: Specialized Naive Bayes adaptation designed for severe class imbalance.
- **Logistic Regression**: Linear decision boundary optimized via sigmoid loss with L2 regularization.
- **Linear Support Vector Classifier (`LinearSVC`)**: Maximizes the margin separating ham and spam vectors in high-dimensional text space.
- **Random Forest Classifier**: Ensemble of randomized decision trees on word features.

### 3. Critical Metric: Precision vs. Recall
In spam filtering, **Precision is paramount**:
- **False Positive**: Legitimate email sent to spam folder (e.g. job interview offer or flight ticket missed). **Extremely costly!**
- **False Negative**: Spam message reaches inbox. Mild nuisance.
- Therefore, we prioritize high **Precision** on the Spam class while maintaining strong overall F1 score.

---

## 🚀 How to Run

### 1. Run Exploratory Data Analysis (EDA)
Inspect class distribution, text lengths, capital letter ratios, and top spam keywords:
```bash
python3 eda.py
```

### 2. Train and Evaluate Models
Train 5 NLP classifiers, print the confusion matrix and leaderboard, and save the best pipeline:
```bash
python3 train.py
```

### 3. Interactive Email Spam Classifier
Test with real-world presets (phishing, lottery scam, meeting invite, shipping confirmation) or test custom email text:
```bash
python3 classify.py
```

Or pass a message directly in terminal:
```bash
python3 classify.py --text "Urgent: Your account is suspended. Click here to verify your identity and claim $500 reward."
```
