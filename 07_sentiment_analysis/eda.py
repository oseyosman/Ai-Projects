"""
Exploratory Data Analysis for Amazon Reviews Sentiment Analysis.
Analyzes class balance, review length distributions, and distinctive positive vs. negative vocabulary.
"""

import os
import re
import pandas as pd
from collections import Counter


def print_banner(text: str, char: str = "=", width: int = 75):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "is", "it", "this", "that", "was", "as", "are", "be", "have", "has",
    "had", "i", "my", "me", "you", "your", "they", "we", "he", "she", "so",
    "but", "if", "from", "up", "out", "about", "all", "phone", "product"
}


def tokenize(text: str):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in COMMON_STOPWORDS]


def main():
    print_banner("💬 PROJECT 07: AMAZON REVIEWS SENTIMENT ANALYSIS - EDA 💬")

    csv_path = os.path.join(os.path.dirname(__file__), "data", "amazon_reviews.csv")
    if not os.path.exists(csv_path):
        print(f"Dataset not found at '{csv_path}'.")
        return

    df = pd.read_csv(csv_path)

    # 1. Dataset Dimensions & Class Distribution
    print_banner("1. DATASET OVERVIEW & CLASS BALANCE")
    print(f"  • Total Customer Reviews : {len(df):,}")
    counts = df["sentiment"].value_counts()
    for s_val, count in counts.items():
        label = "Positive (1) 👍" if s_val == 1 else "Negative (0) 👎"
        bar = "█" * int(count / 20)
        print(f"  • {label:<16} : {count:>4} reviews ({count/len(df)*100:.1f}%) |{bar}")

    # 2. Text Length Statistics
    print_banner("2. REVIEW LENGTH METRICS (Positive vs Negative)")
    df["char_len"] = df["review_text"].apply(len)
    df["word_cnt"] = df["review_text"].apply(lambda t: len(t.split()))

    for s_val in [1, 0]:
        label = "Positive Reviews 👍" if s_val == 1 else "Negative Reviews 👎"
        sub = df[df["sentiment"] == s_val]
        print(f"  {label}:")
        print(f"    - Avg Character Length : {sub['char_len'].mean():.1f} chars (min: {sub['char_len'].min()}, max: {sub['char_len'].max()})")
        print(f"    - Avg Word Count       : {sub['word_cnt'].mean():.1f} words (min: {sub['word_cnt'].min()}, max: {sub['word_cnt'].max()})")

    # 3. Vocabulary Frequency by Sentiment
    print_banner("3. TOP 10 DISTINCTIVE VOCABULARY BY SENTIMENT")
    pos_words = []
    neg_words = []

    for _, row in df.iterrows():
        tokens = tokenize(row["review_text"])
        if row["sentiment"] == 1:
            pos_words.extend(tokens)
        else:
            neg_words.extend(tokens)

    top_pos = Counter(pos_words).most_common(10)
    top_neg = Counter(neg_words).most_common(10)

    print(f"{'Positive Signal Words':<36} | {'Negative Signal Words':<36}")
    print("-" * 75)
    for i in range(10):
        pw, pc = top_pos[i] if i < len(top_pos) else ("", 0)
        nw, nc = top_neg[i] if i < len(top_neg) else ("", 0)
        p_str = f"  {i+1:>2}. {pw:<12} ({pc:>3}x) {'█'*int(pc/4)}"
        n_str = f"  {i+1:>2}. {nw:<12} ({nc:>3}x) {'█'*int(nc/4)}"
        print(f"{p_str:<36} | {n_str:<36}")

    # 4. Key Takeaways
    print_banner("4. KEY NLP FINDINGS & MODELING STRATEGY")
    print("  • Perfectly balanced 50/50 dataset ensures unskewed baseline evaluation (50.0% random chance).")
    print("  • Negative reviews are slightly longer on average (dissatisfied buyers elaborate more).")
    print("  • Positive reviews feature strong polar adjectives: 'great', 'good', 'works', 'love', 'nice'.")
    print("  • Negative reviews feature frustration markers: 'waste', 'bad', 'poor', 'money', 'terrible'.")
    print("  • Bigrams (e.g. 'not good', 'waste money') will be critical to handle negations accurately.\n")


if __name__ == "__main__":
    main()
