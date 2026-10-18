"""
Dataset manager and downloader for Email & SMS Spam Detection.
Fetches the official SMS Spam Collection dataset and formats it into clean CSV.
Includes an offline embedded fallback dataset in case network is unavailable.
"""

import os
import urllib.request
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "spam_dataset.csv")

PRIMARY_URL = "https://raw.githubusercontent.com/ShresthaSudip/SMS_Spam_Detection_DNN_LSTM_BiLSTM/master/SMSSpamCollection"


def download_or_get_dataset() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)

    print(f"Downloading SMS/Email Spam dataset from GitHub...")
    try:
        req = urllib.request.Request(
            PRIMARY_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode("utf-8", errors="ignore")

        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                label, text = parts
                label = label.strip().lower()
                if label in ["ham", "spam"]:
                    rows.append({"label": label, "text": text.strip()})

        df = pd.DataFrame(rows)
        df.to_csv(DATA_FILE, index=False)
        print(f"Successfully downloaded and saved {len(df):,} messages to {DATA_FILE}")
        return df

    except Exception as err:
        print(f"Network download failed ({err}). Using built-in starter dataset...")
        # Offline fallback samples
        samples = [
            ("ham", "Hey, are you free for lunch today at 12:30?"),
            ("ham", "Please find attached the slides from today's sprint sync."),
            ("ham", "Can you review the PR when you get a chance? No rush."),
            ("ham", "Mom called to ask if we are visiting next weekend."),
            ("ham", "Your doctor appointment is confirmed for Tuesday at 10 AM."),
            ("ham", "Don't forget to submit your timesheet before Friday 5 PM."),
            ("ham", "The package was left at the front door as requested."),
            ("spam", "WINNER! You have won a $1,000 Walmart gift card! Claim now at http://win-claim.xyz"),
            ("spam", "URGENT! Your banking access has been suspended. Verify immediately at http://secure-login-check.com"),
            ("spam", "Congratulations! Selected customer reward: Claim your FREE iPhone 15 by replying YES."),
            ("spam", "Guaranteed 10x crypto returns in 48 hours! Limited slots available, deposit now."),
            ("spam", "Hot singles in your area want to chat tonight. Click here to view photos."),
            ("spam", "Final notice: IRS tax refund pending. Call 1-800-555-0199 within 24 hours to avoid penalty.")
        ] * 40
        df = pd.DataFrame(samples, columns=["label", "text"])
        df.to_csv(DATA_FILE, index=False)
        return df


def main():
    df = download_or_get_dataset()
    print("\nDataset Summary:")
    print(f"Total Rows: {len(df):,}")
    print(df["label"].value_counts(normalize=True).map("{:.1%}".format))
    print("\nFirst 5 Records:")
    print(df.head(5).to_string())


if __name__ == "__main__":
    main()
