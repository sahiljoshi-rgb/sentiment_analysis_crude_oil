import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import praw
from datetime import datetime
from tqdm import tqdm

def get_finbert_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["negative", "neutral", "positive"]
    sentiment_dict = {labels[i]: scores[0][i].item() for i in range(len(labels))}
    sentiment_dict["compound"] = sentiment_dict["positive"] - sentiment_dict["negative"]
    return sentiment_dict

def main():
    print("\n📁 Sentiment-Based Crude Oil Strategy\n")
    print("This script explores whether Reddit sentiment can inform a simple trading strategy for crude oil.\n")

    # --- User Inputs ---
    csv_path = input("Enter path to Crude Oil CSV file. Make sure the path does not contain " " (from Investing.com): ").strip()
    client_id = input("Enter your Reddit client_id: ").strip()
    client_secret = input("Enter your Reddit client_secret: ").strip()
    user_agent = input("Enter your Reddit user_agent: ").strip()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # --- Load Crude Oil Price Data ---
    df = pd.read_csv(csv_path.strip('"'))[["Date", "Price"]] 
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index, dayfirst=True)
    one_year_ago = df.index.max() - pd.DateOffset(years=1)
    df_last_year = df.loc[df.index >= one_year_ago]

    # --- Plot Price ---
    df["Price"].plot(figsize=(12, 5), title="Crude Oil WTI Futures Price")
    plt.show()
    df_last_year["Price"].plot(figsize=(15, 5), title="Crude Oil WTI Futures Price (Last 1 Year)")
    plt.show()

    # --- Initialize Reddit API ---
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )

    subreddits = ["oilandgas", "energy", "investing", "wallstreetbets", "stocks", "economics", "commodities", "finance"]
    query = "Crude Oil OR Oil OR Crude OR WTI OR Brent OR OPEC OR Energy"

    # --- Scrape Reddit Posts ---
    posts = []
    for sub in tqdm(subreddits, desc="Subreddits"):
        subreddit = reddit.subreddit(sub)
        for submission in tqdm(subreddit.search(query, sort="new", limit=2000), desc=f"Scraping {sub}", leave=False):
            created = datetime.fromtimestamp(submission.created_utc)
            if start_date <= created <= end_date:
                posts.append([
                    created,
                    submission.id,
                    submission.title,
                    submission.selftext,
                    submission.score,
                    submission.num_comments,
                    str(submission.author)
                ])

    df_posts = pd.DataFrame(posts, columns=[
        "Datetime", "Post Id", "Title", "Text", "Score", "Num_Comments", "Author"
    ])
    df_posts["Content"] = df_posts["Title"].fillna('') + " " + df_posts["Text"].fillna('')

    # --- Load FinBERT ---
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    # --- Apply Sentiment ---
    df_posts["Sentiment"] = df_posts["Content"].apply(lambda x: get_finbert_sentiment(x, tokenizer, model)["compound"])
    df_posts["Date"] = df_posts["Datetime"].dt.date
    daily_sentiment = df_posts.groupby("Date")["Sentiment"].mean().reset_index()

    # --- Plot Sentiment Distribution ---
    plt.figure(figsize=(8,5))
    plt.hist(df_posts["Sentiment"], bins=30, color="tab:purple", alpha=0.7)
    plt.title("Distribution of Reddit Post Sentiment (FinBERT)", fontsize=14)
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.show()

    # --- Plot Daily Sentiment ---
    daily_sentiment.set_index("Date")["Sentiment"].plot(
        figsize=(15,5),
        title="Reddit Sentiment on Crude Oil (Last 1 Year, FinBERT)",
        color="tab:blue"
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.show()

    # --- Overlay Price and Sentiment ---
    fig, ax1 = plt.subplots(figsize=(15,5))
    df_last_year["Price"].plot(ax=ax1, color="tab:green", label="Crude Oil Price")
    ax1.set_ylabel("Price", color="tab:green")
    ax2 = ax1.twinx()
    daily_sentiment.set_index("Date")["Sentiment"].plot(ax=ax2, color="tab:blue", label="Sentiment")
    ax2.set_ylabel("Sentiment", color="tab:blue")
    plt.title("Crude Oil Price vs Reddit Sentiment (Last 1 Year)")
    fig.tight_layout()
    plt.show()

    # --- Merge Price and Sentiment ---
    df_last_year_sort = df_last_year.sort_index().copy()
    df_last_year_sort["Return"] = df_last_year_sort["Price"].pct_change()
    daily_sentiment["Date"] = pd.to_datetime(daily_sentiment["Date"])
    merged = pd.merge(
        df_last_year_sort.reset_index(),
        daily_sentiment,
        on="Date",
        how="inner"
    ).set_index("Date")

    # --- Backtest Strategy ---
    merged["Signal"] = merged["Sentiment"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    merged["StrategyReturn"] = merged["Signal"].shift(1) * merged["Return"]
    merged["CumulativeStrategy"] = (1 + merged["StrategyReturn"].fillna(0)).cumprod()
    merged["CumulativeBuyHold"] = (1 + merged["Return"].fillna(0)).cumprod()

    plt.figure(figsize=(12,5))
    merged["CumulativeStrategy"].plot(label="Sentiment Strategy", linewidth=2)
    merged["CumulativeBuyHold"].plot(label="Buy & Hold", linestyle="--")
    plt.title("Sentiment-Based Strategy vs Buy & Hold")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # --- Metrics ---
    strategy_total = merged["CumulativeStrategy"].iloc[-1] - 1
    buyhold_total = merged["CumulativeBuyHold"].iloc[-1] - 1
    strategy_sharpe = merged["StrategyReturn"].mean() / merged["StrategyReturn"].std()
    buyhold_sharpe = merged["Return"].mean() / merged["Return"].std()

    print(f"\nStrategy Total Return: {strategy_total:.2%}")
    print(f"Buy & Hold Total Return: {buyhold_total:.2%}")
    print(f"Strategy Sharpe: {strategy_sharpe:.2f}")
    print(f"Buy & Hold Sharpe: {buyhold_sharpe:.2f}")

    # --- Export to Excel ---
    export_df = merged[[
        "Price", "Return", "Sentiment", "Signal",
        "StrategyReturn", "CumulativeStrategy", "CumulativeBuyHold"
    ]].copy()
    export_df.to_excel("sentiment_strategy_backtest.xlsx", index=True)
    print("\n✅ Exported results to sentiment_strategy_backtest.xlsx")

if __name__ == "__main__":
    main()
