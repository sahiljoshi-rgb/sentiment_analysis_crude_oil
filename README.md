# Sentiment-Based Crude Oil Strategy (Reddit-Driven Backtest)

This project explores whether Reddit sentiment can be used to inform a simple trading strategy for crude oil.  
It is **experimental**, not predictive — the goal is to test directional intuition, not to prove statistical significance.

---

## 📌 Project Summary

We:
- Scrape Reddit posts from finance and energy-related subreddits
- Use FinBERT to classify post sentiment (positive, negative, neutral)
- Align sentiment with daily crude oil prices (WTI Futures)
- Backtest a toy strategy:  
  - **Long** if sentiment is positive  
  - **Short** if sentiment is negative  
  - **Neutral** if sentiment is zero or missing
- Compare against a Buy & Hold benchmark
- Export results to Excel and visualize performance

> ⚠️ This is a prototype with limited sample size (~116 trading days).  
> No claims of predictive power or investment advice are made.

---

## 🚀 Quickstart

1. Clone the repo:

```bash
git clone https://github.com/your-username/sentiment-crude-oil.git
cd sentiment-crude-oil
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook notebooks/sentiment_strategy.ipynb
```

---

## 📦 Dependencies

- `pandas`, `numpy`, `matplotlib`, `plotly`
- `transformers`, `torch`
- `praw` (Reddit API)
- `statsmodels`, `tqdm`

---

## 📜 License

MIT - feel free to fork, adapt, and build on it.

---

## 🙋‍♂️ Author

**Sahil** - MSc Financial Engineering  

Let me know if you want to include a sample plot, add a badge (e.g., MIT license, Python version), or link to a Colab version.
