import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("📈 Stock Analysis Dashboard")

# --- Input ---
ticker = st.text_input("Enter stock ticker (e.g. AAPL, TSLA)", value="AAPL").upper()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
with col2:
    end_date = st.date_input("End date", value=pd.to_datetime("today"))

if start_date > end_date:
    st.error("Error: Start date must be before end date.")
    st.stop()


# --- Download data ---
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    # data = yf.download(ticker, start=start, end=end)
    # data.dropna(inplace=True)
    ticker = yf.Ticker(ticker)
    data = ticker.history(start=start, end=end)
    data.dropna(inplace=True)
    news = ticker.get_news()
    return data, news


data, news = load_data(ticker, start_date, end_date)

if data.empty:
    st.warning("No data found for this ticker/date range.")
    st.stop()

# --- Indicators ---

# Simple Moving Averages (SMA)
data["SMA20"] = data["Close"].rolling(window=20).mean()
data["SMA50"] = data["Close"].rolling(window=50).mean()


# RSI Calculation
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


data["RSI"] = rsi(data["Close"])

# MACD Calculation
ema12 = data["Close"].ewm(span=12, adjust=False).mean()
ema26 = data["Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = ema12 - ema26
data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

# --- Plot Price + SMA ---

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data.index, data["Close"], label="Close Price", color="blue")
ax.plot(data.index, data["SMA20"], label="SMA20", color="orange")
ax.plot(data.index, data["SMA50"], label="SMA50", color="green")
ax.set_title(f"{ticker} Price & Moving Averages")
ax.legend()
st.pyplot(fig)

# --- Plot RSI ---
fig2, ax2 = plt.subplots(figsize=(12, 2))
ax2.plot(data.index, data["RSI"], label="RSI", color="purple")
ax2.axhline(70, color="red", linestyle="--")
ax2.axhline(30, color="green", linestyle="--")
ax2.set_title("RSI Indicator")
st.pyplot(fig2)

# --- Plot MACD ---
fig3, ax3 = plt.subplots(figsize=(12, 3))
ax3.plot(data.index, data["MACD"], label="MACD", color="black")
ax3.plot(data.index, data["Signal"], label="Signal", color="red")
ax3.set_title("MACD Indicator")
ax3.legend()
st.pyplot(fig3)

# --- Metrics and simple Buy/Sell logic ---
latest = data.iloc[-1]

# SMA crossover signal
# Get last valid SMA20 and SMA50 values (not NaN)
last_sma20 = data["SMA20"].dropna().iloc[-1]
last_sma50 = data["SMA50"].dropna().iloc[-1]

sma_signal = "Buy" if last_sma20 > last_sma50 else "Sell"

# RSI signal
last_rsi = data["RSI"].dropna().iloc[-1]

if last_rsi < 30:
    rsi_signal = "Buy (Oversold)"
elif last_rsi > 70:
    rsi_signal = "Sell (Overbought)"
else:
    rsi_signal = "Hold"

# MACD crossover signal
last_macd = data["MACD"].dropna().iloc[-1]
last_signal = data["Signal"].dropna().iloc[-1]

macd_signal = "Buy" if last_macd > last_signal else "Sell"

# Aggregate simple rule:
signals = [
    sma_signal,
    rsi_signal.split()[0],
    macd_signal,
]  # use first word for rsi ("Buy", "Sell", "Hold")
buy_votes = signals.count("Buy")
sell_votes = signals.count("Sell")

if buy_votes > sell_votes:
    final_recommendation = "BUY"
elif sell_votes > buy_votes:
    final_recommendation = "SELL"
else:
    final_recommendation = "HOLD"

# --- Display metrics and report ---
st.markdown("### 📊 Key Metrics & Signals")
st.write(f"- **SMA20:** {last_sma20:.2f}")
st.write(f"- **SMA50:** {last_sma50:.2f}")
st.write(f"- **RSI:** {last_rsi:.2f} ({rsi_signal})")
st.write(f"- **MACD:** {last_macd:.4f}")
st.write(f"- **Signal Line:** {last_signal:.4f}")
st.write(f"- **SMA Crossover Signal:** {sma_signal}")
st.write(f"- **MACD Signal:** {macd_signal}")

st.markdown("---")
st.markdown("### 📝 Summary Report")

summary = f"""
For the stock **{ticker}** between {start_date} and {end_date}:

- The 20-day SMA is at {last_sma20:.2f} and the 50-day SMA is at {last_sma50:.2f}, suggesting a **{sma_signal.lower()}** signal based on moving average crossover.
- The RSI is currently {last_rsi:.2f}, indicating the stock is **{rsi_signal.lower()}**.
- The MACD line is {last_macd:.4f} and the signal line is {last_signal:.4f}, which generates a **{macd_signal.lower()}** signal.

### Final recommendation: **{final_recommendation}**
"""

st.markdown(summary)

st.download_button(
    label="📥 Download Stock Data as CSV",
    data=data.to_csv().encode("utf-8"),
    file_name=f"{ticker}_{start_date}_{end_date}.csv",
    mime="text/csv",
)

for new in news:
    content = new["content"]
    thumbnail = content["thumbnail"]
    st.text(content["title"])
    st.write(thumbnail["originalUrl"])
    st.write(content["summary"])
    st.markdown("---")
