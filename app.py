import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
import plotly.graph_objects as go
from stocknews import StockNews
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from prophet import Prophet
import numpy as np
from sklearn.metrics import root_mean_squared_error


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide"
)


# =========================================================
# Cached data fetchers
# =========================================================
@st.cache_data
def fetch_stock_info(symbol):
    return yf.Ticker(symbol).info


@st.cache_data
def fetch_quarterly_financials(symbol):
    return yf.Ticker(symbol).quarterly_financials.T


@st.cache_data
def fetch_anual_financials(symbol):
    return yf.Ticker(symbol).financials.T


def fetch_price_history(
    symbol,
    period,
    interval,
    ma_window=None,
    cross_ma=None,
    bb_window=None,
    bb_std=2,
    add_volume=False
):
    df = yf.Ticker(symbol).history(period=period, interval=interval)

    # Moving Average
    if ma_window:
        df["MA"] = df["Close"].rolling(ma_window, min_periods=1).mean()

    # MA Cross
    if cross_ma:
        fast, slow = cross_ma
        df["MA_Fast"] = df["Close"].rolling(fast, min_periods=1).mean()
        df["MA_Slow"] = df["Close"].rolling(slow, min_periods=1).mean()

    # Bollinger Bands
    if bb_window:
        ma = df["Close"].rolling(bb_window, min_periods=1).mean()
        std = df["Close"].rolling(bb_window, min_periods=1).std()
        df["BB_Upper"] = ma + bb_std * std
        df["BB_Lower"] = ma - bb_std * std

    # Volume
    if not add_volume:
        df.drop(columns=["Volume"], inplace=True, errors="ignore")

    return df


@st.cache_data(ttl=600)
def fetch_news(ticker):
    return StockNews(ticker, save_news=False).read_rss()


# =========================================================
# Behavior Helpers
# =========================================================

def sentiment_emoji(value):
    if value > 0:
        return "🟢"
    elif value < 0:
        return "🔴"
    else:
        return "⚪"

def sentiment_label(value):
    if value > 0:
        return "Positive"
    elif value < 0:
        return "Negative"
    else:
        return "Neutral"
    


# =============================
# Metrics
# =============================

def rmse(y_true, y_pred):
    return np.sqrt(root_mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# =============================
# Walk Forward Backtesting
# =============================

def walk_forward_backtest_lr(df, horizon=5, step=5, start_size=100):
    rmses, mapes = [], []

    for i in range(start_size, len(df) - horizon, step):
        train = df.iloc[:i]
        test = df.iloc[i:i + horizon]

        X_train = train[["t"]]
        y_train = train["Close"]
        X_test = test[["t"]]
        y_test = test["Close"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmses.append(rmse(y_test, preds))
        mapes.append(mape(y_test, preds))

    return {
        "RMSE": np.mean(rmses),
        "MAPE": np.mean(mapes)
    }


def walk_forward_backtest_knn(df, k=5, horizon=5, step=5, start_size=100):
    rmses, mapes = [], []

    for i in range(start_size, len(df) - horizon, step):
        train = df.iloc[:i]
        test = df.iloc[i:i + horizon]

        X_train = train[["t"]]
        y_train = train["Close"]
        X_test = test[["t"]]
        y_test = test["Close"]

        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmses.append(rmse(y_test, preds))
        mapes.append(mape(y_test, preds))

    return {
        "RMSE": np.mean(rmses),
        "MAPE": np.mean(mapes)
    }


def walk_forward_backtest_prophet(df, horizon=5, step=5, start_size=100):
    rmses, mapes = [], []

    prophet_df = df[["Date", "Close"]].copy()
    prophet_df["Date"] = pd.to_datetime(prophet_df["Date"]).dt.tz_localize(None)
    prophet_df = prophet_df.rename(columns={"Date": "ds", "Close": "y"})

    for i in range(start_size, len(prophet_df) - horizon, step):
        train = prophet_df.iloc[:i]
        test = prophet_df.iloc[i:i + horizon]

        model = Prophet(daily_seasonality=False)
        model.fit(train)

        future = model.make_future_dataframe(periods=horizon, freq="B")
        forecast = model.predict(future)

        preds = forecast.tail(horizon)["yhat"].values
        actual = test["y"].values

        rmses.append(rmse(actual, preds))
        mapes.append(mape(actual, preds))

    return {
        "RMSE": np.mean(rmses),
        "MAPE": np.mean(mapes)
    }



# =========================================================
# Sidebar
# =========================================================
ticker = st.sidebar.text_input("Ticker", "AAPL")
symbol = ticker


# =========================================================
# Company Information
# =========================================================
information = fetch_stock_info(symbol)

st.header("Company Information")
st.subheader(f"Name: {information['longName']}")
st.subheader(f"Market Cap: ${information['marketCap']:,}")
st.subheader(f"Sector: {information['sector']}")


# =========================================================
# Chart controls — state
# =========================================================
if "hp" not in st.session_state:
    st.session_state.hp = "ytd"

if "prev_hp" not in st.session_state:
    st.session_state.prev_hp = st.session_state.hp

if "tf" not in st.session_state:
    st.session_state.tf = None


# =========================================================
# Period selector
# =========================================================
st.header("Data")
price_data, price_forecasting, news=st.tabs(['Price History', 'Price Forecast', 'Top 10 News'])

with price_data:
    period = st.segmented_control(
        "Period",
        ['1d', '5d', '1mo', '3mo', '6mo', 'ytd', '1y', '3y', '5y', '10y', 'max'],
        key="hp"
    )

    # Reset interval when period changes
    if st.session_state.hp != st.session_state.prev_hp:
        st.session_state.tf = None
        st.session_state.prev_hp = st.session_state.hp


    # =========================================================
    # Interval options
    # =========================================================
    if period == "1d":
        default_tf = "5m"
        options = ['1m', '2m', '5m', '15m', '30m', '1h']

    elif period == "5d":
        default_tf = "1h"
        options = ['1m', '2m', '5m', '15m', '30m', '1h', '1d']

    elif period in ["1mo", "3mo", "6mo"]:
        default_tf = "1d"
        options = ['5m', '15m', '30m', '1h', '1d', '5d', '1wk']

    else:
        default_tf = "1wk"
        options = ['1h', '1d', '5d', '1wk', '1mo']

    if st.session_state.tf is None or st.session_state.tf not in options:
        st.session_state.tf = default_tf

    interval = st.segmented_control("Interval", options, key="tf")


    # =========================================================
    # Indicators
    # =========================================================
    indicator = st.multiselect(
        "Indicators",
        ["MA", "MA Cross", "Bollinger Bands", "Volume"],
        default=[]
    )

    use_ma = "MA" in indicator
    use_cross_ma = "MA Cross" in indicator
    use_bb = "Bollinger Bands" in indicator
    use_vol = "Volume" in indicator

    ma_window = None
    bb_window = None
    bb_std = None
    cross_ma = None

    if use_ma:
        ma_window = st.number_input("MA Window", 5, 200, 20, 5)

    if use_bb:
        bb_window = st.number_input("BB Window", 5, 200, 20, 5)
        bb_std = st.number_input("BB Std Dev", 1.0, 4.0, 2.0, 0.1)

    if use_cross_ma:
        fast = st.number_input("Fast MA", 5, 50, 10)
        slow = st.number_input("Slow MA", 20, 200, 50)
        cross_ma = (fast, slow)


    # =========================================================
    # Fetch price data
    # =========================================================
    price_history = fetch_price_history(
        symbol,
        period,
        interval,
        ma_window,
        cross_ma,
        bb_window,
        bb_std,
        use_vol
    ).rename_axis("Date").reset_index()


    # =========================================================
    # Plotly chart
    # =========================================================
    fig = go.Figure()

    fig.add_candlestick(
        x=price_history["Date"],
        open=price_history["Open"],
        high=price_history["High"],
        low=price_history["Low"],
        close=price_history["Close"],
        name="Price"
    )

    if use_ma:
        fig.add_scatter(x=price_history["Date"], y=price_history["MA"], name="MA")

    if use_cross_ma:
        fig.add_scatter(x=price_history["Date"], y=price_history["MA_Fast"], name="Fast MA")
        fig.add_scatter(x=price_history["Date"], y=price_history["MA_Slow"], name="Slow MA")

    if use_bb:
        fig.add_scatter(x=price_history["Date"], y=price_history["BB_Upper"], name="BB Upper", line=dict(dash="dot"))
        fig.add_scatter(x=price_history["Date"], y=price_history["BB_Lower"], name="BB Lower", line=dict(dash="dot"))

    if use_vol:
        fig.add_bar(
            x=price_history["Date"],
            y=price_history["Volume"],
            name="Volume",
            yaxis="y2",
            opacity=0.3
        )

    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)


    # =========================================================
    # Financials
    # =========================================================
    quarterly_financials = fetch_quarterly_financials(symbol)
    annual_financials = fetch_anual_financials(symbol)

    selection = st.segmented_control(
        "Period",
        ["Quarterly", "Annual"],
        default="Quarterly"
    )

    if selection == "Quarterly":
        df = quarterly_financials.rename_axis("Quarter").reset_index()
        df["Quarter"] = df["Quarter"].astype(str)

        st.altair_chart(
            alt.Chart(df).mark_bar(color="green").encode(x="Quarter:O", y="Total Revenue"),
            use_container_width=True
        )

        st.altair_chart(
            alt.Chart(df).mark_bar(color="blue").encode(x="Quarter:O", y="Net Income"),
            use_container_width=True
        )

    else:
        df = annual_financials.rename_axis("Year").reset_index()
        df["Year"] = df["Year"].astype(str).str.split("-").str[0]

        st.altair_chart(
            alt.Chart(df).mark_bar(color="green").encode(x="Year:O", y="Total Revenue"),
            use_container_width=True
        )

        st.altair_chart(
            alt.Chart(df).mark_bar(color="blue").encode(x="Year:O", y="Net Income"),
            use_container_width=True
        )

with price_forecasting:

    price_history = (
        fetch_price_history(
            symbol,
            period,
            interval,
            ma_window=None,
            cross_ma=None,
            bb_window=None,
            bb_std=None,
            add_volume=False
        )
        .rename_axis("Date")
        .reset_index()
    )

    st.header("📈 Price Forecasting")

    if price_history.empty:
        st.warning("No price data available.")
        st.stop()

    # =========================
    # SETTINGS
    # =========================
    forecast_days = st.slider("Forecast horizon (days)", 5, 60, 14)

    methods = st.multiselect(
        "Forecast methods",
        ["Linear Regression", "KNN Regression", "Prophet", "Monte Carlo"],
        default=["Linear Regression"]
    )

    knn_k = st.slider("KNN: Number of neighbors (k)", 2, 20, 5)
    mc_sims = st.slider("Monte Carlo simulations", 50, 1000, 200)

    # =========================
    # PREPARE DATA
    # =========================
    df = price_history[["Date", "Open", "High", "Low", "Close"]].dropna().copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["t"] = np.arange(len(df))

    # =========================
    # FUTURE INDEX (GLOBAL)
    # =========================
    future_t = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)

    future_dates = pd.date_range(
        start=df["Date"].iloc[-1],
        periods=forecast_days + 1,
        freq="B"
    )[1:]

    # =========================
    # CANDLESTICK
    # =========================
    candle = (
        alt.Chart(df)
        .mark_rule()
        .encode(x="Date:T", y="Low:Q", y2="High:Q")
    ) + (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="Date:T",
            y="Open:Q",
            y2="Close:Q",
            color=alt.condition(
                "datum.Open <= datum.Close",
                alt.value("#2ecc71"),
                alt.value("#e74c3c")
            )
        )
    )

    forecast_lines = []

    # =========================
    # LINEAR REGRESSION
    # =========================
    if "Linear Regression" in methods:
        lr = LinearRegression()
        lr.fit(df[["t"]], df["Close"])
        preds = lr.predict(future_t)

        forecast_lines.append(pd.DataFrame({
            "Date": future_dates,
            "Price": preds,
            "Model": "Linear Regression"
        }))

    # =========================
    # KNN REGRESSION
    # =========================
    if "KNN Regression" in methods:
        knn = KNeighborsRegressor(n_neighbors=knn_k)
        knn.fit(df[["t"]], df["Close"])
        preds = knn.predict(future_t)

        forecast_lines.append(pd.DataFrame({
            "Date": future_dates,
            "Price": preds,
            "Model": "KNN"
        }))

    # =========================
    # PROPHET
    # =========================
    if "Prophet" in methods:
        prophet_df = df[["Date", "Close"]].rename(
            columns={"Date": "ds", "Close": "y"}
        )

        m = Prophet(daily_seasonality=False)
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=forecast_days, freq="B")
        forecast = m.predict(future)

        pf = forecast.tail(forecast_days)[["ds", "yhat"]]
        pf.columns = ["Date", "Price"]
        pf["Model"] = "Prophet"

        forecast_lines.append(pf)

    # =========================
    # MONTE CARLO (GBM) WITH PROBABILITY BANDS
    # =========================
    if "Monte Carlo" in methods:
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        mu, sigma = returns.mean(), returns.std()
        last_price = df["Close"].iloc[-1]

        simulations = []
        for _ in range(mc_sims):
            prices = [last_price]
            for _ in range(forecast_days):
                shock = np.random.normal(mu, sigma)
                prices.append(prices[-1] * np.exp(shock))
            simulations.append(prices[1:])  # remove initial price

        sim_df = pd.DataFrame(simulations).T  # shape: forecast_days x mc_sims
        sim_df["Date"] = future_dates

        # Compute percentiles for shaded bands
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        probs = sim_df.drop(columns="Date").apply(
            lambda x: np.percentile(x, percentiles), axis=1
        )
        probs = pd.DataFrame(list(probs), columns=[f"p{p}" for p in percentiles])
        probs["Date"] = future_dates

        # Median line
        median_forecast = pd.DataFrame({
            "Date": future_dates,
            "Price": probs["p50"],
            "Model": "Monte Carlo Median"
        })
        forecast_lines.append(median_forecast)

        # Shaded probability bands
        shade_bands_specs = [
            {"lower": "p25", "upper": "p75", "color": "#3498db", "label": "25-75%"},
            {"lower": "p10", "upper": "p90", "color": "#2980b9", "label": "10-90%"},
            {"lower": "p5", "upper": "p95", "color": "#1c5980", "label": "5-95%"},
        ]

        for band in shade_bands_specs:
            band_df = pd.DataFrame({
                "Date": future_dates,
                "y_lower": probs[band["lower"]],
                "y_upper": probs[band["upper"]],
                "label": band["label"],
                "color": band["color"]
            })
            forecast_lines.append(band_df)



    # =========================
    # PLOT
    # =========================
    if forecast_lines:
        # Separate median lines from shaded bands
        median_lines = [f for f in forecast_lines if "Model" in f.columns]
        shade_bands = [f for f in forecast_lines if "y_lower" in f.columns and "y_upper" in f.columns]

        # Median line for Monte Carlo + other forecasts
        forecast_df = pd.concat(median_lines)
        forecast_chart = (
            alt.Chart(forecast_df)
            .mark_line(opacity=0.8)
            .encode(
                x="Date:T",
                y="Price:Q",
                color="Model:N"
            )
        )

        # Add shaded probability bands for Monte Carlo
        for band in shade_bands:
            forecast_chart += (
                alt.Chart(band)
                .mark_area(opacity=0.2, color=band["color"].iloc[0])
                .encode(
                    x="Date:T",
                    y="y_lower:Q",
                    y2="y_upper:Q"
                )
            )

        # Combine with candlestick
        st.altair_chart(
            (candle + forecast_chart).properties(height=520),
            use_container_width=True
        )
    else:
        st.altair_chart(candle, use_container_width=True)


    # =========================
    # BACKTESTING
    # =========================
    st.subheader("🧪 Walk-Forward Backtesting")

    enable_backtest = st.checkbox("Enable backtesting")
    if enable_backtest:
        backtest_horizon = st.slider("Backtest horizon (days)", 3, 20, 5)
        backtest_step = st.slider("Step size", 3, 20, 5)
        start_size = int(len(df) * 0.6)

        st.markdown("### Backtest Results")

        if "Linear Regression" in methods:
            res = walk_forward_backtest_lr(df, backtest_horizon, backtest_step, start_size)
            col1, col2 = st.columns(2)
            col1.metric("Linear Regression RMSE", f"{res['RMSE']:.2f}")
            col2.metric("Linear Regression MAPE", f"{res['MAPE']:.2f} %")

        if "KNN Regression" in methods:
            res = walk_forward_backtest_knn(df, knn_k, backtest_horizon, backtest_step, start_size)
            col1, col2 = st.columns(2)
            col1.metric("KNN RMSE", f"{res['RMSE']:.2f}")
            col2.metric("KNN MAPE", f"{res['MAPE']:.2f} %")

        if "Prophet" in methods:
            res = walk_forward_backtest_prophet(df, backtest_horizon, backtest_step, start_size)
            col1, col2 = st.columns(2)
            col1.metric("Prophet RMSE", f"{res['RMSE']:.2f}")
            col2.metric("Prophet MAPE", f"{res['MAPE']:.2f} %")

    st.info("⚠️ Forecasts are experimental and not financial advice. RMSE = Root Mean Squared Error, MAPE = Mean Absolute Percentage Error")




with news:
    st.header(f"Recent news of {ticker}")

    df_news = fetch_news(ticker)

    if df_news.empty:
        st.info("No news available.")
    else:
        # ---------- Prepare data ----------
        sentiment_pie_df = df_news.copy()
        sentiment_pie_df["Sentiment"] = sentiment_pie_df["sentiment_summary"].apply(
            lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral"
        )

        sentiment_counts = (
            sentiment_pie_df["Sentiment"]
            .value_counts()
            .reset_index()
        )
        sentiment_counts.columns = ["Sentiment", "Count"]

        max_items = min(10, len(df_news))
        sentiment_df = (
            df_news.head(max_items)[
                ["sentiment_title", "sentiment_summary"]
            ]
            .reset_index(drop=True)
        )
        sentiment_df["News"] = sentiment_df.index + 1

        # ---------- Layout ----------
        col1, col2 = st.columns(2)

        # ---------- PIE CHART ----------
        with col1:
            st.subheader("Overall News Sentiment")

            pie_chart = (
                alt.Chart(sentiment_counts)
                .mark_arc(innerRadius=40)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color(
                        "Sentiment:N",
                        scale=alt.Scale(
                            domain=["Positive", "Neutral", "Negative"],
                            range=["#2ecc71", "#bdc3c7", "#e74c3c"]
                        )
                    ),
                    tooltip=[
                        alt.Tooltip("Sentiment:N"),
                        alt.Tooltip("Count:Q")
                    ]
                )
            )

            st.altair_chart(pie_chart, use_container_width=True)

        # ---------- TOP 10 BAR CHART ----------
        with col2:
            st.subheader("Top 10 News Sentiment")

            chart = (
                alt.Chart(sentiment_df)
                .transform_fold(
                    ["sentiment_title", "sentiment_summary"],
                    as_=["Type", "Sentiment"]
                )
                .mark_bar()
                .encode(
                    x=alt.X("News:O", title="News Index"),
                    y=alt.Y("Sentiment:Q", title="Sentiment Score"),
                    color=alt.Color("Type:N", title="Sentiment Type"),
                    tooltip=[
                        alt.Tooltip("Type:N"),
                        alt.Tooltip("Sentiment:Q", format=".2f")
                    ]
                )
            )

            st.altair_chart(chart, use_container_width=True)

        # ---------- NEWS LIST ----------
        st.subheader("News (Last 10)")
        for i in range(max_items):
            row = df_news.iloc[i]

            emoji = sentiment_emoji(row["sentiment_summary"])
            label = sentiment_label(row["sentiment_summary"])

            with st.expander(f"{emoji} News {i+1}: {row['title']} ({label})"):
                st.caption(f"🗓️ {row['published'].replace(' +0000', '')}")
                st.write(row["summary"])

                col_a, col_b = st.columns(2)
                col_a.write(
                    f"**Title Sentiment:** "
                    f"{sentiment_emoji(row['sentiment_title'])} "
                    f"{sentiment_label(row['sentiment_title'])}"
                )
                col_b.write(
                    f"**Summary Sentiment:** "
                    f"{sentiment_emoji(row['sentiment_summary'])} "
                    f"{sentiment_label(row['sentiment_summary'])}"
                )
