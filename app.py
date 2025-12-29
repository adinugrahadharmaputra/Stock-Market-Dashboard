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
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from plotly.subplots import make_subplots



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
    
def format_percent(value):
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"
    
def get_company_logo(website_url):
    """
    Try to fetch a company's logo from its website.
    Returns a URL to the image or None.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(website_url, headers=headers, timeout=5)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 1. Try OpenGraph logo
        og_logo = soup.find("meta", property="og:image")
        if og_logo and og_logo.get("content"):
            return og_logo["content"]

        # 2. Try favicon
        icon = soup.find("link", rel=lambda x: x and "icon" in x.lower())
        if icon and icon.get("href"):
            href = icon["href"]
            return href if href.startswith("http") else urljoin(website_url, href)

        # 3. Try common logo filenames
        for logo_file in ["logo.png", "logo.svg", "logo.jpg"]:
            test_url = urljoin(website_url, logo_file)
            r = requests.head(test_url, timeout=3)
            if r.status_code == 200:
                return test_url

    except:
        return None

    return None

def generate_placeholder_logo(name, size=150):
    """
    Generate a simple placeholder logo with initials of the company name.
    """
    initials = "".join([word[0] for word in name.split()[:2]]).upper()
    img = Image.new("RGB", (size, size), color="#3498db")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size // 2)
    except:
        font = ImageFont.load_default()
    w, h = draw.textsize(initials, font=font)
    draw.text(((size - w) / 2, (size - h) / 2), initials, fill="white", font=font)
    return img

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
# Sidebar Input
# =========================================================
st.sidebar.header("🔍 Stock Lookup")
ticker = st.sidebar.text_input("Enter ticker symbol", "AAPL")
symbol = ticker

# =========================================================
# Company Information
# =========================================================
information = fetch_stock_info(symbol)

# Try to get website from yfinance info
website = information.get("website")
logo_url = None

if website:
    # Try Clearbit logo service
    domain = website.split("//")[-1].split("/")[0]
    logo_url = f"https://logo.clearbit.com/{domain}"

    # Check if logo exists, otherwise fallback to scraping
    try:
        r = requests.head(logo_url, timeout=3)
        if r.status_code != 200:
            logo_url = get_company_logo(website)
    except:
        logo_url = get_company_logo(website)

# Layout: logo on the left, info on the right
col1, col2 = st.columns([1, 3])

with col1:
    if logo_url:
        try:
            r = requests.get(logo_url, timeout=3)
            if r.status_code == 200:
                st.image(r.content, width=100)
        except:
            pass  # silently skip if logo cannot be fetched

with col2:
    st.markdown(f"### 🏢 {information.get('longName', 'N/A')}")
    st.markdown(f"**📊 Sector:** {information.get('sector', 'N/A')}")
    
    market_cap = information.get("marketCap")
    if market_cap:
        st.metric(label="Market Capitalization", value=f"${market_cap:,}")
    
    # Optional: display website
    if website:
        st.markdown(f"**🌐 Website:** [{website}]({website})")


# =========================================================
# Chart controls
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
price_data, fundamentals ,price_forecasting, news=st.tabs(['Price History', 'Fundamentals', 'Price Forecast', 'Top 10 News'])


with price_data:
    # =========================================================
    # Price Data & Indicators
    # =========================================================
    with st.expander("📈 Price Data & Technical Indicators", expanded=True):
        col1, col2 = st.columns([2, 3])

        # Period & Interval controls
        with col1:
            period = st.selectbox(
                "Period",
                ['1d', '5d', '1mo', '3mo', '6mo', 'ytd', '1y', '3y', '5y', '10y', 'max'],
                index=6
            )

            interval_options = {
                "1d": ['1m', '2m', '5m', '15m', '30m', '1h'],
                "5d": ['1m', '2m', '5m', '15m', '30m', '1h', '1d'],
                "1mo": ['1h', '1d', '5d', '1wk'],
                "3mo": ['1h', '1d', '5d', '1wk'],
                "6mo": ['1h', '1d', '5d', '1wk'],
                "1y": ['1d', '5d', '1wk', '1mo'],
                "3y": ['1d', '5d', '1wk', '1mo'],
                "5y": ['1d', '5d', '1wk', '1mo'],
                "10y": ['1d', '5d', '1wk', '1mo'],
                "ytd": ['1d', '5d', '1wk', '1mo'],
                "max": ['1d', '5d', '1wk', '1mo'],
            }

            default_interval = interval_options.get(period, ['1d'])[0]
            interval = st.selectbox("Interval", interval_options.get(period, ['1d']), index=0)

        # Indicators selection
        with col2:
            indicator = st.multiselect(
                "Indicators",
                ["MA", "MA Cross", "Bollinger Bands", "Volume"],
                default=[]
            )

            use_ma = "MA" in indicator
            use_cross_ma = "MA Cross" in indicator
            use_bb = "Bollinger Bands" in indicator
            use_vol = "Volume" in indicator

            ma_window, bb_window, bb_std, cross_ma = None, None, None, None

            if use_ma:
                ma_window = st.number_input("MA Window", 5, 200, 20, 5)
            if use_bb:
                bb_window = st.number_input("BB Window", 5, 200, 20, 5)
                bb_std = st.number_input("BB Std Dev", 1.0, 4.0, 2.0, 0.1)
            if use_cross_ma:
                fast = st.number_input("Fast MA", 5, 50, 10)
                slow = st.number_input("Slow MA", 20, 200, 50)
                cross_ma = (fast, slow)

        # Fetch price data
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

        # Determine if volume subplot is needed
        if use_vol:
            rows = 2
            row_heights = [0.7, 0.3]
        else:
            rows = 1
            row_heights = [1]

        # Create subplots
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=row_heights
        )

        # -----------------------
        # Candlestick
        # -----------------------
        fig.add_trace(
            go.Candlestick(
                x=price_history["Date"],
                open=price_history["Open"],
                high=price_history["High"],
                low=price_history["Low"],
                close=price_history["Close"],
                name="Price"
            ),
            row=1, col=1
        )

        # -----------------------
        # Indicators
        # -----------------------
        if use_ma:
            fig.add_trace(go.Scatter(x=price_history["Date"], y=price_history["MA"], 
                                    name="MA", line=dict(color="orange")), row=1, col=1)
        if use_cross_ma:
            fig.add_trace(go.Scatter(x=price_history["Date"], y=price_history["MA_Fast"], 
                                    name="Fast MA", line=dict(color="green")), row=1, col=1)
            fig.add_trace(go.Scatter(x=price_history["Date"], y=price_history["MA_Slow"], 
                                    name="Slow MA", line=dict(color="red")), row=1, col=1)
        if use_bb:
            fig.add_trace(go.Scatter(x=price_history["Date"], y=price_history["BB_Upper"], 
                                    name="BB Upper", line=dict(color="purple", dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=price_history["Date"], y=price_history["BB_Lower"], 
                                    name="BB Lower", line=dict(color="purple", dash="dot")), row=1, col=1)

        # -----------------------
        # Volume subplot
        # -----------------------
        if use_vol:
            fig.add_trace(
                go.Bar(
                    x=price_history["Date"], 
                    y=price_history["Volume"], 
                    name="Volume", 
                    marker_color="green", 
                    opacity=1
                ),
                row=2, col=1
            )

        # -----------------------
        # Layout
        # -----------------------
        fig.update_layout(
            dragmode="pan",
            template="plotly_dark",
            height=1000,
            margin=dict(l=20, r=20, t=30, b=30),
            xaxis_rangeslider=dict(
                visible=False  # make the slider smaller
            )
        )

        # -----------------------
        # Y-axis ranges
        # -----------------------
        y_max_price = price_history["High"].max() * 1.05
        y_min_price = price_history["Low"].min() * 0.95
        fig.update_yaxes(range=[y_min_price, y_max_price], row=1, col=1)

        if use_vol:
            y_max_vol = price_history["Volume"].max() * 1.05
            y_min_vol = 0
            fig.update_yaxes(range=[y_min_vol, y_max_vol], row=2, col=1)


       # -----------------------
        # Add separate sliders for independent zoom
        # -----------------------
        sliders = []

        # # Price zoom slider (above volume slider, at bottom of price chart)
        # sliders.append({
        #     'active': 0,
        #     'y': 0.4,  # slightly above volume subplot
        #     'x': 0,
        #     'xanchor': 'left',
        #     'currentvalue': {"prefix": "Price Zoom: "},
        #     'pad': {"b": 10},
        #     'steps': [
        #         {
        #             'label': f'{i}%',
        #             'method': 'relayout',
        #             'args': [
        #                 'yaxis.range',
        #                 [y_min_price, y_min_price + (y_max_price - y_min_price) * i/100]
        #             ]
        #         }
        #         for i in range(0, 101, 10)
        #     ]
        # })

        # Volume zoom slider (below volume subplot, near bottom)
        if use_vol:
            sliders.append({
                'active': 0,
                'y': 0.00,  # near bottom of figure
                'x': 0,
                'xanchor': 'left',
                'currentvalue': {"prefix": "Volume Zoom: "},
                'pad': {"b": 10},
                'steps': [
                    {
                        'label': f'{i}%',
                        'method': 'relayout',
                        'args': [
                            'yaxis2.range',
                            [y_min_vol, y_min_vol + (y_max_vol - y_min_vol) * i/100]
                        ]
                    }
                    for i in range(0, 101, 10)
                ]
            })

        fig.update_layout(sliders=sliders)


        st.plotly_chart(fig, use_container_width=True)

    # =========================================================
    # Financials
    # =========================================================
    with st.expander("💰 Financials", expanded=True):
        quarterly_financials = fetch_quarterly_financials(symbol)
        annual_financials = fetch_anual_financials(symbol)

        selection = st.radio("Period", ["Quarterly", "Annual"], horizontal=True)

        if selection == "Quarterly":
            df = quarterly_financials.rename_axis("Quarter").reset_index()
            df["Quarter"] = df["Quarter"].astype(str)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Total Revenue")
                st.altair_chart(
                    alt.Chart(df).mark_bar(color="#2ecc71").encode(x="Quarter:O", y="Total Revenue"),
                    use_container_width=True
                )
            with col2:
                st.subheader("Net Income")
                st.altair_chart(
                    alt.Chart(df).mark_bar(color="#3498db").encode(x="Quarter:O", y="Net Income"),
                    use_container_width=True
                )
        else:
            df = annual_financials.rename_axis("Year").reset_index()
            df["Year"] = df["Year"].astype(str).str.split("-").str[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Total Revenue")
                st.altair_chart(
                    alt.Chart(df).mark_bar(color="#2ecc71").encode(x="Year:O", y="Total Revenue"),
                    use_container_width=True
                )
            with col2:
                st.subheader("Net Income")
                st.altair_chart(
                    alt.Chart(df).mark_bar(color="#3498db").encode(x="Year:O", y="Net Income"),
                    use_container_width=True
                )


with fundamentals:
    with st.expander("📊 Fundamentals", expanded=True):
        # Fetch fundamentals data
        info = fetch_stock_info(symbol)  # your existing function
        
        if not info:
            st.warning("No fundamental data available for this ticker.")
            st.stop()
        
        # Company Header
        st.markdown(f"## 🏢 {info.get('longName', 'N/A')}")
        if info.get("logo_url"):
            st.image(info["logo_url"], width=120)
        
        # =========================
        # Key Metrics
        # =========================
        st.subheader("📌 Key Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Market Cap", f"${info.get('marketCap', 0):,}")
            st.metric("Enterprise Value", f"${info.get('enterpriseValue', 'N/A'):,}")
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")
        
        with col2:
            st.metric("EPS (TTM)", f"{info.get('trailingEps', 'N/A')}")
            st.metric("Forward EPS", f"{info.get('forwardEps', 'N/A')}")
            st.metric("PEG Ratio", f"{info.get('pegRatio', 'N/A')}")
            st.metric("Price to Sales (P/S)", f"{info.get('priceToSalesTrailing12Months', 'N/A'):.2f}")
        
        with col3:
            st.metric("Price to Book (P/B)", f"{info.get('priceToBook', 'N/A')}")
            st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A')}")
            st.metric("Beta", f"{info.get('beta', 'N/A')}")
            st.metric("52-Week Range", f"${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")
        
        # =========================
        # Financial Overview
        # =========================
        st.subheader("💰 Financial Overview")
        
        fin_cols = st.columns(2)
        
        balance_sheet = info.get("balanceSheetHistoryQuarterly", {}).get("balanceSheetStatements", [])
        income_stmt = info.get("incomeStatementHistoryQuarterly", {}).get("incomeStatementHistory", [])
        cash_flow = info.get("cashflowStatementHistoryQuarterly", {}).get("cashflowStatements", [])
        
        if balance_sheet:
            with fin_cols[0]:
                st.write("**Balance Sheet (Latest Quarter)**")
                df_bs = pd.DataFrame(balance_sheet[0])
                st.dataframe(df_bs)
        
        if income_stmt:
            with fin_cols[1]:
                st.write("**Income Statement (Latest Quarter)**")
                df_is = pd.DataFrame(income_stmt[0])
                st.dataframe(df_is)
        
        # Cash Flow separately
        if cash_flow:
            st.write("**Cash Flow Statement (Latest Quarter)**")
            df_cf = pd.DataFrame(cash_flow[0])
            st.dataframe(df_cf)
        
        # =========================
        # Ratios
        # =========================
        st.subheader("📊 Financial Ratios")
        ratios_cols = st.columns(3)
        
        with ratios_cols[0]:
            st.metric("Current Ratio", info.get("currentRatio", "N/A"))
            st.metric("Quick Ratio", info.get("quickRatio", "N/A"))
            st.metric("Debt to Equity", info.get("debtToEquity", "N/A"))
        
        with ratios_cols[1]:
            st.metric("Gross Margin", format_percent(info.get("grossMargins")))
            st.metric("Operating Margin", format_percent(info.get("operatingMargins")))
            st.metric("Net Margin", format_percent(info.get("profitMargins")))
        
        with ratios_cols[2]:
            st.metric("Return on Assets (ROA)", format_percent(info.get("returnOnAssets")))
            st.metric("Return on Equity (ROE)", format_percent(info.get("returnOnEquity")))
            st.metric("Return on Investment (ROI)", format_percent(info.get("returnOnInvestment")))
        # =========================
        # Company Overview
        # =========================
        st.subheader("📝 Company Overview")
        description = info.get("longBusinessSummary", "No description available.")
        st.write(description)



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
