import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# --- PAGE SETUP ---
st.set_page_config(page_title="Autonomous Rotation Scanner", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", ["Scanner Dashboard", "Market Heatmap 📊", "How It Works (Metrics)"])
st.sidebar.divider()

# ==========================================
# THE AUTONOMOUS YAHOO FINANCE ENGINE
# ==========================================
@st.cache_data(ttl=3600) 
def fetch_market_data():
    sheet_url = "https://docs.google.com/spreadsheets/d/1jTpv3fahEDxpgvc1Xg3kJ74ZMUhWh6Goq9kWeFHNqIg/export?format=csv"
    
    try:
        master_list = pd.read_csv(sheet_url)
    except Exception as e:
        raise ValueError(f"Could not read Google Sheet. Ensure the URL ends with /export?format=csv. Error: {e}")
        
    master_list.columns = master_list.columns.str.strip()
    if not all(col in master_list.columns for col in ['Symbol', 'Exchange', 'Sector', 'Industry']):
        raise ValueError("Google Sheet must contain exactly these columns: Symbol, Exchange, Sector, Industry")

    tickers = master_list['Symbol'].dropna().astype(str).str.strip().unique().tolist()
    
    if not tickers:
        raise ValueError("No tickers found in the Google Sheet.")

    raw_data = yf.download(tickers, period="3mo", group_by='ticker', threads=True)
    
    all_stocks = []
    
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                stock_df = raw_data.copy()
            else:
                if ticker not in raw_data.columns.levels[0]: 
                    continue
                stock_df = raw_data[ticker].copy()
                
            stock_df = stock_df.dropna(subset=['Close'])
            if stock_df.empty:
                continue
                
            stock_df = stock_df.reset_index()
            stock_df['Symbol'] = ticker
            
            if 'Date' in stock_df.columns:
                stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
            
            all_stocks.append(stock_df)
        except Exception as e:
            continue
            
    if not all_stocks:
        raise ValueError("Failed to download price data for the provided symbols from Yahoo Finance.")
        
    final_price_df = pd.concat(all_stocks, ignore_index=True)
    merged_df = pd.merge(final_price_df, master_list, on='Symbol', how='left')
    
    return merged_df

# ==========================================
# PAGE 1: THE SCANNER DASHBOARD
# ==========================================
if page == "Scanner Dashboard":
    st.title("🔄 Autonomous Market Rotation Scanner")
    st.markdown("Identify capital flows and find the strongest stocks within leading groups.")

    st.sidebar.header("1. Data Feed")
    st.sidebar.success("🟢 Autonomous Engine Active")

    st.sidebar.header("2. Scan Level")
    grouping_level = st.sidebar.radio("Group stocks by:", ["Sector", "Industry"])

    with st.spinner("Compiling live market data from Yahoo Finance..."):
        try:
            df = fetch_market_data()
            
            has_ohlc = all(c in df.columns for c in ['Open', 'High', 'Low'])
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            df[grouping_level] = df[grouping_level].astype(str).str.strip()
            df['Exchange'] = df['Exchange'].astype(str).str.strip()
            
            available_exchanges = sorted(df[df['Exchange'] != 'nan']['Exchange'].unique().tolist())
            selected_exchanges = st.sidebar.multiselect("Select Exchange(s):", options=available_exchanges, default=available_exchanges)
            if not selected_exchanges:
                st.warning("⚠️ Please select at least one exchange in the sidebar to run the scan.")
                st.stop()
            df = df[df['Exchange'].isin(selected_exchanges)]
            
            df = df.sort_values(by=['Symbol', 'Date'])
            trading_days = len(df['Date'].unique())
            latest_date = df['Date'].max()
            
            st.sidebar.caption(f"Last Price Update: {latest_date.strftime('%Y-%m-%d')}")
            
        except Exception as e:
            st.error(f"Engine Error: {e}")
            st.stop()

        st.sidebar.header("3. Analysis Parameters")
        min_price = st.sidebar.number_input("Minimum Stock Price ($)", min_value=0.0, value=10.0, step=1.0)
        
        latest_closes = df.groupby('Symbol')['Close'].last()
        valid_symbols = latest_closes[latest_closes >= min_price].index
        df = df[df['Symbol'].isin(valid_symbols)]
        
        if df.empty:
            st.error(f"⚠️ No stocks found closing above ${min_price}. Please lower your minimum price.")
            st.stop()
        
        max_periods = trading_days - 1
        long_window = st.sidebar.slider("Long-Term Momentum (Days)", min_value=2, max_value=max_periods, value=min(21, max_periods))
        short_window = st.sidebar.slider("Short-Term Momentum (Days)", min_value=1, max_value=long_window-1, value=min(5, long_window-1))
        vol_window = st.sidebar.slider("Baseline Volume Avg (Days)", min_value=2, max_value=trading_days, value=min(50, trading_days))
        rs_window = st.sidebar.slider("Relative Strength Trend (Days)", min_value=2, max_value=trading_days, value=min(10, trading_days))

        # --- PHASE 1: MACRO ROLL-UP ---
        st.header(f"Phase 1: {grouping_level} Rotation (Capital Flows)")
        group_df = df.groupby(['Date', grouping_level]).agg({'Close': 'mean', 'Volume': 'sum'}).reset_index()
        group_metrics = []
        groups = group_df[grouping_level].unique()
        
        for grp in groups:
            if grp == 'nan' or grp == 'None' or grp == '': continue
            g_data = group_df[group_df[grouping_level] == grp].sort_values('Date').copy()
            g_data['Short_Return'] = g_data['Close'].pct_change(periods=short_window) * 100
            g_data['Long_Return'] = g_data['Close'].pct_change(periods=long_window) * 100
            g_data['Vol_Baseline'] = g_data['Volume'].rolling(window=vol_window).mean()
            
            latest_data = g_data.iloc[-1]
            acceleration = latest_data['Short_Return'] - latest_data['Long_Return']
            vol_surge_ratio = latest_data['Volume'] / latest_data['Vol_Baseline'] if latest_data['Vol_Baseline'] > 0 else 0
            
            group_metrics.append({
                grouping_level: grp,
                f'{short_window}-Day Return (%)': round(latest_data['Short_Return'], 2),
                f'{long_window}-Day Return (%)': round(latest_data['Long_Return'], 2),
                'Momentum Acceleration': round(acceleration, 2),
                f'Volume Surge (Today vs {vol_window}D Avg)': round(vol_surge_ratio, 2)
            })

        group_results = pd.DataFrame(group_metrics).dropna().sort_values(by='Momentum Acceleration', ascending=False)
        st.dataframe(group_results, use_container_width=True)
        st.divider()

        # --- PHASE 2 & 3: STOCK SCREENER ---
        st.header(f"Phase 2: Stock Screener (Breakouts & Accumulation)")
        selected_group = st.selectbox(f"Select a Leading {grouping_level} to Scan:", options=group_results[grouping_level].tolist())
        
        if selected_group:
            stock_df = df[df[grouping_level] == selected_group].copy()
            stock_metrics = []
            tickers = stock_df['Symbol'].unique()
            min_required = max(long_window + 1, vol_window, rs_window, 10) 
            
            with st.spinner(f"Scanning stocks in {selected_group}..."):
                for ticker in tickers:
                    t_data = stock_df[stock_df['Symbol'] == ticker].sort_values('Date').copy()
                    if len(t_data) < min_required: continue 
                    
                    t_data = pd.merge(t_data, group_df[group_df[grouping_level] == selected_group][['Date', 'Close']], on='Date', suffixes=('', '_Group'))
                    t_data['RS'] = t_data['Close'] / t_data['Close_Group']
                    t_data['RS_MA'] = t_data['RS'].rolling(window=rs_window).mean()
                    t_data['Vol_Baseline'] = t_data['Volume'].rolling(window=vol_window).mean()
                    
                    if has_ohlc:
                        t_data['Range'] = t_data['High'] - t_data['Low']
                        t_data['CLV'] = np.where(t_data['Range'] > 0, ((t_data['Close'] - t_data['Low']) - (t_data['High'] - t_data['Close'])) / t_data['Range'], 0)
                        
                        t_data['Green_Vol'] = np.where(t_data['Close'] > t_data['Open'], t_data['Volume'], 0)
                        t_data['Red_Vol'] = np.where(t_data['Close'] < t_data['Open'], t_data['Volume'], 0)
                        t_data['Green_Vol_10D'] = t_data['Green_Vol'].rolling(window=10).sum()
                        t_data['Red_Vol_10D'] = t_data['Red_Vol'].rolling(window=10).sum()
                        t_data['Vol_Flow_Ratio'] = np.where(t_data['Red_Vol_10D'] > 0, t_data['Green_Vol_10D'] / t_data['Red_Vol_10D'], np.nan)
                    
                    latest_t = t_data.iloc[-1]
                    rs_trend = "Uptrend" if latest_t['RS'] > latest_t['RS_MA'] else "Downtrend"
                    vol_breakout = "Yes 🔥" if latest_t['Volume'] > (1.5 * latest_t['Vol_Baseline']) else "No"
                    
                    if has_ohlc:
                        if latest_t['CLV'] > 0.5:
                            clv_status = "Bullish Accumulation 🟢"
                        elif latest_t['CLV'] < -0.5:
                            clv_status = "Bearish Distribution 🔴"
                        else:
                            clv_status = "Neutral ⚪"

                        flow_ratio = latest_t['Vol_Flow_Ratio']
                        if pd.isna(flow_ratio) or np.isinf(flow_ratio):
                            flow_status = "N/A"
                        else:
                            flow_status = f"{round(flow_ratio, 2)}x (Buy > Sell)" if flow_ratio > 1.0 else f"{round(flow_ratio, 2)}x (Sell > Buy)"
                    else:
                        clv_status = "N/A (Missing High/Low)"
                        flow_status = "N/A (Missing Open)"
                    
                    stock_metrics.append({
                        'Symbol': ticker,
                        'Exchange': latest_t['Exchange'] if 'Exchange' in latest_t else 'N/A',
                        'Latest Close': latest_t['Close'],
                        f'RS vs {grouping_level}': rs_trend,
                        'Volume Breakout (>150%)': vol_breakout,
                        'Daily Control (CLV)': clv_status,
                        '10-Day Vol Flow': flow_status
                    })
            
            if stock_metrics:
                final_stocks = pd.DataFrame(stock_metrics)
                strong_stocks = final_stocks[(final_stocks[f'RS vs {grouping_level}'] == 'Uptrend') & (final_stocks['Volume Breakout (>150%)'] == 'Yes 🔥')].copy()
                
                if strong_stocks.empty:
                    strong_stocks = final_stocks[(final_stocks[f'RS vs {grouping_level}'] == 'Uptrend')].copy()
                    st.write(f"No breakouts today. Showing outperforming stocks in **{selected_group}**:")
                else:
                    st.write(f"🔥 Showing outperforming stocks in **{selected_group}** with Active Volume Breakouts:")
                    
                st.dataframe(strong_stocks, use_container_width=True)
            else:
                st.warning(f"No stocks in {selected_group} have enough historical data.")

        st.divider()

        # ==========================================
        # NEW PHASE 4: INDIVIDUAL TICKER DEEP DIVE
        # ==========================================
        st.header("Phase 4: Ticker Deep Dive (10-Day Chart)")
        st.markdown("Enter a specific ticker symbol to view its latest 10-day trends.")

        deep_dive_ticker = st.text_input("Enter Ticker Symbol (e.g., SHOP.TO, RY.TO):", "").strip().upper()

        if deep_dive_ticker:
            if deep_dive_ticker not in df['Symbol'].values:
                st.error(f"Ticker '{deep_dive_ticker}' not found in the dataset.")
            else:
                with st.spinner(f"Generating 10-day chart for {deep_dive_ticker}..."):
                    t_df = df[df['Symbol'] == deep_dive_ticker].sort_values('Date').copy()
                    
                    grp = t_df[grouping_level].iloc[0]
                    t_df = pd.merge(t_df, group_df[group_df[grouping_level] == grp][['Date', 'Close']], on='Date', suffixes=('', '_Group'))
                    
                    t_df['RS'] = t_df['Close'] / t_df['Close_Group']
                    t_df['RS_MA'] = t_df['RS'].rolling(window=rs_window).mean()
                    t_df['Vol_Baseline'] = t_df['Volume'].rolling(window=vol_window).mean()
                    
                    if has_ohlc:
                        t_df['Range'] = t_df['High'] - t_df['Low']
                        t_df['CLV'] = np.where(t_df['Range'] > 0, ((t_df['Close'] - t_df['Low']) - (t_df['High'] - t_df['Close'])) / t_df['Range'], 0)
                        
                        t_df['Green_Vol'] = np.where(t_df['Close'] > t_df['Open'], t_df['Volume'], 0)
                        t_df['Red_Vol'] = np.where(t_df['Close'] < t_df['Open'], t_df['Volume'], 0)
                        t_df['Green_Vol_10D'] = t_df['Green_Vol'].rolling(window=10).sum()
                        t_df['Red_Vol_10D'] = t_df['Red_Vol'].rolling(window=10).sum()
                        t_df['Vol_Flow_Ratio'] = np.where(t_df['Red_Vol_10D'] > 0, t_df['Green_Vol_10D'] / t_df['Red_Vol_10D'], np.nan)
                    
                    # Get the last 10 days
                    chart_df = t_df.tail(10).copy()
                    
                    if chart_df.empty:
                         st.warning(f"Not enough data to plot {deep_dive_ticker}.")
                    else:
                        # Create Plotly Subplots
                        fig = make_subplots(
                            rows=5, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=(
                                "Close Price", 
                                f"Relative Strength (vs {grouping_level})", 
                                "Volume & Breakout Level", 
                                "Daily Control (CLV)", 
                                "10-Day Volume Flow Ratio"
                            )
                        )

                        # Row 1: Close Price
                        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Close'], mode='lines+markers', name='Close Price', line=dict(color='blue')), row=1, col=1)

                        # Row 2: Relative Strength
                        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['RS'], mode='lines+markers', name='RS', line=dict(color='purple')), row=2, col=1)
                        fig.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['RS_MA'], mode='lines', name=f'{rs_window}-Day RS MA', line=dict(color='orange', dash='dash')), row=2, col=1)

                        # Row 3: Volume and Breakout Threshold
                        # Color volume bars green if Close > Open, else red
                        vol_colors = ['green' if (has_ohlc and row['Close'] > row['Open']) else 'red' for _, row in chart_df.iterrows()]
                        fig.add_trace(go.Bar(x=chart_df['Date'], y=chart_df['Volume'], name='Volume', marker_color=vol_colors), row=3, col=1)
                        
                        # Add the 1.5x breakout threshold line
                        breakout_level = chart_df['Vol_Baseline'] * 1.5
                        fig.add_trace(go.Scatter(x=chart_df['Date'], y=breakout_level, mode='lines', name='Breakout Threshold (1.5x)', line=dict(color='red', dash='dot')), row=3, col=1)

                        # Row 4: CLV
                        if has_ohlc:
                            clv_colors = ['green' if val > 0.5 else 'red' if val < -0.5 else 'gray' for val in chart_df['CLV']]
                            fig.add_trace(go.Bar(x=chart_df['Date'], y=chart_df['CLV'], name='CLV', marker_color=clv_colors), row=4, col=1)
                            # Add reference lines for extreme buying (+0.5) and extreme selling (-0.5)
                            fig.add_hline(y=0.5, line_dash="dash", line_color="green", row=4, col=1)
                            fig.add_hline(y=-0.5, line_dash="dash", line_color="red", row=4, col=1)

                        # Row 5: Volume Flow Ratio
                        if has_ohlc:
                             # Fill NaN with 0 for plotting purposes if needed, though they shouldn't exist if data is clean
                             chart_df['Vol_Flow_Ratio'] = chart_df['Vol_Flow_Ratio'].fillna(0)
                             flow_colors = ['green' if val > 1 else 'red' for val in chart_df['Vol_Flow_Ratio']]
                             fig.add_trace(go.Bar(x=chart_df['Date'], y=chart_df['Vol_Flow_Ratio'], name='Vol Flow Ratio', marker_color=flow_colors), row=5, col=1)
                             fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=5, col=1)

                        fig.update_layout(height=1000, showlegend=False, title_text=f"10-Day Technical Analysis for {deep_dive_ticker}")
                        fig.update_xaxes(type='category') # Ensures weekends are skipped in the chart
                        
                        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: MARKET HEATMAP
# ==========================================
elif page == "Market Heatmap 📊":
    st.title("📊 Market Heatmap")
    st.markdown("Visualize capital flow and performance across the market simultaneously.")

    st.sidebar.header("1. Data Feed")
    st.sidebar.success("🟢 Autonomous Engine Active")

    st.sidebar.header("2. Map Level")
    grouping_level = st.sidebar.radio("Group heatmap by:", ["Sector", "Industry"])

    with st.spinner("Generating heatmap..."):
        try:
            df = fetch_market_data()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            df[grouping_level] = df[grouping_level].astype(str).str.strip()
            df['Exchange'] = df['Exchange'].astype(str).str.strip()
            
            available_exchanges = sorted(df[df['Exchange'] != 'nan']['Exchange'].unique().tolist())
            selected_exchanges = st.sidebar.multiselect("Select Exchange(s):", options=available_exchanges, default=available_exchanges)
            if not selected_exchanges:
                st.warning("⚠️ Please select at least one exchange in the sidebar to generate the heatmap.")
                st.stop()
            df = df[df['Exchange'].isin(selected_exchanges)]
            
            df = df.sort_values(by=['Symbol', 'Date'])
            trading_days = len(df['Date'].unique())
            latest_date = df['Date'].max()
                
        except Exception as e:
            st.error(f"Failed to load data. Error: {e}")
            st.stop()

    st.sidebar.header("3. Heatmap Settings")
    min_price = st.sidebar.number_input("Minimum Stock Price ($)", min_value=0.0, value=10.0, step=1.0)
    
    latest_closes = df.groupby('Symbol')['Close'].last()
    valid_symbols = latest_closes[latest_closes >= min_price].index
    df = df[df['Symbol'].isin(valid_symbols)]
    
    if df.empty:
        st.error(f"⚠️ No stocks found closing above ${min_price}. Please lower your minimum price.")
        st.stop()

    max_map_periods = trading_days - 1
    map_window = st.sidebar.slider("Heatmap Timeframe (Days to measure Return)", min_value=1, max_value=max_map_periods, value=1)
    color_sensitivity = st.sidebar.slider("Color Sensitivity (Max Return %)", min_value=1.0, max_value=30.0, value=5.0, step=0.5)

    df['Return (%)'] = df.groupby('Symbol')['Close'].pct_change(periods=map_window) * 100
    latest_df = df[df['Date'] == latest_date].copy()
    
    latest_df = latest_df.replace([np.inf, -np.inf], np.nan)
    latest_df = latest_df.dropna(subset=['Return (%)', 'Volume', 'Symbol', grouping_level])
    latest_df = latest_df[latest_df['Volume'] > 0] 
    latest_df = latest_df.reset_index(drop=True)
    
    latest_df['Symbol'] = latest_df['Symbol'].astype(str).str.strip()
    latest_df = latest_df[latest_df['Symbol'] != '']
    latest_df = latest_df[latest_df['Symbol'].str.lower() != 'nan']
    
    latest_df[grouping_level] = latest_df[grouping_level].astype(str).str.strip()
    latest_df.loc[latest_df[grouping_level] == '', grouping_level] = 'Unclassified'
    latest_df.loc[latest_df[grouping_level].str.lower() == 'nan', grouping_level] = 'Unclassified'

    latest_df['Root'] = 'Overall Market'
    latest_df[grouping_level] = latest_df[grouping_level] + " "
    latest_df['Symbol'] = latest_df['Symbol'] + "  "
    latest_df = latest_df.drop_duplicates(subset=['Symbol'], keep='last')

    if not latest_df.empty:
        fig = px.treemap(
            latest_df,
            path=['Root', grouping_level, 'Symbol'],
            values='Volume',
            color='Return (%)',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,  
            range_color=[-color_sensitivity, color_sensitivity], 
            hover_data={'Close': ':.2f', 'Volume': ':.0f', 'Return (%)': ':.2f'}
        )
        fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), height=700, coloraxis_colorbar=dict(title=f"{map_window}-Day Return (%)"))
        fig.update_traces(textposition='middle center', textinfo="label+text+value", hovertemplate="<b>%{label}</b><br>Return: %{color:.2f}%<br>Volume: %{value:,.0f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Viewing Date:** {latest_date.strftime('%Y-%m-%d')}")
    else:
        st.warning("No data available to plot.")

# ==========================================
# PAGE 3: HOW IT WORKS (METRICS GUIDE)
# ==========================================
elif page == "How It Works (Metrics)":
    st.title("📚 How It Works: A Trader's Guide to the Data")
    st.markdown("Trading isn't about predicting the future; it's about following the footprints of big money. This guide breaks down the core calculations powering the scanner so you can read the market's story with confidence.")
    st.divider()

    st.header("1. Momentum Acceleration")
    st.markdown("**The Concept:** This measures whether a group's trend is speeding up or running out of gas. Instead of just looking at raw returns, we compare the short-term trend against the long-term trend.")
    st.markdown("**🚗 The Novice Analogy:** Imagine you are driving a car. Your *Long-Term Return* is your average speed on the highway (60 mph). Your *Short-Term Return* is pressing the gas pedal to pass someone (80 mph). This metric measures how hard you are pressing the gas right now.")
    st.success("**💡 Real-World Insight:** If a sector is up 5% over your long-term window but up 8% in your short-term window, the acceleration is **+3.00**. This tells you new buyers are aggressively stepping in *right now*. You want to look for positive numbers at the top of your scanner.")
    st.latex(r"Acceleration = R_{short} - R_{long}")
    st.write("---")

    st.header("2. Macro Volume Surge")
    st.markdown("**The Concept:** This measures the influx of broad capital into an entire sector relative to its baseline. Price moves can be faked in low-liquidity environments, but massive sector-wide volume cannot be faked.")
    st.markdown("**👣 The Novice Analogy:** Think of volume like footprints in the snow. A few retail traders leave a light trail. But when hedge funds and mutual funds buy into an industry, they march an entire army through the snow. This metric looks for the army.")
    st.success("**💡 Real-World Insight:** A ratio of **1.25** means today's sector volume is **25%** higher than its recent average. If a group is accelerating in price *and* has a volume ratio above 1.0, the move is legitimate. Institutions are actively buying.")
    st.latex(r"\text{Macro Surge} = \frac{V_{current}}{\overline{V}_{baseline}}")
    st.write("---")

    st.header("3. RS vs Group (Relative Strength)")
    st.markdown("**The Concept:** This evaluates a stock's performance directly against its own industry peers rather than the broad market.")
    st.markdown("**🏃‍♂️ The Novice Analogy:** Imagine a track team running a race with a heavy tailwind. The whole team will run faster times (the industry is going up). But Relative Strength asks: *Who is the fastest runner on the team?* You don't just want a stock that is being dragged up by the wind; you want the strongest athlete.")
    st.success("**💡 Real-World Insight:** If the scanner shows **'Uptrend'**, it means this specific stock is gaining more on green days and losing less on red days compared to its peers. This is your prime target.")
    st.latex(r"RS = \frac{P_{stock}}{P_{group}}")
    st.write("---")
    
    st.header("4. Daily Control (Close Location Value)")
    st.markdown("**The Concept:** Derived from institutional **Technical Analysis** principles, this identifies whether institutions bought heavily into the closing bell, even on volatile days.")
    st.markdown("**🥊 The Novice Analogy:** Think of the trading day like a tug-of-war. The bears might pull the price down in a morning panic, but if the bulls step in at 3:30 PM and drag the price all the way back up to close at the highs, the bulls won the day.")
    st.success("**💡 Real-World Insight:** A status of **Bullish Accumulation 🟢** means that even if the stock had a scary drop intraday, institutions stepped in and supported the price into the close. This confirms heavy buying pressure.")
    st.latex(r"CLV = \frac{(Close - Low) - (High - Close)}{High - Low}")
    st.write("---")
    
    st.header("5. Volume-Weighted Trend (10-Day Volume Flow)")
    st.markdown("**The Concept:** Based on core **Technical Analysis** accumulation models, this compares total trading volume on up-days versus down-days to expose hidden institutional buying or selling over the last two weeks.")
    st.markdown("**🌊 The Novice Analogy:** Imagine looking under the hood of a stock's recent performance. If the stock goes up slightly on massive volume (Green Days), but pulls back on barely any volume (Red Days), the big money is quietly building a position without alerting the public.")
    st.success("**💡 Real-World Insight:** If the flow ratio shows **> 1.0x (Buy > Sell)**, it means there is more buying volume than selling volume over the last 10 days. This confirms a volume breakout is a safe setup and not a 'bull trap.'")
    st.latex(r"\text{Flow} = \frac{\sum V_{green\_days}}{\sum V_{red\_days}}")
