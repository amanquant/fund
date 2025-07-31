import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

# Input
ticker = st.text_input("🔍 Search ticker:", value="AAPL").strip().upper()

if ticker:
    try:
        df = yf.download(ticker, start='2019-01-01', end='2025-07-30', progress=False)

        if df.empty:
            st.error(f"No data found for {ticker}")
        else:
            df['Daily Return'] = df['Close'].pct_change()
            returns = df['Daily Return'].dropna()
            cumulative_returns = (1 + returns).cumprod()

            # Placeholder stats
            sharpe_ratio = np.round((returns.mean() / returns.std()) * np.sqrt(252), 2)
            garch_vol_forecast = np.round(returns.std() * np.sqrt(5), 4)  # 5-day proxy
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns < var_95].mean()

            # VAR/CVAR dummy plot
            var_cvar_df = pd.DataFrame({
                'Daily Return': returns,
                'Date': returns.index
            })

            fig_var = px.histogram(var_cvar_df, x="Daily Return", nbins=100, title="Value at Risk (95%)")
            fig_var.add_vline(x=var_95, line_color="red", annotation_text="VaR 95%")
            fig_var.add_vline(x=cvar_95, line_color="orange", annotation_text="CVaR 95%")

            # Correlation dummy
            correlation_heatmap = px.imshow(
                df[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change().corr(),
                text_auto=True,
                title="Correlation Matrix"
            )

            # Dashboard layout
            st.title(f"📊 Portfolio Dashboard: {ticker}")

            tabs = st.tabs(["Performance", "Prediction", "Risk"])

            with tabs[0]:
                st.header("📈 Performance Metrics")
                st.line_chart(cumulative_returns)
                st.metric("Sharpe Ratio", f"{sharpe_ratio}")

            with tabs[1]:
                st.header("📉 Prediction Models")
                st.write("Next-week GARCH volatility estimate (placeholder):", garch_vol_forecast)
                st.line_chart(returns)

            with tabs[2]:
                st.header("⚠️ Risk Analysis")
                st.plotly_chart(fig_var, use_container_width=True)
                st.plotly_chart(correlation_heatmap, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load or process data: {e}")
