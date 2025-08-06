import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Try to import riskfolio, if not available use manual calculations
try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False
    st.warning("⚠️ RiskFolio library not available. Using manual portfolio optimization.")

st.set_page_config(page_title="Portfolio Dashboard with Mean Variance", layout="wide")

def manual_portfolio_optimization(returns, risk_free_rate=0.0):
    """Manual implementation of mean variance optimization"""
    mu = returns.mean() * 252  # Annualized returns
    cov_matrix = returns.cov() * 252  # Annualized covariance
    
    n_assets = len(mu)
    
    # Equal weight portfolio as baseline
    equal_weights = np.array([1/n_assets] * n_assets)
    
    # Simple optimization - maximize Sharpe ratio
    # This is a simplified version - in practice you'd use scipy.optimize
    best_sharpe = -np.inf
    best_weights = equal_weights
    
    # Random search for better weights (simplified optimization)
    for _ in range(1000):
        weights = np.random.random(n_assets)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        portfolio_return = np.dot(weights, mu)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        if portfolio_vol > 0:
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights
    
    return pd.DataFrame(best_weights, index=returns.columns, columns=['Weight'])

def create_efficient_frontier(returns, num_portfolios=50):
    """Create efficient frontier manually"""
    mu = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Generate random portfolios for frontier approximation
    results = np.zeros((3, num_portfolios))
    np.random.seed(42)
    
    for i in range(num_portfolios):
        weights = np.random.random(len(mu))
        weights = weights / weights.sum()
        
        portfolio_return = np.dot(weights, mu)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        results[0, i] = portfolio_return
        results[1, i] = portfolio_vol
        results[2, i] = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    return results

# Sidebar inputs
st.sidebar.header("📊 Portfolio Configuration")

# Date inputs
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime(2024, 12, 31))

# Asset selection
st.sidebar.subheader("Asset Selection")
asset_input_method = st.sidebar.radio(
    "Choose input method:",
    ["Use predefined list", "Enter custom tickers"]
)

if asset_input_method == "Use predefined list":
    # Default assets from the notebook
    default_assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
                     'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
                     'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
    
    selected_assets = st.sidebar.multiselect(
        "Select assets for portfolio:",
        default_assets,
        default=default_assets[:10]  # Select first 10 as default
    )
else:
    ticker_input = st.sidebar.text_area(
        "Enter tickers (comma-separated):",
        value="AAPL,MSFT,GOOGL,AMZN,TSLA"
    )
    selected_assets = [ticker.strip().upper() for ticker in ticker_input.split(',') if ticker.strip()]

# Risk-free rate input
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)", 
    min_value=0.0, 
    max_value=10.0, 
    value=2.0, 
    step=0.1
) / 100

# Main application
st.title("📊 Enhanced Portfolio Dashboard with Mean Variance Optimization")

if not selected_assets:
    st.warning("Please select at least one asset to continue.")
    st.stop()

if len(selected_assets) < 2:
    st.warning("Please select at least 2 assets for portfolio optimization.")
    st.stop()

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

try:
    status_text.text("Downloading data...")
    progress_bar.progress(20)
    
    # Download data
    data = yf.download(selected_assets, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        st.error("No data found for the selected assets and date range.")
        st.stop()
    
    # Handle single asset case
    if len(selected_assets) == 1:
        data.columns = ['Close']
        data = data.to_frame()
        data.columns = selected_assets
    else:
        data = data['Close']
    
    progress_bar.progress(40)
    status_text.text("Calculating returns...")
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    progress_bar.progress(60)
    status_text.text("Optimizing portfolio...")
    
    # Portfolio optimization
    if RISKFOLIO_AVAILABLE and len(selected_assets) > 1:
        # Use RiskFolio
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu='hist', method_cov='hist')
        
        optimal_weights = port.optimization(
            model='Classic',
            rm='MV',
            obj='Sharpe',
            rf=risk_free_rate,
            hist=True
        )
        
        # Create efficient frontier
        frontier = port.efficient_frontier(
            model='Classic',
            rm='MV',
            points=50,
            rf=risk_free_rate,
            hist=True
        )
        
    else:
        # Use manual optimization
        optimal_weights = manual_portfolio_optimization(returns, risk_free_rate)
        frontier = None
    
    progress_bar.progress(80)
    status_text.text("Creating visualizations...")
    
    # Calculate portfolio metrics
    portfolio_returns = (returns * optimal_weights.values.flatten()).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # Individual asset cumulative returns
    individual_cumulative = (1 + returns).cumprod()
    
    progress_bar.progress(100)
    status_text.text("Done!")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Dashboard tabs
    tabs = st.tabs(["Portfolio Overview", "Optimization Results", "Risk Analysis", "Individual Assets"])
    
    with tabs[0]:
        st.header("📈 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            st.metric("Portfolio Sharpe Ratio", f"{portfolio_sharpe:.3f}")
        
        with col2:
            annualized_return = portfolio_returns.mean() * 252 * 100
            st.metric("Annualized Return", f"{annualized_return:.2f}%")
        
        with col3:
            annualized_vol = portfolio_returns.std() * np.sqrt(252) * 100
            st.metric("Annualized Volatility", f"{annualized_vol:.2f}%")
        
        with col4:
            max_drawdown = ((cumulative_returns / cumulative_returns.expanding().max()) - 1).min() * 100
            st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
        
        # Portfolio performance chart
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Optimized Portfolio',
            line=dict(color='blue', width=3)
        ))
        
        fig_perf.update_layout(
            title="Portfolio Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with tabs[1]:
        st.header("🎯 Optimization Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Optimal Weights")
            weights_df = optimal_weights.copy()
            weights_df.columns = ['Weight']
            weights_df['Weight %'] = weights_df['Weight'] * 100
            weights_df = weights_df.sort_values('Weight', ascending=False)
            st.dataframe(weights_df.style.format({'Weight': '{:.4f}', 'Weight %': '{:.2f}%'}))
        
        with col2:
            # Portfolio composition pie chart
            weights_for_pie = weights_df[weights_df['Weight'] > 0.001]  # Filter very small weights
            
            fig_pie = px.pie(
                values=weights_for_pie['Weight'],
                names=weights_for_pie.index,
                title="Portfolio Composition"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Efficient frontier (if available)
        if RISKFOLIO_AVAILABLE and frontier is not None:
            st.subheader("Efficient Frontier")
            
            # Calculate risk and return for frontier
            frontier_returns = []
            frontier_risks = []
            
            for i in range(frontier.shape[1]):
                weights = frontier.iloc[:, i]
                port_ret = (returns.mean() * weights).sum() * 252
                port_risk = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
                frontier_returns.append(port_ret)
                frontier_risks.append(port_risk)
            
            # Optimal portfolio point
            opt_ret = (returns.mean() * optimal_weights.values.flatten()).sum() * 252
            opt_risk = np.sqrt(np.dot(optimal_weights.values.flatten(), 
                                    np.dot(returns.cov() * 252, optimal_weights.values.flatten())))
            
            fig_frontier = go.Figure()
            fig_frontier.add_trace(go.Scatter(
                x=frontier_risks,
                y=frontier_returns,
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue')
            ))
            fig_frontier.add_trace(go.Scatter(
                x=[opt_risk],
                y=[opt_ret],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='red', size=12, symbol='star')
            ))
            
            fig_frontier.update_layout(
                title="Efficient Frontier",
                xaxis_title="Risk (Standard Deviation)",
                yaxis_title="Expected Return",
                height=400
            )
            
            st.plotly_chart(fig_frontier, use_container_width=True)
    
    with tabs[2]:
        st.header("⚠️ Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # VaR and CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            st.metric("VaR (95%)", f"{var_95*100:.3f}%")
            st.metric("CVaR (95%)", f"{cvar_95*100:.3f}%")
            
            # Returns distribution
            fig_hist = px.histogram(
                portfolio_returns,
                nbins=50,
                title="Portfolio Returns Distribution"
            )
            fig_hist.add_vline(x=var_95, line_color="red", annotation_text="VaR 95%")
            fig_hist.add_vline(x=cvar_95, line_color="orange", annotation_text="CVaR 95%")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Correlation matrix
            corr_matrix = returns.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Asset Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tabs[3]:
        st.header("📊 Individual Assets")
        
        # Asset performance comparison
        fig_assets = go.Figure()
        
        for asset in selected_assets:
            fig_assets.add_trace(go.Scatter(
                x=individual_cumulative.index,
                y=individual_cumulative[asset],
                mode='lines',
                name=asset
            ))
        
        # Add portfolio performance
        fig_assets.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Optimized Portfolio',
            line=dict(color='black', width=3, dash='dash')
        ))
        
        fig_assets.update_layout(
            title="Individual Assets vs Optimized Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=500
        )
        
        st.plotly_chart(fig_assets, use_container_width=True)
        
        # Asset statistics table
        asset_stats = pd.DataFrame(index=selected_assets)
        asset_stats['Annualized Return'] = returns.mean() * 252 * 100
        asset_stats['Annualized Volatility'] = returns.std() * np.sqrt(252) * 100
        asset_stats['Sharpe Ratio'] = (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252)
        asset_stats['Portfolio Weight'] = optimal_weights.values.flatten() * 100
        
        st.subheader("Asset Statistics")
        st.dataframe(
            asset_stats.style.format({
                'Annualized Return': '{:.2f}%',
                'Annualized Volatility': '{:.2f}%',
                'Sharpe Ratio': '{:.3f}',
                'Portfolio Weight': '{:.2f}%'
            })
        )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Portfolio optimization using Modern Portfolio Theory")

