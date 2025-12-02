
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="NIFTY Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def get_available_prediction_files():
    """Get list of available prediction files from output folder"""
    pred_files = {}
    output_dir = 'output'

    if not os.path.exists(output_dir):
        return pred_files

    # Check for model-specific predictions
    for file in os.listdir(output_dir):
        if file.startswith('predictions_') and file.endswith('.csv'):
            model_name = file.replace('predictions_', '').replace('.csv', '')
            display_name = model_name.replace('_', ' ').title()
            pred_files[display_name] = os.path.join(output_dir, file)

    return pred_files

@st.cache_data
def get_best_model_prediction():
    """Get the best model prediction from final_output folder"""
    final_pred_path = 'final_output/final_predictions.csv'
    
    if os.path.exists(final_pred_path):
        return final_pred_path
    else:
        return None

@st.cache_data
def load_predictions(pred_path):
    """Load predictions from specified file"""
    try:
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None

@st.cache_data
def load_metrics():
    """Load model metrics"""
    try:
        metrics_path = 'final_output/model_evaluation_report.txt'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                content = f.read()
            return content
        else:
            return None
    except Exception as e:
        return None

def create_price_chart(df):
    """Create interactive price chart with buy/sell signals"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price with Trading Signals', 'Cumulative PnL'),
        row_heights=[0.7, 0.3]
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['Timestamp'],
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#667eea', width=2)
        ),
        row=1, col=1
    )

    # Buy signals
    buy_signals = df[df['model_call'] == 'buy']
    fig.add_trace(
        go.Scatter(
            x=buy_signals['Timestamp'],
            y=buy_signals['Close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='#10b981',
                line=dict(color='white', width=1)
            )
        ),
        row=1, col=1
    )

    # Sell signals
    sell_signals = df[df['model_call'] == 'sell']
    fig.add_trace(
        go.Scatter(
            x=sell_signals['Timestamp'],
            y=sell_signals['Close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='#ef4444',
                line=dict(color='white', width=1)
            )
        ),
        row=1, col=1
    )

    # PnL line
    fig.add_trace(
        go.Scatter(
            x=df['Timestamp'],
            y=df['model_pnl'],
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='#f59e0b', width=2),
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.1)'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="PnL (₹)", row=2, col=1)

    return fig

def create_prediction_distribution(df):
    """Create prediction distribution chart"""
    pred_counts = df['model_call'].value_counts()

    fig = go.Figure(data=[
        go.Bar(
            x=pred_counts.index,
            y=pred_counts.values,
            marker_color=['#10b981', '#ef4444'],
            text=pred_counts.values,
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Signal Distribution",
        xaxis_title="Signal Type",
        yaxis_title="Count",
        height=400,
        template='plotly_white'
    )

    return fig

def create_pnl_histogram(df):
    """Create PnL distribution histogram"""
    fig = px.histogram(
        df,
        x='model_pnl',
        nbins=50,
        title="PnL Distribution",
        labels={'model_pnl': 'PnL (₹)'},
        color_discrete_sequence=['#667eea']
    )

    fig.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 NIFTY Trading Decision System</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered trading predictions with 57.82% accuracy**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        # Logo/Header using emoji and text
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 0.5rem; margin-bottom: 1rem;'>
            <h1 style='color: white; margin: 0; font-size: 2rem;'>📈</h1>
            <p style='color: white; margin: 0; font-weight: bold;'>NIFTY AI</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Prediction Selection")
        
        # Radio button to choose between best model or other models
        selection_mode = st.radio(
            "Choose Prediction Source:",
            ["🏆 Best Model (Production)", "📊 Compare Other Models"],
            index=0
        )

        st.markdown("---")

        # Based on selection, show appropriate dropdown
        if selection_mode == "🏆 Best Model (Production)":
            # Check if best model prediction exists
            best_pred_path = get_best_model_prediction()
            
            if best_pred_path:
                st.success("✅ Best Model Loaded")
                st.info("**Calibrated Random Forest**")
                st.metric("Accuracy", "57.82%")
                selected_pred_path = best_pred_path
                selected_pred_name = "Best Model (57.82%)"
            else:
                st.error("❌ Best model predictions not found")
                st.warning("Run `python train_production.py` to generate best model predictions")
                selected_pred_path = None
                selected_pred_name = None
        
        else:  # Compare Other Models
            # Get all available model predictions
            available_predictions = get_available_prediction_files()
            
            if available_predictions:
                selected_pred_name = st.selectbox(
                    "Select Model Predictions:",
                    options=list(available_predictions.keys()),
                    index=0
                )
                selected_pred_path = available_predictions[selected_pred_name]
                st.info(f"**{selected_pred_name}**")
                st.caption("Compare different model performances")
            else:
                st.warning("⚠️ No other model predictions found")
                st.info("Run `python basic_approach.py` to generate predictions from all models")
                selected_pred_path = None
                selected_pred_name = None

        st.markdown("---")

        # Options
        st.markdown("### ⚙️ Display Options")
        show_raw_data = st.checkbox("Show Raw Data", value=False)
        date_range = st.selectbox(
            "Date Range",
            ["All", "Last Week", "Last Month", "Last 3 Months"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📊 Best Model Info")
        st.metric("Accuracy", "57.82%", "+4.33%")
        st.metric("Baseline", "53.49%")
        st.markdown("**Model**: Calibrated Random Forest")
        st.markdown("**Features**: Top 20 selected")

        st.markdown("---")
        st.markdown("### 📚 Quick Guide")
        st.markdown("""
        - **Buy Signal**: Predicts price increase
        - **Sell Signal**: Predicts price decrease
        - **PnL**: Cumulative profit/loss
        - **Best Model**: Highest accuracy (57.82%)
        """)

    # Load data
    if selected_pred_path:
        df = load_predictions(selected_pred_path)
    else:
        df = None

    if df is None or df.empty:
        st.error("❌ No predictions found.")
        
        if selection_mode == "🏆 Best Model (Production)":
            st.info("💡 Run `python train_production.py` to generate best model predictions")
        else:
            st.info("💡 Run `python basic_approach.py` to train all models and generate predictions")
        
        st.stop()

    # Display current selection
    st.info(f"📊 Currently viewing: **{selected_pred_name}**")

    # Filter by date range
    if date_range != "All":
        if date_range == "Last Week":
            cutoff = df['Timestamp'].max() - pd.Timedelta(days=7)
        elif date_range == "Last Month":
            cutoff = df['Timestamp'].max() - pd.Timedelta(days=30)
        else:  # Last 3 Months
            cutoff = df['Timestamp'].max() - pd.Timedelta(days=90)
        df = df[df['Timestamp'] >= cutoff]

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Predictions", "📈 Charts", "🎯 Performance"])

    # Tab 1: Dashboard
    with tab1:


        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(create_prediction_distribution(df), use_container_width=True)

        with col2:
            st.plotly_chart(create_pnl_histogram(df), use_container_width=True)

        st.markdown("---")

        # Recent predictions
        st.markdown("### 🕐 Recent Predictions")
        recent = df.tail(10)[['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']].copy()
        recent['Predicted'] = recent['Predicted'].map({1: '⬆️ Up', 0: '⬇️ Down'})
        recent['model_call'] = recent['model_call'].map({'buy': '🟢 Buy', 'sell': '🔴 Sell'})
        st.dataframe(recent, use_container_width=True, hide_index=True)

    # Tab 2: Predictions
    with tab2:
        st.markdown("### 📋 All Predictions")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            filter_signal = st.selectbox("Filter by Signal", ["All", "Buy", "Sell"])

        with col2:
            filter_prediction = st.selectbox("Filter by Prediction", ["All", "Up", "Down"])

        with col3:
            sort_by = st.selectbox("Sort by", ["Timestamp", "Close", "PnL"])

        # Apply filters
        filtered_df = df.copy()
        if filter_signal != "All":
            filtered_df = filtered_df[filtered_df['model_call'] == filter_signal.lower()]
        if filter_prediction != "All":
            pred_value = 1 if filter_prediction == "Up" else 0
            filtered_df = filtered_df[filtered_df['Predicted'] == pred_value]

        # Sort
        filtered_df = filtered_df.sort_values(by=sort_by, ascending=False)

        # Display
        display_df = filtered_df[['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']].copy()
        display_df['Predicted'] = display_df['Predicted'].map({1: '⬆️ Up', 0: '⬇️ Down'})
        display_df['model_call'] = display_df['model_call'].map({'buy': '🟢 Buy', 'sell': '🔴 Sell'})

        st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"predictions_{selected_pred_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        if show_raw_data:
            st.markdown("### 🔍 Raw Data")
            st.dataframe(filtered_df, use_container_width=True)

    # Tab 3: Charts
    with tab3:
        st.markdown("### 📈 Interactive Visualizations")

        # Main chart
        fig = create_price_chart(df)
        st.plotly_chart(fig, use_container_width=True)

        # Additional insights
        st.markdown("---")
        st.markdown("### 📊 Additional Insights")

        col1, col2 = st.columns(2)

        with col1:
            # Price range analysis
            st.markdown("**Price Statistics**")
            st.write(f"- Max Price: ₹{df['Close'].max():,.2f}")
            st.write(f"- Min Price: ₹{df['Close'].min():,.2f}")
            st.write(f"- Avg Price: ₹{df['Close'].mean():,.2f}")
            st.write(f"- Std Dev: ₹{df['Close'].std():,.2f}")

        with col2:
            # PnL statistics
            st.markdown("**PnL Statistics**")
            st.write(f"- Final PnL: ₹{df['model_pnl'].iloc[-1]:,.2f}")
            st.write(f"- Max PnL: ₹{df['model_pnl'].max():,.2f}")
            st.write(f"- Min PnL: ₹{df['model_pnl'].min():,.2f}")
            st.write(f"- PnL Volatility: ₹{df['model_pnl'].std():,.2f}")

    # Tab 4: Performance
    with tab4:
        st.markdown("### 🎯 Model Performance")

        # Load and display metrics (only for best model)
        if selection_mode == "🏆 Best Model (Production)":
            metrics_content = load_metrics()

            if metrics_content:
                # Parse key metrics from the report
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Model Accuracy", "57.82%", "+4.33%", help="Improvement over baseline")

                with col2:
                    st.metric("Baseline", "53.49%")

                with col3:
                    st.metric("Features Used", "20", help="Selected from 51 features")

                with col4:
                    st.metric("Test Samples", f"{len(df):,}")

                st.markdown("---")

                # Show full metrics report
                with st.expander("📄 View Full Metrics Report", expanded=False):
                    st.text(metrics_content)
            else:
                st.warning("⚠️ Metrics report not found")
        else:
            st.info("📊 Detailed metrics available only for Best Model (Production)")
            st.markdown("Switch to 'Best Model (Production)' to view comprehensive performance metrics")

        st.markdown("---")

        # Model information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔧 Model Details")
            st.markdown("""
            - **Algorithm**: Calibrated Random Forest
            - **Calibration**: CalibratedClassifierCV
            - **Feature Selection**: Top 20 from 51
            - **Target Threshold**: 0.15% movement
            - **Train/Test Split**: 70/30
            - **Class Balancing**: Applied
            """)

        with col2:
            st.markdown("### 📚 Top Features")
            st.markdown("""
            1. MACD indicators
            2. Momentum strength
            3. Volume expansion
            4. ATR (Average True Range)
            5. Volatility features
            6. Price position
            7. Bollinger Bands
            8. RSI (14-period)
            9. Stochastic Oscillator
            10. Moving Average crossovers
            """)

        st.markdown("---")

        # Trading guidelines
        st.markdown("### 💡 Trading Guidelines")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**✅ Entry Rules**")
            st.markdown("""
            1. Model confidence > 60%
            2. Expected movement > 0.15%
            3. Trade during high volatility + trending
            4. Avoid high volatility + ranging markets
            """)

        with col2:
            st.markdown("**⚠️ Risk Management**")
            st.markdown("""
            1. Position size: 1-2% of capital
            2. Stop loss: 0.3-0.5% from entry
            3. Take profit: 1.5× stop loss (minimum)
            4. Only trade high-confidence signals
            """)

if __name__ == "__main__":
    main()