Here is the completely updated and optimized code.

### 🚀 What was fixed and improved in this version:

1. **✅ Plotly 6.0 Fix Applied:** The deprecated `titlefont` dictionary was correctly nested inside the `title` dictionary to comply with Plotly's newest layout syntax. This permanently prevents the crash you experienced.
2. **📅 Shifted to Monthly Granularity (2026 Projection):** The data engine was completely rewritten. It now synthetically translates historical annual data into a monthly time-series ending in December 2025. The model utilizes **Holt-Winters Seasonal Smoothing** to project January 2026 through December 2026 month-by-month.
3. **⚙️ Mathematical Optimization for Monthly Rates:** Replacement tire assumptions (RT) are normally measured annually (e.g., 4.5 tires/year). The code now mathematically translates this into a Monthly Run Rate (`annual_rate / 12`) so the volumetric charts accurately reflect monthly demand rather than blowing the numbers out of proportion.
4. **📊 Apples-to-Apples Comparisons:** The KPI summary cards now intelligently compare the aggregated sum of the 12 projected months of 2026 against the historical 12 months of 2025, providing a much clearer annualized business view.

### 💻 Updated Code (`app.py`)

Copy and paste this completely over your current `app.py` file, commit it, and push it to GitHub. Streamlit Cloud will automatically update your live app within 30 to 60 seconds!

```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Suppress harmless statsmodels convergence warnings for a clean console
warnings.filterwarnings('ignore', '.*ConvergenceWarning.*')

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="US Class 8 Tire Market Projector",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. DATA INGESTION & MONTHLY TRANSFORMATION
# ==========================================
@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """
    Loads historical proxy data and interpolates it into a monthly time-series 
    ending in December 2025, applying standard commercial truck seasonality.
    """
    # Base Annual Data (2014 to 2025 estimated)
    years = np.arange(2014, 2026)
    sales_k_annual = [220, 250, 192, 192, 250, 276, 200, 222, 254, 269, 240, 255]
    vio_m_annual = [3.50, 3.55, 3.60, 3.65, 3.70, 3.75, 3.78, 3.82, 3.86, 3.91, 3.95, 3.99]
    
    # Generate Monthly Dates up to the end of 2025
    dates = pd.date_range(start='2014-01-01', end='2025-12-31', freq='MS')
    
    # Apply standard industry seasonality to sales (Stronger Q2 and Q4)
    seasonality = np.array([0.85, 0.90, 1.00, 1.05, 1.00, 1.05, 0.95, 0.95, 1.00, 1.05, 1.05, 1.15])
    seasonality = seasonality / seasonality.mean() # Normalize
    
    monthly_sales = []
    monthly_vio = []
    
    for i in range(len(years)):
        # Distribute annual sales across 12 months with seasonality
        monthly_sales.extend((sales_k_annual[i] / 12.0) * seasonality)
        
        # Smoothly interpolate Active Fleet (VIO) growth across the year
        start_vio = vio_m_annual[i]
        end_vio = vio_m_annual[i+1] if i+1 < len(vio_m_annual) else vio_m_annual[i] + 0.04
        monthly_vio.extend(np.linspace(start_vio, end_vio, 12, endpoint=False))
        
    df = pd.DataFrame({
        'Date': dates,
        'Sales_k': monthly_sales,
        'VIO_m': monthly_vio,
        'Data_Type': 'Historical'
    })
    return df

# ==========================================
# 3. MONTHLY FORECASTING ENGINE
# ==========================================
@st.cache_data
def generate_forecast(df_hist: pd.DataFrame, horizon_months: int, apply_epa_cycle: bool, 
                      oe_tires_per_truck: int, rt_tires_per_truck_annual: float) -> pd.DataFrame:
    """
    Forecasts monthly truck sales and VIO using Holt-Winters Seasonal Smoothing.
    Projects bottom-up volumetric demand for the next N months.
    """
    # 1. Fit Time-Series Models on Monthly Data
    # Sales uses seasonal periods of 12
    sales_model = ExponentialSmoothing(
        df_hist['Sales_k'], trend='add', seasonal='add', seasonal_periods=12, damped_trend=True, initialization_method="estimated"
    ).fit()
    
    # VIO is a steady growth curve, no seasonality needed
    vio_model = ExponentialSmoothing(
        df_hist['VIO_m'], trend='add', seasonal=None, damped_trend=False, initialization_method="estimated"
    ).fit()
    
    last_date = df_hist['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon_months, freq='MS')
    
    forecast_sales = sales_model.forecast(horizon_months).values
    forecast_vio = vio_model.forecast(horizon_months).values
    
    # 2. Apply Macro Scenario: EPA 2027 Emissions Mandate
    if apply_epa_cycle:
        for i, date in enumerate(future_dates):
            if date.year == 2026:
                # 2026 is the major pre-buy. Ramp up purchasing behavior towards the end of the year.
                ramp_multiplier = 1.05 + (date.month / 12.0) * 0.25
                forecast_sales[i] *= ramp_multiplier
            elif date.year == 2027:
                # 2027 is the severe post-buy slump
                forecast_sales[i] *= 0.75  
                
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Sales_k': np.round(forecast_sales, 2),
        'VIO_m': np.round(forecast_vio, 3),
        'Data_Type': 'Forecast'
    })
    
    # 3. Combine and Calculate Bottom-Up Tire Demand (Monthly Run-Rate)
    combined_df = pd.concat([df_hist, forecast_df], ignore_index=True)
    
    # OE Demand (Millions) = (Monthly Sales in thousands * 1000 * tires_per_truck) / 1,000,000
    combined_df['OE_Tires_m'] = (combined_df['Sales_k'] * oe_tires_per_truck) / 1000
    
    # RT Demand (Millions) = VIO in millions * (Annual RT rate / 12 months)
    monthly_rt_rate = rt_tires_per_truck_annual / 12.0
    combined_df['RT_Tires_m'] = combined_df['VIO_m'] * monthly_rt_rate
    
    combined_df['Total_Tires_m'] = combined_df['OE_Tires_m'] + combined_df['RT_Tires_m']
    
    return combined_df

# ==========================================
# 4. USER INTERFACE & LAYOUT
# ==========================================
def main():
    st.title("🚛 US Class 8 Truck Tire Market Projector")
    st.markdown("""
    This application projects the **Original Equipment (OE)** and **Replacement Tire (RT)** market 
    for Class 8 Heavy-Duty Trucks. The baseline is trained on historical data ending Dec 2025, 
    with **monthly forecasts starting in January 2026**.
    """)
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Model Parameters")
    
    st.sidebar.subheader("1. Duty-Cycle Assumptions")
    oe_tires = st.sidebar.slider(
        "Tires per New Truck (OE)", 
        min_value=6, max_value=18, value=10, step=2,
        help="A standard Class 8 6x4 tractor uses 10 tires."
    )
    
    rt_tires_annual = st.sidebar.slider(
        "Annual Replacement Tires / Truck", 
        min_value=1.0, max_value=8.0, value=4.5, step=0.1,
        help="Average annual replacement rate. The model converts this to a monthly run-rate automatically."
    )
    
    st.sidebar.subheader("2. Forecasting Scenarios")
    horizon = st.sidebar.slider("Forecast Horizon (Months)", min_value=12, max_value=36, value=12, step=12)
    
    apply_epa_cycle = st.sidebar.toggle(
        "Simulate EPA 2027 Pre-Buy Surge", 
        value=True,
        help="Ramps up 2026 monthly sales to account for fleets pre-buying ahead of strict 2027 EPA emissions rules."
    )
    
    # --- PIPELINE EXECUTION ---
    df_hist = load_historical_data()
    df_proj = generate_forecast(df_hist, horizon, apply_epa_cycle, oe_tires, rt_tires_annual)
    
    # --- KPI METRICS (2025 vs 2026) ---
    df_2025 = df_proj[df_proj['Date'].dt.year == 2025]
    df_2026 = df_proj[df_proj['Date'].dt.year == 2026]
    
    tot_25 = df_2025['Total_Tires_m'].sum()
    tot_26 = df_2026['Total_Tires_m'].sum()
    
    oe_25 = df_2025['OE_Tires_m'].sum()
    oe_26 = df_2026['OE_Tires_m'].sum()
    
    rt_25 = df_2025['RT_Tires_m'].sum()
    rt_26 = df_2026['RT_Tires_m'].sum()
    
    # VIO is a snapshot, so we take the End-of-Year value, not the sum!
    vio_25 = df_2025['VIO_m'].iloc[-1]
    vio_26 = df_2026['VIO_m'].iloc[-1] if not df_2026.empty else vio_25
    
    st.subheader("📊 Annual Projected Totals (2026 vs 2025)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("2026 Total Tire Demand", f"{tot_26:.2f}M", f"{((tot_26/tot_25)-1)*100:.1f}% vs 2025")
    col2.metric("2026 Total OE Tires", f"{oe_26:.2f}M", f"{((oe_26/oe_25)-1)*100:.1f}% vs 2025")
    col3.metric("2026 Total RT Tires", f"{rt_26:.2f}M", f"{((rt_26/rt_25)-1)*100:.1f}% vs 2025")
    col4.metric("EOY Active Fleet (VIO)", f"{vio_26:.2f}M", f"{((vio_26/vio_25)-1)*100:.1f}% vs 2025")

    st.divider()
    
    # Only plot from 2024 onwards to make the monthly fidelity easier to read
    plot_df = df_proj[df_proj['Date'] >= '2024-01-01']
    last_hist_date = df_hist['Date'].max()
    forecast_start = last_hist_date + pd.DateOffset(days=15) # For vertical line placement
    
    # --- VISUALIZATIONS & TABS ---
    tab1, tab2, tab3 = st.tabs(["📈 Market Volume Chart", "📉 Underlying Macro Drivers", "🗄️ Raw Data"])
    
    with tab1:
        st.subheader("Projected Tire Market Volumes (Monthly)")
        fig1 = px.bar(
            plot_df, x='Date', y=['OE_Tires_m', 'RT_Tires_m'],
            labels={'value': 'Tire Units (Millions)', 'variable': 'Market Segment'},
            color_discrete_map={'OE_Tires_m': '#1f77b4', 'RT_Tires_m': '#ff7f0e'},
            barmode='stack'
        )
        fig1.add_vline(x=forecast_start, line_width=2, line_dash="dash", line_color="black")
        fig1.add_annotation(x=forecast_start, y=plot_df['Total_Tires_m'].max() * 0.95, 
                            text=" Jan 2026 Forecast Starts ➔", showarrow=False, xanchor="left")
        fig1.update_layout(hovermode="x unified", xaxis_title="")
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("Drivers: Monthly Truck Sales vs Active Fleet")
        fig2 = go.Figure()
        
        # New Sales (Left Y-Axis)
        fig2.add_trace(go.Bar(
            x=plot_df['Date'], y=plot_df['Sales_k'],
            name='New Class 8 Sales (k units)',
            marker_color='rgba(44, 160, 44, 0.6)',
            yaxis='y1'
        ))
        
        # VIO (Right Y-Axis)
        fig2.add_trace(go.Scatter(
            x=plot_df['Date'], y=plot_df['VIO_m'],
            name='Active Fleet / VIO (M units)',
            mode='lines+markers',
            line=dict(color='#d62728', width=3),
            yaxis='y2'
        ))
        
        # ✅ PLOTLY 6.0 FIX APPLIED HERE: Nested dicts for title and font
        fig2.update_layout(
            hovermode="x unified",
            xaxis=dict(title=""),
            yaxis=dict(
                title=dict(text='Monthly New Truck Sales (Thousands)', font=dict(color='#2ca02c'))
            ),
            yaxis2=dict(
                title=dict(text='Active Fleet VIO (Millions)', font=dict(color='#d62728')),
                overlaying='y', side='right', showgrid=False
            ),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        fig2.add_vline(x=forecast_start, line_width=2, line_dash="dash", line_color="black")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Tabular Monthly Projection Data")
        display_df = df_proj.copy().sort_values('Date', ascending=False)
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m')
        display_df.set_index('Date', inplace=True)
        
        st.dataframe(display_df.style.format({
            'Sales_k': "{:,.1f}",
            'VIO_m': "{:,.3f}",
            'OE_Tires_m': "{:,.3f}",
            'RT_Tires_m': "{:,.3f}",
            'Total_Tires_m': "{:,.3f}"
        }), use_container_width=True)
        
        csv = df_proj.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name='class8_monthly_tire_market.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()

```
