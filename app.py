import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Suppress harmless statsmodels convergence warnings for a clean console
warnings.filterwarnings("ignore", ".*ConvergenceWarning.*")

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="US Class 8 Tire Market Projector",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. DATA INGESTION & CACHING
# ==========================================
@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """
    Loads historical proxy data for US Class 8 Truck Sales and Active Fleet (VIO).
    Based on aggregated public industry estimates (e.g., ACT Research, FTR, ATA).
    Hardcoding vetted baseline estimates prevents API breakage and ensures reproducibility.
    """
    years = np.arange(2014, 2025)

    # Approximate US Class 8 Retail Sales in thousands (Highly Cyclical)
    sales_k = [220, 250, 192, 192, 250, 276, 200, 222, 254, 269, 240]

    # Approximate Active Fleet (Vehicles in Operation) in millions (Steady, Inelastic Growth)
    vio_m = [3.50, 3.55, 3.60, 3.65, 3.70, 3.75, 3.78, 3.82, 3.86, 3.91, 3.95]

    return pd.DataFrame(
        {"Year": years, "Sales_k": sales_k, "VIO_m": vio_m, "Data_Type": "Historical"}
    )

# ==========================================
# 3. DATA SCIENCE FORECASTING MODEL
# ==========================================
@st.cache_data
def generate_forecast(
    df_hist: pd.DataFrame,
    horizon: int,
    apply_epa_cycle: bool,
    oe_tires_per_truck: int,
    rt_tires_per_truck: float,
) -> pd.DataFrame:
    """
    Forecasts future truck sales and VIO using Double Exponential Smoothing (Holt's Method).
    Optionally simulates a domain-specific macroeconomic shock (EPA 2027 Pre-buy).
    Calculates OE and RT volumetric tire demand bottom-up.
    """
    # Fit time-series models (ETS)
    sales_model = ExponentialSmoothing(
        df_hist["Sales_k"],
        trend="add",
        damped_trend=True,
        initialization_method="estimated",
    ).fit()

    vio_model = ExponentialSmoothing(
        df_hist["VIO_m"],
        trend="add",
        damped_trend=False,
        initialization_method="estimated",
    ).fit()

    last_year = int(df_hist["Year"].max())
    future_years = np.arange(last_year + 1, last_year + 1 + horizon)

    forecast_sales = sales_model.forecast(horizon).values
    forecast_vio = vio_model.forecast(horizon).values

    # Apply domain-specific scenario: EPA 2027 emissions mandate cycle
    if apply_epa_cycle:
        for i, year in enumerate(future_years):
            if year == 2025:
                forecast_sales[i] *= 1.05
            elif year == 2026:
                forecast_sales[i] *= 1.20
            elif year == 2027:
                forecast_sales[i] *= 0.75
            elif year == 2028:
                forecast_sales[i] *= 0.90

    forecast_df = pd.DataFrame(
        {
            "Year": future_years,
            "Sales_k": np.round(forecast_sales, 1),
            "VIO_m": np.round(forecast_vio, 3),
            "Data_Type": "Forecast",
        }
    )

    combined_df = pd.concat([df_hist, forecast_df], ignore_index=True)

    # OE Demand (Millions) = (Sales in thousands * tires_per_truck) / 1000
    combined_df["OE_Tires_m"] = (combined_df["Sales_k"] * oe_tires_per_truck) / 1000

    # RT Demand (Millions) = VIO in millions * rt_tires_per_truck
    combined_df["RT_Tires_m"] = combined_df["VIO_m"] * rt_tires_per_truck

    combined_df["Total_Tires_m"] = combined_df["OE_Tires_m"] + combined_df["RT_Tires_m"]
    return combined_df

# ==========================================
# 4. PLOTLY SAFETY HELPERS
# ==========================================
def safe_hovermode(fig, preferred="x unified"):
    """
    Some environments/versions can raise on 'x unified'. Try preferred, fall back to 'x'.
    """
    try:
        fig.update_layout(hovermode=preferred)
    except Exception:
        fig.update_layout(hovermode="x")
    return fig

# ==========================================
# 5. USER INTERFACE & LAYOUT
# ==========================================
def main():
    st.title("🚛 US Class 8 Truck Tire Market Projector")
    st.markdown(
        """
        This application projects the **Original Equipment (OE)** and **Replacement Tire (RT)** market
        for Class 8 Heavy-Duty Trucks in the United States. It utilizes time-series trend modeling
        and translates truck production and fleet demographics into volumetric tire demand.
        """
    )

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Model Parameters")

    st.sidebar.subheader("1. Duty-Cycle Assumptions")
    oe_tires = st.sidebar.slider(
        "Tires per New Truck (OE)",
        min_value=6,
        max_value=18,
        value=10,
        step=2,
        help="A standard Class 8 6x4 tractor uses 10 tires. Adjust higher to account for attached OE trailers.",
    )

    rt_tires = st.sidebar.slider(
        "Replacement Tires / Truck / Year",
        min_value=1.0,
        max_value=8.0,
        value=4.5,
        step=0.1,
        help="Average annual replacement rate accounting for tread wear, retreads, and ~100k miles driven per year.",
    )

    st.sidebar.subheader("2. Forecasting Scenarios")
    horizon = st.sidebar.slider("Forecast Horizon (Years)", min_value=1, max_value=10, value=6)

    apply_epa_cycle = st.sidebar.toggle(
        "Simulate EPA 2027 Pre-Buy Cycle",
        value=True,
        help=(
            "Injects cyclicality: fleets historically buy excess trucks ahead of stricter, more expensive "
            "EPA emissions regulations (2026 surge), followed by a massive slump (2027)."
        ),
    )

    # --- PIPELINE EXECUTION ---
    df_hist = load_historical_data()
    df_proj = generate_forecast(df_hist, horizon, apply_epa_cycle, oe_tires, rt_tires)

    # --- KPI METRICS ---
    current_year = int(df_hist["Year"].max())
    end_year = int(df_proj["Year"].max())

    curr_data = df_proj[df_proj["Year"] == current_year].iloc[0]
    end_data = df_proj[df_proj["Year"] == end_year].iloc[0]

    st.subheader(f"📊 Market Summary ({current_year} ➔ {end_year})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Tire Market",
        f"{end_data['Total_Tires_m']:.2f}M",
        f"{((end_data['Total_Tires_m'] / curr_data['Total_Tires_m']) - 1) * 100:.1f}% vs {current_year}",
    )
    col2.metric(
        "Original Equipment (OE)",
        f"{end_data['OE_Tires_m']:.2f}M",
        f"{((end_data['OE_Tires_m'] / curr_data['OE_Tires_m']) - 1) * 100:.1f}%",
    )
    col3.metric(
        "Replacement Tires (RT)",
        f"{end_data['RT_Tires_m']:.2f}M",
        f"{((end_data['RT_Tires_m'] / curr_data['RT_Tires_m']) - 1) * 100:.1f}%",
    )
    col4.metric(
        "Active Fleet (VIO)",
        f"{end_data['VIO_m']:.2f}M",
        f"{((end_data['VIO_m'] / curr_data['VIO_m']) - 1) * 100:.1f}%",
    )

    st.divider()

    # --- VISUALIZATIONS & TABS ---
    tab1, tab2, tab3 = st.tabs(
        ["📈 Market Volume Chart", "📉 Underlying Macro Drivers", "🗄️ Raw Data & Methodology"]
    )

    with tab1:
        st.subheader("Projected Tire Market Volumes: OE vs RT")
        fig1 = px.bar(
            df_proj,
            x="Year",
            y=["OE_Tires_m", "RT_Tires_m"],
            labels={"value": "Tire Units (Millions)", "variable": "Market Segment"},
            color_discrete_map={"OE_Tires_m": "#1f77b4", "RT_Tires_m": "#ff7f0e"},
            barmode="stack",
        )
        fig1.add_vline(x=current_year + 0.5, line_width=2, line_dash="dash", line_color="black")
        fig1.add_annotation(
            x=current_year + 0.5,
            y=float(df_proj["Total_Tires_m"].max()),
            text=" Forecast Starts ➔",
            showarrow=False,
            xanchor="left",
        )

        safe_hovermode(fig1, "x unified")
        st.plotly_chart(fig1, use_container_width=True)

        st.info(
            "💡 **Data Science Insight:** The Replacement Tire (RT) market provides a stable revenue base "
            "driven by the total active fleet size. The Original Equipment (OE) market is volatile and heavily "
            "influenced by freight rates, capital expenditure, and emissions regulations."
        )

    with tab2:
        st.subheader("Underlying Drivers: Truck Sales vs Active Fleet")
        fig2 = go.Figure()

        # New Sales (Left Y-Axis)
        fig2.add_trace(
            go.Bar(
                x=df_proj["Year"],
                y=df_proj["Sales_k"],
                name="New Class 8 Sales (k units)",
                marker_color="rgba(44, 160, 44, 0.6)",
                yaxis="y1",
            )
        )

        # VIO (Right Y-Axis)
        fig2.add_trace(
            go.Scatter(
                x=df_proj["Year"],
                y=df_proj["VIO_m"],
                name="Active Fleet / VIO (M units)",
                mode="lines+markers",
                line=dict(color="#d62728", width=3),
                yaxis="y2",
            )
        )

        # ✅ FIXED INDENTATION + safer hovermode fallback
        fig2.update_layout(
            yaxis=dict(
                title=dict(text="New Truck Sales (Thousands)", font=dict(color="#2ca02c"))
            ),
            yaxis2=dict(
                title=dict(text="Active Fleet VIO (Millions)", font=dict(color="#d62728")),
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        )
        safe_hovermode(fig2, "x unified")

        fig2.add_vline(x=current_year + 0.5, line_width=2, line_dash="dash", line_color="black")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Tabular Projection Data")
        display_df = df_proj.copy()
        display_df["Year"] = display_df["Year"].astype(str)
        display_df.set_index("Year", inplace=True)

        st.dataframe(
            display_df.style.format(
                {
                    "Sales_k": "{:,.1f}",
                    "VIO_m": "{:,.3f}",
                    "OE_Tires_m": "{:,.2f}",
                    "RT_Tires_m": "{:,.2f}",
                    "Total_Tires_m": "{:,.2f}",
                }
            ),
            use_container_width=True,
        )

        csv = df_proj.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="class8_tire_market_forecast.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown(
            """
            ### 📚 Methodology
            **1. Public Data Proxies:** Total tire units sold per year are often paywalled (e.g., USTMA data). This model relies on **Bottom-up Modeling**, where we use public US truck population proxies and historical Class 8 retail sales to build our own target sizing.

            **2. Statistical Forecasting:** The core baseline utilizes **Double Exponential Smoothing (Holt's Linear Trend)** via `statsmodels`. We damp the trend for future truck sales to prevent unrealistic long-term compounding, while keeping a standard linear trend for the slower-moving active fleet (VIO).

            **3. Scenario Overlays:** A core tenet of data science is realizing that time-series models fail at predicting regulatory market shocks. The app features a toggle to inject domain expertise by manually simulating the expected fleet pre-buying / post-buying behavior around the upcoming **EPA 2027 Emissions Mandate**.
            """
        )

if __name__ == "__main__":
    main()
``
