import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Prophet for forecasting (You may need to install it with 'pip install prophet')
from prophet import Prophet
from prophet.plot import plot_plotly

# This must be the first Streamlit command
st.set_page_config(
    page_title="Global Internet Usage Analysis",
    layout="wide",
    page_icon="🌐"
)

# Load and cache data
@st.cache_data
def load_data():
    # Load main internet usage data
    df = pd.read_csv("processed_data.csv")
    # Convert relevant columns to numeric
    df['Internet Usage (%)'] = pd.to_numeric(df['Internet Usage (%)'], errors='coerce')
    df['Mobile Subscriptions (%)'] = pd.to_numeric(df.get('Mobile Subscriptions (%)'), errors='coerce')
    df['Broadband Subscriptions (%)'] = pd.to_numeric(df.get('Broadband Subscriptions (%)'), errors='coerce')
    # Drop NaN values in 'Internet Usage (%)'
    df = df.dropna(subset=['Internet Usage (%)'])
    # Ensure 'Year' is integer
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype(int)
    return df

@st.cache_data
def load_socioeconomic_data():
    # Load socioeconomic data
    socio_df = pd.read_csv('socioeconomic_data.csv')
    # Convert relevant columns to numeric
    socio_df['GDP'] = pd.to_numeric(socio_df['GDP'], errors='coerce')
    socio_df['PopTotal'] = pd.to_numeric(socio_df['PopTotal'], errors='coerce')
    # Calculate GDP per Capita
    socio_df['GDP per Capita'] = socio_df['GDP'] / socio_df['PopTotal']
    # Handle potential division by zero or NaN
    socio_df['GDP per Capita'].replace([np.inf, -np.inf], np.nan, inplace=True)
    # Rename columns to match for merging
    socio_df.rename(columns={'Country': 'Country Name'}, inplace=True)
    # Merge with internet usage data on 'Country Name' and 'Year'
    merged_df = df.merge(socio_df[['Country Name', 'Year', 'GDP per Capita']], on=['Country Name', 'Year'], how='left')
    return merged_df

df = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
selected_year = st.sidebar.slider("Select Year", int(df['Year'].min()), int(df['Year'].max()), int(df['Year'].max()))
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    df['Country Name'].unique(),
    default=['United States', 'China', 'India', 'Brazil', 'Nigeria']
)

# Visualization Settings
st.sidebar.header("Visualization Settings")
metrics = ['Internet Usage (%)', 'Mobile Subscriptions (%)', 'Broadband Subscriptions (%)']
selected_metric = st.sidebar.selectbox("Select Metric to Analyze", metrics)
chart_type = st.sidebar.selectbox("Select Chart Type", ['Line Chart', 'Bar Chart'])
smooth_data = st.sidebar.checkbox("Smooth Data (Moving Average)")

# Top/Bottom N Settings
st.sidebar.header("Top/Bottom N Settings")
top_n = st.sidebar.slider("Select Number of Countries to Display", min_value=5, max_value=50, value=10)

# Feedback Section
st.sidebar.header("Feedback")
feedback = st.sidebar.text_area("Your feedback or suggestions:")
if st.sidebar.button("Submit"):
    # In practice, save this feedback to a file or database
    st.sidebar.success("Thank you for your feedback!")

# About Section
st.sidebar.header("About")
st.sidebar.markdown("""
This app visualizes global internet usage data sourced from reliable organizations like the **World Bank** and **ITU**. The data includes metrics such as internet users as a percentage of the population, mobile subscriptions, and broadband subscriptions.

**Methodology:**

- Data Cleaning: Missing values were handled appropriately.
- Data Integration: Additional socioeconomic indicators were merged.
- Analysis: Various statistical methods were applied to gain insights.

For more information, visit the [GitHub repository](https://github.com/hafidaso/Global-Internet-Usage-Analysis-2000-2023).
""")

st.title("🌐 Global Internet Usage Analysis (2000-2023)")

# Create tabs
tab_overview, tab_country_comp, tab_global_map, tab_data_table, tab_trend_analysis, tab_correlation, tab_forecast, tab_top_bottom = st.tabs([
    "Overview",
    "Country Comparison",
    "Global Map",
    "Data Table",
    "Trend Analysis",
    "Correlation",
    "Forecasting",
    "Top/Bottom Countries"
])

# --- Overview Tab ---
with tab_overview:
    st.header("Global Internet Usage Overview")
    # Global Trend
    global_trend = df.groupby('Year')['Internet Usage (%)'].mean().reset_index()
    fig_global_trend = px.line(
        global_trend,
        x='Year',
        y='Internet Usage (%)',
        title='Global Average Internet Usage Over Time'
    )
    st.plotly_chart(fig_global_trend, use_container_width=True)

    # Highlight Significant Events
    significant_events = {
        2000: "Dot-com Bubble Burst",
        2007: "Launch of iPhone",
        2020: "COVID-19 Pandemic",
    }
    annotations = []
    for year, event in significant_events.items():
        if year in global_trend['Year'].values:
            annotations.append(dict(
                x=year,
                y=global_trend[global_trend['Year'] == year]['Internet Usage (%)'].values[0],
                xref='x',
                yref='y',
                text=event,
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            ))

    fig_global_trend.update_layout(annotations=annotations)
    st.plotly_chart(fig_global_trend, use_container_width=True)

# --- Country Comparison Tab ---
with tab_country_comp:
    st.header("Country Comparison")
    if selected_countries:
        data_for_countries = df[df['Country Name'].isin(selected_countries)]
        # Apply moving average if selected
        if smooth_data:
            data_for_countries[selected_metric] = data_for_countries.groupby('Country Name')[selected_metric] \
                .transform(lambda x: x.rolling(window=3, min_periods=1).mean())

        # Use the selected chart type
        if chart_type == 'Line Chart':
            fig_custom = px.line(
                data_for_countries,
                x="Year",
                y=selected_metric,
                color='Country Name',
                markers=True,
                title=f"{selected_metric} over Time"
            )
        elif chart_type == 'Bar Chart':
            fig_custom = px.bar(
                data_for_countries,
                x="Year",
                y=selected_metric,
                color='Country Name',
                title=f"{selected_metric} over Time"
            )
        st.plotly_chart(fig_custom, use_container_width=True)
    else:
        st.warning("Please select at least one country")

# --- Global Map Tab ---
with tab_global_map:
    st.header(f"Global {selected_metric} in {selected_year}")
    data_for_year = df[df['Year'] == selected_year].dropna(subset=[selected_metric])

    fig_map = px.choropleth(
        data_for_year,
        locations="Country Code",
        color=selected_metric,
        hover_name="Country Name",
        color_continuous_scale=px.colors.sequential.Blues,
        range_color=(0, data_for_year[selected_metric].max()),
        title=f'Global {selected_metric} in {selected_year}'
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Interactive Choropleth Map Over Time (Animation)
    st.header("Animated Global Internet Usage Over Time")
    fig_map_animated = px.choropleth(
        df.dropna(subset=[selected_metric]),
        locations="Country Code",
        color=selected_metric,
        hover_name="Country Name",
        animation_frame="Year",
        color_continuous_scale=px.colors.sequential.Blues,
        range_color=(0, df[selected_metric].max()),
        title=f'Animated {selected_metric} Over Time'
    )
    st.plotly_chart(fig_map_animated, use_container_width=True)

# --- Data Table Tab ---
with tab_data_table:
    st.header(f"Data Table for {selected_year}")
    st.dataframe(data_for_year)

    # Download option
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(data_for_year)
    st.download_button(
        label="Download Data as CSV",
        data=csv_data,
        file_name=f'internet_usage_{selected_year}.csv',
        mime='text/csv'
    )

# --- Trend Analysis Tab ---
with tab_trend_analysis:
    st.header("Trend Analysis Over Time")
    # Allow user to select global or regional trend
    trend_options = ['Global', 'Regional', 'Country']
    selected_trend = st.selectbox("Select Trend Type", trend_options)

    if selected_trend == 'Global':
        st.subheader("Global Internet Usage Trend Over Time")
        st.plotly_chart(fig_global_trend, use_container_width=True)
    elif selected_trend == 'Regional':
        if 'Region' in df.columns:
            regions = df['Region'].unique()
            selected_regions = st.multiselect("Select Regions", regions, default=regions[:3])

            regional_trend = df[df['Region'].isin(selected_regions)].groupby(['Year', 'Region'])[selected_metric].mean().reset_index()
            fig_regional_trend = px.line(
                regional_trend,
                x='Year',
                y=selected_metric,
                color='Region',
                title='Regional Internet Usage Trends Over Time'
            )
            st.plotly_chart(fig_regional_trend, use_container_width=True)
        else:
            st.warning("Region data not available in the dataset.")
    elif selected_trend == 'Country':
        if selected_countries:
            data_for_countries = df[df['Country Name'].isin(selected_countries)]
            fig_line = px.line(
                data_for_countries,
                x="Year",
                y=selected_metric,
                color='Country Name',
                markers=True,
                title=f"{selected_metric} over Time for Selected Countries"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("Please select at least one country")

# --- Correlation Tab ---
with tab_correlation:
    st.header("Correlation with Socioeconomic Indicators")
    st.markdown("### Correlation Between Internet Usage and GDP per Capita")
    # Load socioeconomic data
    df_corr = load_socioeconomic_data()
    selected_year_corr = st.slider("Select Year for Correlation Analysis", int(df_corr['Year'].min()), int(df_corr['Year'].max()), int(df_corr['Year'].max()))
    data_for_corr = df_corr[df_corr['Year'] == selected_year_corr].dropna(subset=['GDP per Capita', selected_metric])

    if data_for_corr.empty:
        st.warning("No data available for the selected year.")
    else:
        fig_corr = px.scatter(
            data_for_corr,
            x='GDP per Capita',
            y=selected_metric,
            hover_name='Country Name',
            trendline='ols',
            title=f'{selected_metric} vs GDP per Capita in {selected_year_corr}'
        )
        st.plotly_chart(fig_corr, use_container_width=True)


# --- Forecasting Tab ---
with tab_forecast:
    st.header("Predictive Analysis (Forecasting)")
    # Select a country for forecasting
    country_for_forecast = st.selectbox("Select Country for Forecasting", df['Country Name'].unique())
    data_forecast = df[df['Country Name'] == country_for_forecast][['Year', selected_metric]].rename(columns={'Year': 'ds', selected_metric: 'y'})
    # Ensure 'ds' is datetime
    data_forecast['ds'] = pd.to_datetime(data_forecast['ds'], format='%Y')
    data_forecast = data_forecast.dropna(subset=['y'])

    if data_forecast.empty:
        st.warning(f"No data available for {country_for_forecast}.")
    else:
        model = Prophet()
        model.fit(data_forecast)

        # Create future dataframe
        future = model.make_future_dataframe(periods=5, freq='Y')
        forecast = model.predict(future)

        # Plot the forecast
        fig_forecast = plot_plotly(model, forecast, xlabel='Year', ylabel=selected_metric)
        st.plotly_chart(fig_forecast, use_container_width=True)

# --- Top/Bottom Countries Tab ---
with tab_top_bottom:
    st.header(f"Top {top_n} and Bottom {top_n} Countries in {selected_year}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Top {top_n} Countries")
        top_countries = data_for_year.nlargest(top_n, selected_metric)
        fig_top = px.bar(top_countries, x=selected_metric, y='Country Name', orientation='h', title=f"Top {top_n} Countries in {selected_year}")
        st.plotly_chart(fig_top, use_container_width=True)
    with col2:
        st.markdown(f"### Bottom {top_n} Countries")
        bottom_countries = data_for_year.nsmallest(top_n, selected_metric)
        fig_bottom = px.bar(bottom_countries, x=selected_metric, y='Country Name', orientation='h', title=f"Bottom {top_n} Countries in {selected_year}")
        st.plotly_chart(fig_bottom, use_container_width=True)