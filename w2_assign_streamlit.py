import streamlit as st
import seaborn as sns
import pandas as pd

# Load data
data = sns.load_dataset("tips")

# Page config
st.set_page_config(page_title='Tips Data Analysis', layout='wide')

# Title and description
st.title('Tips Data Analysis with Streamlit')
st.write('Tips rise with total bill, but the tip/total_bill ratio goes down as total bill increases.')
st.write('---')

# Sidebar widgets
with st.sidebar:
    st.subheader('Filter Options')
    threshold = st.slider('Threshold for Total Bill', min_value=0, max_value=100, value=10, step=1)
    day_selector = st.multiselect('Select Day(s)', options=data['day'].unique(), default=data['day'].unique())

# Filter data
filtered_data = data[data['total_bill'] > threshold]
filtered_data = filtered_data[filtered_data['day'].isin(day_selector)]

# Main content
st.subheader('Total Bill vs Tip Scatter Plot')

if not filtered_data.empty:
    current_selected_day = ', '.join(filtered_data['day'].unique())
    st.write(f"Scatter plot for total bill vs tip for {current_selected_day} with total bill above {threshold}.")
    st.write(
        f"You can see current average tip {filtered_data['tip'].mean():.2f} for selected days. "
        f"But tip/total_bill ratio is {(filtered_data['tip']/filtered_data['total_bill']).mean():.2%}"
    )
    st.scatter_chart(data=filtered_data, x='total_bill', y='tip', color='day')
    st.dataframe(filtered_data)
else:
    st.warning("No data available for the selected filters.")

st.write('---')



