import streamlit as  st
import pandas as pd
import seaborn as sns


penguins = sns.load_dataset("penguins")

st.set_page_config(page_title='Hello Streamlit', layout='wide')

st.title('Flipper Length vs. Bill Length by Species')
st.write('Turn notebooks into shareable apps in minutes!')

# add selection box for species
species = st.multiselect('Select Species', options=penguins['species'].unique(), default=penguins['species'].unique())

threshold = st.slider('Threshold_flipper', min_value=165, max_value=270, value=170, step=1)
filter_penguins = penguins[penguins['flipper_length_mm'] > threshold]
# filter by species
filter_penguins = filter_penguins[filter_penguins['species'].isin(species)]
st.scatter_chart(data=filter_penguins, x='flipper_length_mm', y='bill_length_mm', color='species')
# set range for x axis

st.write('above threshold')
st.dataframe(filter_penguins)


