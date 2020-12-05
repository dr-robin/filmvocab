import streamlit as st
import pandas as pd
import spacy

st.sidebar.title('Views')

st.text_input('Type a word', value='', max_chars=None, key=None, type='default')

from pysubparser import parser

subtitles = parser.parse('top gun-English.sub')

for subtitle in subtitles:
    st.write(subtitle.text)
