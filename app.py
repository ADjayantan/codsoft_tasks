import streamlit as st

st.title("🚀 My First Streamlit Website")
st.write("Hello! This website is built using Python + Streamlit.")

name = st.text_input("Enter your name")

if st.button("Click Me"):
    st.success(f"Welcome {name}! 👋")