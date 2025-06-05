import numpy as np
import pandas as pd
import streamlit as st

from src.env import ShopEnv
from src.users import CheapSeekerUser
from src.recommenders import RandomRecommender

st.set_page_config(layout="wide")
st.title("SimShop")

def action_to_indices(action):
    """
    Convert the action vector to indices of selected items.
    """
    return np.where(action == 1)[0].tolist()

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    items = pd.read_csv("src/data/catalog.csv")
    st.session_state.user = CheapSeekerUser("user_A", 0.92, 0.95)
    st.session_state.env = ShopEnv(st.session_state.user, items)
    st.session_state.recommender = RandomRecommender()

# Initialize session state
if 'done' not in st.session_state:
    st.session_state.done = False
if 'state' not in st.session_state:
    st.session_state.state = st.session_state.env.reset()  # Initial state
if 'action' not in st.session_state:
    st.session_state.action = None

with st.sidebar:
    with st.expander("## Settings"):
        click_threshold = st.slider("Click Threshold", 0.0, 1.0, 0.5)
        buy_threshold = st.slider("Buy Threshold", 0.0, 1.0, 0.5)

    with st.expander("## State Information"):
        st.write("User:", st.session_state.state['user'])
        st.write("Page Count:", st.session_state.state['history']['page_count'])
        st.write("Click Count:", st.session_state.state['history']['click_count'])
        st.write("Buy Count:", st.session_state.state['history']['buy_count'])
        st.write("Last Click Item:", st.session_state.state['history']['last_click_item'])
        st.write("Consecutive No Click Pages:", st.session_state.state['history']['consecutive_no_click_pages'])


# Recommend action
if not st.session_state.done:
    st.session_state.action = st.session_state.recommender.recommend(st.session_state.state)
    st.write("## Recommendations")
    recommendations = st.session_state.state['candidates'].loc[action_to_indices(st.session_state.action)]
    st.dataframe(recommendations, use_container_width=True)

# Button to take action
if st.button("Take Action") and not st.session_state.done:

    next_state, reward, done, info = st.session_state.env.step(st.session_state.action)

    # Update state
    st.session_state.state = next_state
    st.session_state.done = done

    # Display step results
    with st.sidebar.expander("## Step Results"):
        st.write("Reward:", reward)
        st.write("Done:", done)
        # st.write("Info:")
        # st.write("Clicked items:", state['candidates'].loc[action_to_indices(info['clicked_items'])])
        # st.write("Bought items:", info['bought_items'].tolist())
        st.write("Click Through Rate:", info['click_through_rate'])
        st.write("Buy Through Rate:", info['buy_through_rate'])

if st.session_state.done:
    st.sidebar.write("Simulation finished.")

if st.button("Reset Simulation"):
    st.session_state.done = False
    st.session_state.state = st.session_state.env.reset()
    st.session_state.action = None
    st.sidebar.write("Simulation reset.")