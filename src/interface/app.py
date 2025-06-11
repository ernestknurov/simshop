import numpy as np
import pandas as pd
import streamlit as st

from src.env import ShopEnv
from src.config import Config
from src.users import (
    CheapSeekerUser,
    BrandLoverUser,
    RandomChooserUser,
    ValueOptimizerUser,
    FamiliaritySeekerUser
)
from src.recommenders import (
    RandomRecommender,
    PopularityRecommender
)


def action_to_indices(action):
    """
    Convert the action vector to indices of selected items.
    """
    return np.where(action == 1)[0].tolist()

def snake_case_to_camel_case(snake_str: str) -> str:
    return''.join(part.capitalize() for part in snake_str.split('_'))

def print_state_info():
    """
    Print the state information in a readable format.
    """
    st.write("User:", st.session_state.state['user'])
    st.write("Page Count:", st.session_state.state['history']['page_count'])
    st.write("Click Count:", st.session_state.state['history']['click_count'])
    st.write("Buy Count:", st.session_state.state['history']['buy_count'])
    st.write("Last Click Item:", st.session_state.state['history']['last_click_item'])
    st.write("Consecutive No Click Pages:", st.session_state.state['history']['consecutive_no_click_pages'])
    

config = Config()
user_params = config.get("user_params")
username_to_user = {
    user: globals()[snake_case_to_camel_case(user) + 'User'](user, **params)
    for user, params in user_params.items()
}
name_to_recommender = {
    "random": RandomRecommender(),
    "popularity": PopularityRecommender()
}

if 'initialized' not in st.session_state:
    st.session_state.user = username_to_user["cheap_seeker"]
    st.session_state.recommender = name_to_recommender["random"]
    items = pd.read_csv("src/data/catalog.csv")
    st.session_state.env = ShopEnv(items)
    st.session_state.state = st.session_state.env.reset(st.session_state.user)  # Initial state
    st.session_state.done = False
    st.session_state.action = None
    st.session_state.initialized = True

st.set_page_config(layout="wide")
st.title("SimShop")

with st.sidebar:
    with st.expander("Run settings"):
        user = st.selectbox("Select User", ["cheap_seeker", "brand_lover", "random_chooser", "value_optimizer", "familiarity_seeker"])
        recommender = st.selectbox("Select Recommender", ["random", "popularity"])
        if st.button("Update Settings"):
            st.session_state.user = username_to_user[user]
            st.session_state.recommender = name_to_recommender[recommender]
        
    take_action_btn = st.button("Take Action")
    reset_simulation_btn = st.button("Reset Simulation")

# Button to take action (recommend items + reaction on env (user))
if take_action_btn and not st.session_state.done:

    st.session_state.action = st.session_state.recommender.recommend(st.session_state.state)
    st.write("## Recommendations")
    recommendations = st.session_state.state['candidates'].loc[action_to_indices(st.session_state.action)]
    next_state, reward, done, info = st.session_state.env.step(st.session_state.action, st.session_state.user)

    # Highlight recommended rows that were clicked or bought
    clicked_idxs = np.where(info.get('clicked_items', []))[0]
    clicked_idxs = recommendations.iloc[clicked_idxs].product_id.tolist()
    bought_idxs = np.where(info.get('bought_items', []))[0]
    bought_idxs = recommendations.iloc[bought_idxs].product_id.tolist()

    def highlight_row(row):
        styling = [''] * len(row)
        if row.product_id in clicked_idxs:
            styling = ['background-color: lightyellow'] * len(row)
        if row.product_id in bought_idxs:
            styling = ['background-color: lightgreen'] * len(row)
        return styling

    styled_recommendations = recommendations.style.apply(highlight_row, axis=1)
    st.dataframe(styled_recommendations, use_container_width=True, hide_index=True)

    # Update state
    st.session_state.state = next_state
    st.session_state.done = done

    # Display step results
    with st.sidebar.expander("State information"):
        print_state_info()

    with st.sidebar.expander("Step Results"):
        st.write("Reward:", reward)
        st.write("Done:", done)
        # st.write("Clicked items:", state['candidates'].loc[action_to_indices(info['clicked_items'])])
        st.write("Clicked items:", clicked_idxs)
        st.write("Bought items:", bought_idxs)
        st.write("Click Through Rate:", info['click_through_rate'])
        st.write("Buy Through Rate:", info['buy_through_rate'])

    with st.sidebar.expander("Metrics"):
        st.write("Coverage:", st.session_state.env.coverage)
        st.write("Click Through Rate (CTR):", st.session_state.env.ctr)
        st.write("Buy Through Rate (BTR):", st.session_state.env.btr)

if st.session_state.done:
    st.sidebar.write("Simulation finished")

if reset_simulation_btn:
    st.session_state.done = False
    st.session_state.action = None
    st.session_state.state = st.session_state.env.reset(st.session_state.user)  # Reset state
    st.sidebar.write("Simulation reset")