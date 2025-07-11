import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

from src.env import ShopEnv
from src.config import Config
from src.recommenders import (
    RandomRecommender,
    PopularityRecommender,
)
from src.utils import (
    load_catalog,
    username_to_user
)

# Constants
MODEL_PATH = 'src/models/rl_recommender.zip'
CATALOG_PATH = 'src/data/catalog.csv'
USER_TYPES = ["cheap_seeker", "brand_lover", "random_chooser", "value_optimizer", "familiarity_seeker"]
RECOMMENDER_TYPES = ["random", "popularity", "rl"]
DEFAULT_USER = "cheap_seeker"
DEFAULT_RECOMMENDER = "random"

# ============================================================================
# Helper Functions
# ============================================================================

def get_rl_recommender():
    """
    Lazy import of RLRecommender to avoid torch conflicts at startup.
    
    Returns:
        RLRecommender: A newly instantiated RL recommender model
    """
    from src.recommenders import RLRecommender
    return RLRecommender()


def print_state_info(history: Dict[str, Any]) -> None:
    """
    Print the state information in a readable format.
    
    Args:
        history: Dictionary containing history information
    """
    st.write("Page Count:", history['page_count'])
    st.write("Click Count:", history['click_count'])
    st.write("Buy Count:", history['buy_count'])
    st.write("Last Click Item:", history['last_click_item'])
    st.write("Consecutive No Click Pages:", history['consecutive_no_click_pages'])


def highlight_row(row, clicked_ids: List[int], bought_ids: List[int]):
    """
    Apply styling to rows based on whether they were clicked or bought.
    
    Args:
        row: DataFrame row to style
        clicked_ids: List of product IDs that were clicked
        bought_ids: List of product IDs that were bought
        
    Returns:
        List of styling strings for each cell in the row
    """
    styling = [''] * len(row)
    if row.product_id in clicked_ids:
        styling = ['background-color: lightyellow'] * len(row)
    if row.product_id in bought_ids:
        styling = ['background-color: lightgreen'] * len(row)
    return styling


def reset_env() -> None:
    """
    Reset the environment to its initial state.
    """
    items = load_catalog(CATALOG_PATH)
    st.session_state.env = ShopEnv(items, st.session_state.user)
    st.session_state.state, _ = st.session_state.env.reset()  # Initial state
    st.session_state.done = False
    st.session_state.action = None


def load_rl_model_if_needed() -> bool:
    """
    Load RL model only when first needed to avoid startup conflicts.
    
    Returns:
        bool: True if the model was loaded successfully, False otherwise
    """
    if not st.session_state.rl_model_loaded:
        try:
            # Lazy instantiate the RL recommender if not already done
            if 'rl' not in st.session_state.name_to_recommender:
                st.session_state.name_to_recommender['rl'] = get_rl_recommender()
            
            st.session_state.name_to_recommender['rl'].load_model(MODEL_PATH)
            st.session_state.rl_model_loaded = True
        except Exception as e:
            st.error(f"Failed to load RL model: {e}")
            # Fall back to random recommender
            st.session_state.recommender = st.session_state.name_to_recommender["random"]
            return False
    return True


def initialize_session_state() -> None:
    """Initialize the session state if it hasn't been initialized yet."""
    if 'initialized' not in st.session_state:
        # config = Config()
        st.session_state.name_to_recommender = {
            "random": RandomRecommender(),
            "popularity": PopularityRecommender(),
            # Don't instantiate RLRecommender yet - will be done lazily
        }
        # Don't load the RL model immediately - defer until needed
        st.session_state.rl_model_loaded = False
        st.session_state.user = username_to_user[DEFAULT_USER]
        st.session_state.recommender = st.session_state.name_to_recommender[DEFAULT_RECOMMENDER]
        reset_env()  # Initialize environment and state
        st.session_state.initialized = True


def update_settings(user_type: str, recommender_type: str) -> bool:
    """
    Update the user and recommender settings.
    
    Args:
        user_type: Type of user to simulate
        recommender_type: Type of recommender to use
        
    Returns:
        bool: True if settings were updated successfully, False otherwise
    """
    # Load RL model if RL recommender is selected
    if recommender_type == "rl" and not load_rl_model_if_needed():
        return False
    
    st.session_state.user = username_to_user[user_type]
    st.session_state.recommender = st.session_state.name_to_recommender[recommender_type]
    reset_env()
    return True


def display_recommendations(recommendations, clicked_idxs, bought_idxs):
    """
    Display the recommendations with highlighting for clicked and bought items.
    
    Args:
        recommendations: DataFrame of recommended items
        clicked_idxs: List of product IDs that were clicked
        bought_idxs: List of product IDs that were bought
    """
    st.write("## Recommendations")
    styled_recommendations = recommendations.style.apply(
        lambda row: highlight_row(row, clicked_idxs, bought_idxs), 
        axis=1
    )
    st.dataframe(styled_recommendations, use_container_width=True, hide_index=True)


def display_sidebar_info(info, reward, done, clicked_idxs, bought_idxs):
    """
    Display various information in the sidebar.
    
    Args:
        info: Dictionary with step information
        reward: Reward from the last step
        done: Whether the episode is done
        clicked_idxs: List of product IDs that were clicked
        bought_idxs: List of product IDs that were bought
    """
    with st.sidebar.expander("State information"):
        print_state_info(info['history'])

    with st.sidebar.expander("Step Results"):
        st.write("Reward:", reward)
        st.write("Done:", done)
        st.write("Clicked items:", clicked_idxs)
        st.write("Bought items:", bought_idxs)
        st.write("Click Through Rate:", info['click_through_rate'])
        st.write("Buy Through Rate:", info['buy_through_rate'])

    with st.sidebar.expander("Metrics"):
        st.write("Coverage:", st.session_state.env.coverage)
        st.write("Click Through Rate (CTR):", st.session_state.env.ctr)
        st.write("Buy Through Rate (BTR):", st.session_state.env.btr)


def take_action():
    """Handle the action-taking process and update the UI."""
    st.session_state.action = st.session_state.recommender.recommend(st.session_state.state)
    next_state, reward, done, _, info = st.session_state.env.step(st.session_state.action)
    recommendations = info['recommended_items']

    # Highlight recommended rows that were clicked or bought
    clicked_idxs = np.where(info.get('clicked_items', []))[0]
    clicked_idxs = recommendations.iloc[clicked_idxs].product_id.tolist()
    bought_idxs = np.where(info.get('bought_items', []))[0]
    bought_idxs = recommendations.iloc[bought_idxs].product_id.tolist()
    
    display_recommendations(recommendations, clicked_idxs, bought_idxs)

    # Update state
    st.session_state.state = next_state
    st.session_state.done = done
    
    # Display information in sidebar
    display_sidebar_info(info, reward, done, clicked_idxs, bought_idxs)


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application entry point."""
    initialize_session_state()
    
    st.set_page_config(layout="wide")
    st.title("SimShop")

    # Sidebar components
    with st.sidebar:
        with st.expander("Run settings"):
            user = st.selectbox("Select User", USER_TYPES)
            recommender = st.selectbox("Select Recommender", RECOMMENDER_TYPES)
            if st.button("Update Settings"):
                if not update_settings(user, recommender):
                    st.stop()
            
        take_action_btn = st.button("Take Action")
        reset_simulation_btn = st.button("Reset Simulation")

    # Handle button actions
    if take_action_btn and not st.session_state.done:
        take_action()

    if st.session_state.done:
        st.sidebar.write("Simulation finished")

    if reset_simulation_btn:
        reset_env()
        st.sidebar.write("Simulation reset")


if __name__ == "__main__":
    main()