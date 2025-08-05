import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import random
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

config = Config()

# Constants
MODEL_PATH = config.get("paths")["model_path"]
CATALOG_PATH = config.get("paths")["catalog_path"]
USER_TYPES = config.get("users_list")
RECOMMENDER_TYPES = config.get("recommenders_list")
DEFAULT_USER = "cheap_seeker"
DEFAULT_RECOMMENDER = "random"
NUM_RECOMMENDATIONS = config.get("num_recommendations")


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
    items = load_catalog(CATALOG_PATH, config.get("catalog_size"))
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
        st.session_state.product_id_to_emoji = {}
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
    Display the recommendations as item cards with highlighting for clicked and bought items.
    
    Args:
        recommendations: DataFrame of recommended items
        clicked_idxs: List of product IDs that were clicked
        bought_idxs: List of product IDs that were bought
    """
    st.write("## Recommendations")

    # A list of emojis to be randomly assigned to products
    EMOJI_LIST = [
        "🍎", "🍌", "🍇", "🍓", "🥝", "🥥", "🍍", "🥭", "🍑", "🍒",
        "🌶️", "🫑", "🌽", "🥕", "🥑", "🍆", "🥔", "🥦", "🥬", "🥒",
        "🍔", "🍕", "🍟", "🌭", "🍿", "🥐", "🍞", "🥖", "🥨", "🥯",
        "👕", "👖", "👚", "👗", "👔", "👘", "👠", "👡", "👢", "👟",
        "🧢", "👒", "🕶️", "👜", "🎒", "⌚", "📱", "💻", "🖥️", "🖱️",
    ]

    num_recommendations = len(recommendations)
    num_columns = 5  # Number of columns to display
    num_rows = num_recommendations // num_columns + (1 if num_recommendations % num_columns > 0 else 0)
    grid = [st.columns(num_columns, gap="small") for _ in range(num_rows)]

    for i, row in recommendations.iterrows():
        is_clicked = row.product_id in clicked_idxs
        is_bought = row.product_id in bought_idxs

        tile = grid[i // num_columns][i % num_columns]
        
        # Assign a random emoji to the product if it doesn't have one yet
        if row.product_id not in st.session_state.product_id_to_emoji:
            st.session_state.product_id_to_emoji[row.product_id] = random.choice(EMOJI_LIST)
        
        product_emoji = st.session_state.product_id_to_emoji[row.product_id]

        # Define card style based on clicked/bought status
        border_style = "border: 2px solid #4CAF50;" if is_bought else ("border: 2px solid #FFD700;" if is_clicked else "border: 1px solid #e0e0e0;")
        bg_style = "background-color: #f9fff9;" if is_bought else ("background-color: #fffff9;" if is_clicked else "background-color: #ffffff;")
        
        html_content = f"""
        <div style="{border_style} {bg_style} border-radius: 8px; padding: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px; height: 100%;">
            <div style="position: relative; text-align: center;">
                <div style="font-size: 80px; line-height: 100px; height: 100px; border-radius: 6px; background-color: #f0f0f0;">
                    {product_emoji}
                </div>
                <div style="position: absolute; top: 0; right: 0; background-color: rgba(0,0,0,0.6); color: white; padding: 2px 6px; border-radius: 0 0 0 6px; font-size: 0.8em;">
                    ${row['price']:.2f}
                </div>
            </div>
            
            <h4 style="margin: 6px 0 2px 0; font-size: 1em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{row['name']}</h4>
            
            <table style="width:100%; font-size: 0.75em; border-collapse: collapse; margin-top: 4px;">
                <tr><td style="color: #666; width: 50%;">ID:</td><td>{row['product_id']}</td></tr>
                <tr><td style="color: #666;">Quality:</td><td>{row['quality_score']}</td></tr>
                <tr><td style="color: #666;">Popularity:</td><td>{row['popularity']}</td></tr>
                <tr><td style="color: #666;">Brand:</td><td style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{row['brand']}</td></tr>
                <tr><td style="color: #666;">Category:</td><td style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{row['category']}</td></tr>
                <tr><td style="color: #666;">Subcategory:</td><td style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{row['subcategory']}</td></tr>
                <tr><td style="color: #666;">Color:</td><td>{row['color']}</td></tr>
                <tr><td style="color: #666;">Released:</td><td>{row['release_date']}</td></tr>
            </table>
            
            <div style="margin-top: 6px; font-size: 0.7em; color: #555; height: 40px; overflow: hidden; text-overflow: ellipsis;">
                {row['description']}
            </div>
        </div>
        """
        with tile:
            components.html(html_content, height=370)


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
    with st.sidebar.expander("State information", icon="ℹ️"):
        print_state_info(info['history'])

    with st.sidebar.expander("Step Results", icon="📊"):
        st.write("Reward:", reward)
        st.write("Done:", done)
        st.write("Clicked items:", clicked_idxs)
        st.write("Bought items:", bought_idxs)
        st.write("Click Through Rate:", info['click_through_rate'])
        st.write("Buy Through Rate:", info['buy_through_rate'])

    with st.sidebar.expander("Metrics", icon="📈"):
        st.write("Coverage:", st.session_state.env.coverage)
        st.write("Click Through Rate (CTR):", st.session_state.env.ctr)
        st.write("Buy Through Rate (BTR):", st.session_state.env.btr)

def take_action():
    """Handle the action-taking process and update the UI."""
    st.session_state.action = st.session_state.recommender.recommend(st.session_state.state, num_recommendations=NUM_RECOMMENDATIONS)
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
        take_action_btn = st.button("Take Action", use_container_width=True)
        reset_simulation_btn = st.button("Reset Simulation", use_container_width=True)

        with st.expander("Run settings", icon="⚙️"):
            user = st.selectbox("Select User", USER_TYPES)
            recommender = st.selectbox("Select Recommender", RECOMMENDER_TYPES)
            if st.button("Update Settings"):
                if not update_settings(user, recommender):
                    st.stop()
            

    # Handle button actions
    if take_action_btn and not st.session_state.done:
        take_action()

    if st.session_state.done:
        st.sidebar.write("Simulation finished")

    if reset_simulation_btn:
        reset_env()
        st.session_state.product_id_to_emoji = {}  # Reset emoji mapping
        st.sidebar.write("Simulation reset")


if __name__ == "__main__":
    main()