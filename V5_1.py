import streamlit as st
st.set_page_config(layout="wide", page_title="Dungeon Master!")  # Must be first Streamlit call
st.write(f"Streamlit version being used: {st.__version__}") # Add this line
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import io
import base64
import html
import itertools
import time
from io import BytesIO

# --- Configuration Constants ---
MAPS_FOLDER = "maps"
MONSTERS_DATA_PATH = "data/monsters.json"
CAMPAIGN_DATA_PATH = "data/campaigns.json"
ENCOUNTER_DATA_PATH = "DM Helper - Encounter Data.csv"  # Path for saving encounter state
INITIAL_ENCOUNTER_DATA_PATH = "DM Helper - Encounter Command Center (1).csv"  # Initial encounter data
FOREST_STYLE_ORGANIC = "organic"
FOREST_STYLE_RIGID = "rigid"

# --- Encounter Data Persistence Functions ---
def save_encounter_data(df):
    """Saves the current encounter DataFrame to a CSV file."""
    try:
        df.to_csv(ENCOUNTER_DATA_PATH, index=False)
        # st.success("Encounter data saved successfully!") # Removed this as it can be spammy
    except Exception as e:
        st.error(f"Error saving encounter data: {e}")

def load_encounter_data():
    """Loads encounter data from a CSV file if it exists."""
    if os.path.exists(ENCOUNTER_DATA_PATH):
        try:
            df = pd.read_csv(ENCOUNTER_DATA_PATH, dtype={'Conditions': str}) # Explicitly load Conditions as string
            return df
        except Exception as e:
            st.error(f"Error loading encounter data: {e}")
            return pd.DataFrame(columns=DEFAULT_ENCOUNTER_COLUMNS) # Return empty with default columns on error
    return pd.DataFrame(columns=DEFAULT_ENCOUNTER_COLUMNS) # Return empty with default columns if file doesn't exist

# Define default monster columns for an empty Bestiary
DEFAULT_MONSTER_COLUMNS = [
    "Name", "Type", "CR", "HP", "AC", "Speed", "Saves", "Skills", 
    "Damage Immunities", "Damage Resistances", "Damage Vulnerabilities", 
    "Condition Immunities", "Senses", "Languages", "Challenge", 
    "Proficiency Bonus", "Actions", "Legendary Actions", "Reactions", "Description"
]

# Define default columns for Encounter Command Center participants
DEFAULT_ENCOUNTER_COLUMNS = ["Name", "Type", "Initiative", "Max HP", "Current HP", "AC", "Conditions"]

# Define default columns for Campaign Manager nested data editors
DEFAULT_PLAYER_COLUMNS = ["Name", "Character Name", "Race", "Class", "Notes"]
DEFAULT_NPC_COLUMNS = ["Name", "Role", "Location", "Status", "Notes"]
DEFAULT_CAMPAIGN_MONSTER_COLUMNS = ["Name", "Type", "CR", "Notes"]
DEFAULT_PLOT_LINE_COLUMNS = ["Title", "Status", "Synopsis", "Key NPCs", "Locations"]

# Place this function definition near the top of your script,
# usually after imports and other helper function definitions.

def update_bestiary_df_callback():
    """Callback to update monsters_df in session state and save it."""
    # The value of the data editor (with key "bestiary_editor") is automatically
    # available as st.session_state.bestiary_editor.
    # It directly modifies st.session_state.monsters_df if that DataFrame was passed.
    # So, we just need to save the current state of st.session_state.monsters_df.
    save_monsters(st.session_state.monsters_df)
    st.success("Bestiary updated and saved automatically!")

def update_encounter_df_callback():
    """Callback to update current_encounter_df in session state and save it."""
    # The value of the data editor (with key "encounter_data_editor") is automatically
    # available in st.session_state[key].
    st.session_state.current_encounter_df = st.session_state.encounter_data_editor
    save_encounter_data(st.session_state.current_encounter_df)
    st.success("Encounter updated and saved automatically!")

# Place this function definition near the top of your script,
# typically after imports and other helper function definitions.

# Place this function definition near the top of your script,
# typically after imports and other helper function definitions.

def update_campaign_sublist_callback(sublist_key):
    """Callback to update players/NPCs sublist in session state based on data editor changes."""
    st.write(f"--- Debugging update_campaign_sublist_callback for {sublist_key} ---")
    try:
        # The full, edited DataFrame is available directly in st.session_state[key]
        edited_df = st.session_state[f"campaign_{sublist_key}_editor"]
        st.write(f"Content of edited_df from data_editor (as dict records):")
        st.json(edited_df.to_dict('records')) # Display the raw data from the editor

        # Ensure the 'Name' column is present before proceeding
        if "Name" not in edited_df.columns:
            st.error("Error: 'Name' column not found in edited data. Cannot save.")
            return

        # Convert the DataFrame to a list of dictionaries (records)
        filtered_records = []
        for record in edited_df.to_dict('records'):
            # Check if the row is truly empty (all relevant values are None, empty string, or NaN)
            is_truly_empty_row = all(
                (pd.isna(v) or (isinstance(v, str) and not v.strip()))
                for k, v in record.items() if k not in ["_index"]
            )
            
            if not is_truly_empty_row:
                filtered_records.append(record)

        st.write(f"Content after filtering empty rows (filtered_records):")
        st.json(filtered_records) # Display what will actually be saved for the sublist

        # Update the specific sublist (players or npcs) in selected_campaign_details
        st.session_state.selected_campaign_details[sublist_key] = filtered_records
        
        st.write(f"st.session_state.selected_campaign_details['{sublist_key}'] before saving:")
        st.json(st.session_state.selected_campaign_details[sublist_key]) # Display the updated sublist

        # Save the entire campaign data after updating the sublist
        if st.session_state.current_campaign_index != -1:
            st.session_state.campaigns[st.session_state.current_campaign_index] = st.session_state.selected_campaign_details
            save_campaigns(st.session_state.campaigns)
            st.success(f"Campaign {sublist_key.replace('_', ' ').title()} updated and saved automatically!")
        else:
            st.warning("Cannot auto-save sublist changes for an unsaved new campaign. Please 'Save Campaign Details' first.")
    except Exception as e:
        st.error(f"An error occurred in update_campaign_sublist_callback for '{sublist_key}': {e}")
    st.write("--- End update_campaign_sublist_callback debug ---")

# Map generation constraints
MIN_ROOM_WIDTH = 5
MAX_ROOM_WIDTH = 8
MIN_ROOM_HEIGHT = 4
MAX_ROOM_HEIGHT = 6
MIN_MAP_REQUIRED_WIDTH = MAX_ROOM_WIDTH + 4
MIN_MAP_REQUIRED_HEIGHT = MAX_ROOM_HEIGHT + 4
MAX_ELEMENTS = 10  # Max number of rooms/clearings
MAX_PLACEMENT_ATTEMPTS = 500  # Max attempts to place a room/clearing or item

# Map Tile Definitions
EMPTY_SPACE = ' '
WALL = '#'
DOOR = 'D'
ENTRANCE = 'E'
TREASURE_SYMBOL = 'T'
TRAP_SYMBOL = 'X'
VOID_SYMBOL = 'V' # New: Symbol for voids/pits
MONSTER_SYMBOL_PREFIX = 'M-'
ROOM_FLOOR = '.'
HALLWAY = '~'
FOREST_DENSE = '&' # Represents dense trees
FOREST_LIGHT = 't' # Represents light trees
FOREST_CLEARING = '.' # Represents clearings in a forest
RIVER_TILE = '=' # For rivers
STREAM_TILE = ';' # For streams
PLAYER_SYMBOL = '@' # Defined: Symbol for player character
ITEM_SYMBOL = 'I' # Defined: Generic symbol for items (used in MAP_COLORS)
FOREST_TRAIL = '-' # Defined: For trails in a forest (was implicitly used in MAP_COLORS)
EXIT_SYMBOL = 'Z'

# Ensure data directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(MAPS_FOLDER, exist_ok=True)

# Helper function to load monsters
@st.cache_data
def load_monsters():
    """
    Loads monster data from MONSTERS_DATA_PATH (data/monsters.json) into a pandas DataFrame.
    If the file doesn't exist or is empty/corrupted, it returns an empty DataFrame
    with predefined columns.
    """
    if os.path.exists(MONSTERS_DATA_PATH):
        try:
            with open(MONSTERS_DATA_PATH, 'r', encoding='utf-8') as f: # Added encoding='utf-8' here
                monsters_data = json.load(f)
            
            # Ensure all default columns are present, adding NaNs for missing ones
            df = pd.DataFrame(monsters_data)
            for col in DEFAULT_MONSTER_COLUMNS:
                if col not in df.columns:
                    df[col] = pd.NA
            # Reorder columns to match DEFAULT_MONSTER_COLUMNS
            df = df[DEFAULT_MONSTER_COLUMNS]
            return df
        except json.JSONDecodeError:
            st.warning("Monsters JSON file is corrupted or empty. Starting with an empty Bestiary.")
            # Ensure the 'data' directory exists even if JSON is bad
            os.makedirs(os.path.dirname(MONSTERS_DATA_PATH), exist_ok=True)
            return pd.DataFrame(columns=DEFAULT_MONSTER_COLUMNS)
    else:
        st.info(f"Monsters JSON file not found at '{MONSTERS_DATA_PATH}'. Creating a new empty Bestiary.")
        # Ensure the 'data' directory exists before creating a new file
        os.makedirs(os.path.dirname(MONSTERS_DATA_PATH), exist_ok=True)
        return pd.DataFrame(columns=DEFAULT_MONSTER_COLUMNS)

# The save_monsters function should also be present as provided previously:
def save_monsters(df):
    """
    Saves monster data from a pandas DataFrame to MONSTERS_DATA_PATH (data/monsters.json).
    Ensures the 'data' directory exists.
    """
    os.makedirs(os.path.dirname(MONSTERS_DATA_PATH), exist_ok=True)
    with open(MONSTERS_DATA_PATH, 'w', encoding='utf-8') as f: # Added encoding='utf-8' here too
        # Convert DataFrame to a list of dictionaries before saving to JSON
        df.to_json(f, orient="records", indent=4)
    st.success("Bestiary saved successfully!")

# Helper function to load campaign data
def load_campaigns():
    """
    Loads campaign data from CAMPAIGN_DATA_PATH (data/campaigns.json) into a list of dictionaries.
    If the file doesn't exist or is empty/corrupted, it returns an empty list.
    """
    if os.path.exists(CAMPAIGN_DATA_PATH):
        try:
            with open(CAMPAIGN_DATA_PATH, 'r', encoding='utf-8') as f:
                campaigns_data = json.load(f)
            # Ensure it's always a list of dictionaries
            if not isinstance(campaigns_data, list):
                st.warning("Campaigns JSON file is not in the expected list format. Starting with an empty Campaign Manager.")
                return []
            
            # Ensure each loaded campaign has the necessary sub-lists
            for campaign in campaigns_data:
                campaign.setdefault("players", [])
                campaign.setdefault("npcs", [])
                campaign.setdefault("monsters_in_campaign", []) # Use this for campaign-specific monsters
                campaign.setdefault("plot_lines", []) # For plot/quest tracking
            return campaigns_data
        except json.JSONDecodeError:
            st.warning("Campaigns JSON file is corrupted or empty. Starting with an empty Campaign Manager.")
            return []
    else:
        st.info(f"Campaigns JSON file not found at '{CAMPAIGN_DATA_PATH}'. Creating a new empty Campaign Manager.")
        # Ensure the 'data' directory exists before creating a new file
        os.makedirs(os.path.dirname(CAMPAIGN_DATA_PATH), exist_ok=True)
        return []

# Helper function to save campaign data
def save_campaigns(campaigns_list):
    """
    Saves campaign data from a list of dictionaries to CAMPAIGN_DATA_PATH (data/campaigns.json).
    Ensures the 'data' directory exists.
    """
    os.makedirs(os.path.dirname(CAMPAIGN_DATA_PATH), exist_ok=True)
    with open(CAMPAIGN_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(campaigns_list, f, indent=4)
    st.success("Campaigns saved successfully!")

# Dungeon generation helpers
def _place_dungeon_elements(grid, num_elements, element_min_w, element_max_w, element_min_h, element_max_h, element_type=ROOM_FLOOR):
    h, w = len(grid), len(grid[0])
    elements_data = {}
    placed_elements_count = 0

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        if placed_elements_count >= num_elements:
            break

        current_w = random.randint(element_min_w, element_max_w)
        current_h = random.randint(element_min_h, element_max_h)

        start_r = random.randint(1, h - current_h - 2)
        start_c = random.randint(1, w - current_w - 2)

        end_r = start_r + current_h
        end_c = start_c + current_w

        # Check for overlap with existing placed elements or walls
        overlap = False
        for r_check in range(start_r - 1, end_r + 2): # Check with padding (1 unit around the proposed element)
            for c_check in range(start_c - 1, end_c + 2):
                if not (0 <= r_check < h and 0 <= c_check < w):
                    overlap = True
                    break

                # --- CRITICAL MODIFICATION FOR PLACEMENT LOGIC ---
                if element_type == FOREST_CLEARING:
                    # For forest clearings:
                    # The core area (where the clearing itself goes) must be EMPTY_SPACE.
                    # The padding/border area (where walls *would* go, or just the surrounding terrain)
                    # can be either EMPTY_SPACE or FOREST_LIGHT.
                    is_core_area = (start_r <= r_check < end_r and start_c <= c_check < end_c)

                    if is_core_area:
                        # Core clearing area must be truly empty
                        if grid[r_check][c_check] != EMPTY_SPACE:
                            overlap = True
                            break
                    else: # This is the padding area around the core clearing
                        # Padding can be empty space or light forest
                        if grid[r_check][c_check] != EMPTY_SPACE and grid[r_check][c_check] != FOREST_LIGHT:
                            overlap = True
                            break
                else: # For other element types (like ROOM_FLOOR for dungeons), keep the strict EMPTY_SPACE check
                    if grid[r_check][c_check] != EMPTY_SPACE:
                        overlap = True
                        break
            # --- END CRITICAL MODIFICATION ---
            if overlap:
                break

        if not overlap:
            # Place the element
            element_name = f"Element_{placed_elements_count + 1}"
            elements_data[element_name] = {'coords': []}
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    grid[r][c] = element_type
                    elements_data[element_name]['coords'].append((r,c))

            # Place walls around the element (this part is fine, as WALL will replace EMPTY_SPACE or FOREST_LIGHT)
            for r in range(start_r - 1, end_r + 1):
                for c in range(start_c - 1, end_c + 1):
                    # Ensure we only replace empty or light forest with walls, not dense forest or other elements
                    if grid[r][c] == EMPTY_SPACE or grid[r][c] == FOREST_LIGHT:
                        grid[r][c] = WALL

            placed_elements_count += 1

    print(f"DEBUG: _place_dungeon_elements FINAL return (Type: {element_type}):")
    print(f"  elements_data: {elements_data}")
    print(f"  placed_elements_count: {placed_elements_count}")
    return grid, elements_data, placed_elements_count

def _connect_elements(grid, elements_data, element_type=ROOM_FLOOR):
    # This is a simplified connection logic for demonstration.
    # In a real dungeon generator, you'd use algorithms like BSP trees or Randomized Prim's.
    element_centers = []
    for name, data in elements_data.items():
        if data['coords']:
            min_r = min(coord[0] for coord in data['coords'])
            max_r = max(coord[0] for coord in data['coords'])
            min_c = min(coord[1] for coord in data['coords'])
            max_c = max(coord[1] for coord in data['coords'])
            center_r = (min_r + max_r) // 2
            center_c = (min_c + max_c) // 2
            element_centers.append((center_r, center_c, name))
    
    if len(element_centers) < 2:
        return grid

    # Connect all elements to the first element's center
    first_center_r, first_center_c, _ = element_centers[0]

    for i in range(1, len(element_centers)):
        target_r, target_c, _ = element_centers[i]
        
        # Simple L-shaped corridor
        curr_r, curr_c = first_center_r, first_center_c
        
        while curr_r != target_r:
            grid[curr_r][curr_c] = HALLWAY
            curr_r += 1 if target_r > curr_r else -1
            if grid[curr_r][curr_c] == WALL: # If we hit a wall, make a door
                grid[curr_r][curr_c] = DOOR
        
        while curr_c != target_c:
            grid[curr_r][curr_c] = HALLWAY
            curr_c += 1 if target_c > curr_c else -1
            if grid[curr_r][curr_c] == WALL: # If we hit a wall, make a door
                grid[curr_r][curr_c] = DOOR
        
        grid[curr_r][curr_c] = HALLWAY # Mark the final cell
    
    return grid

def _generate_forest_terrain(grid, width, height, density_ratio=0.05):
    # This function uses the new forest symbols
    for r in range(height):
        for c in range(width):
            if random.random() < density_ratio:
                grid[r][c] = FOREST_DENSE
            elif random.random() < density_ratio * 2: # Lighter density for light trees
                grid[r][c] = FOREST_LIGHT
            else:
                grid[r][c] = EMPTY_SPACE # Use EMPTY_SPACE for pathable areas not explicitly clearings

    # Ensure clearings are distinct
    #for name, data in st.session_state.map_elements_data.items():
        #if "clearing" in name.lower() or "element" in name.lower(): # Assuming "clearing" is in element name
             #for r, c in data['coords']:
                 #if 0 <= r < height and 0 <= c < width:
                     #grid[r][c] = FOREST_CLEARING # Apply clearing symbol to element areas
    
    return grid

def _place_entrance(grid, map_type, elements_data):
    h, w = len(grid), len(grid[0])
    entrance_r, entrance_c = -1, -1 # Initialize with invalid coordinates

    # --- Dungeon and Mountain: Linked to a hallway/room ---
    if map_type in ["Dungeon", "Mountain"]:
        suitable_spots = []
        for element_name, data in elements_data.items():
            for r, c in data['coords']:
                # Consider the element's own valid connection points (ROOM_FLOOR or HALLWAY)
                if grid[r][c] in [ROOM_FLOOR, HALLWAY]:
                    suitable_spots.append((r, c))
                # Also consider adjacent EMPTY_SPACE cells
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == EMPTY_SPACE:
                        suitable_spots.append((nr, nc))
        
        if suitable_spots:
            # Choose a random unique suitable spot from the collected list
            entrance_r, entrance_c = random.choice(list(set(suitable_spots)))
        else:
            # Fallback if no elements or suitable spots found (e.g., very small map, 0 elements)
            # Try to place on the first available EMPTY_SPACE from (1,1) inwards
            for r_scan in range(1, h - 1):
                for c_scan in range(1, w - 1):
                    if grid[r_scan][c_scan] == EMPTY_SPACE:
                        entrance_r, entrance_c = r_scan, c_scan
                        break
                if entrance_r != -1: break

    # --- Forest, Desert, Underdark: At the end of the grid (random edge) ---
    elif map_type in ["Forest", "Desert", "Underdark"]:
        side = random.choice(["top", "bottom", "left", "right"])
        
        if side == "top":
            entrance_r = 0
            # Find a non-wall column on the top edge (avoiding corners 0,0 and 0,w-1)
            valid_cols = [c for c in range(1, w - 1) if grid[0][c] != WALL]
            if valid_cols: entrance_c = random.choice(valid_cols)
            else: entrance_c = random.randint(1, w - 2) # Fallback if only walls
        elif side == "bottom":
            entrance_r = h - 1
            valid_cols = [c for c in range(1, w - 1) if grid[h-1][c] != WALL]
            if valid_cols: entrance_c = random.choice(valid_cols)
            else: entrance_c = random.randint(1, w - 2)
        elif side == "left":
            entrance_c = 0
            valid_rows = [r for r in range(1, h - 1) if grid[r][0] != WALL]
            if valid_rows: entrance_r = random.choice(valid_rows)
            else: entrance_r = random.randint(1, h - 2)
        elif side == "right":
            entrance_c = w - 1
            valid_rows = [r for r in range(1, h - 1) if grid[r][w-1] != WALL]
            if valid_rows: entrance_r = random.choice(valid_rows)
            else: entrance_r = random.randint(1, h - 2)
        
        # --- Underdark specific: Generate a pathway from edge entrance to a hallway or room ---
        if map_type == "Underdark":
            if elements_data and entrance_r != -1 and entrance_c != -1:
                closest_element_coord = None
                min_dist = float('inf')
                
                # Find the closest existing hallway or room floor to connect to
                for element_name, data in elements_data.items():
                    for r_elem, c_elem in data['coords']:
                        if grid[r_elem][c_elem] in [ROOM_FLOOR, HALLWAY]: # Connect only to valid element types
                            dist = abs(entrance_r - r_elem) + abs(entrance_c - c_elem)
                            if dist < min_dist:
                                min_dist = dist
                                closest_element_coord = (r_elem, c_elem)
                
                if closest_element_coord:
                    curr_r, curr_c = entrance_r, entrance_c
                    target_r, target_c = closest_element_coord

                    # Draw horizontal segment of the path
                    while curr_c != target_c:
                        # Only draw path on EMPTY_SPACE to avoid overwriting existing features
                        if grid[curr_r][curr_c] == EMPTY_SPACE:
                            grid[curr_r][curr_c] = HALLWAY 
                        if curr_c < target_c: curr_c += 1
                        else: curr_c -= 1
                    
                    # Draw vertical segment of the path
                    while curr_r != target_r:
                        if grid[curr_r][curr_c] == EMPTY_SPACE:
                            grid[curr_r][curr_c] = HALLWAY
                        if curr_r < target_r: curr_r += 1
                        else: curr_r -= 1
                    
                    # Ensure the target element's cell itself is marked as HALLWAY if it was empty
                    if grid[target_r][target_c] == EMPTY_SPACE:
                        grid[target_r][target_c] = HALLWAY


    # --- Final Entrance Placement at the determined coordinates ---
    if 0 <= entrance_r < h and 0 <= entrance_c < w:
        # If the chosen spot is a WALL, try to find an adjacent non-wall spot
        if grid[entrance_r][entrance_c] == WALL:
            found_clear_spot = False
            for dr, dc in [(0,0), (0,1), (0,-1), (1,0), (-1,0)]: # Check current spot then neighbors
                nr, nc = entrance_r + dr, entrance_c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != WALL:
                    entrance_r, entrance_c = nr, nc
                    found_clear_spot = True
                    break
            if not found_clear_spot:
                # If still stuck on a wall, fall back to finding first empty spot from (1,1)
                for r_scan in range(1, h - 1):
                    for c_scan in range(1, w - 1):
                        if grid[r_scan][c_scan] == EMPTY_SPACE:
                            entrance_r, entrance_c = r_scan, c_scan
                            break
                    if entrance_r != -1: break

        grid[entrance_r][entrance_c] = ENTRANCE
        elements_data["Entrance"] = {'coords': [(entrance_r, entrance_c)]}
    else:
        # Absolute fallback: if no valid spot found, place at (1,1) or a nearby alternative
        if grid[1][1] != WALL:
            grid[1][1] = ENTRANCE
            elements_data["Entrance"] = {'coords': [(1,1)]}
        elif grid[1][2] != WALL:
            grid[1][2] = ENTRANCE
            elements_data["Entrance"] = {'coords': [(1,2)]}
        elif grid[2][1] != WALL:
            grid[2][1] = ENTRANCE
            elements_data["Entrance"] = {'coords': [(2,1)]}
        # Further fallback logic can be added here if needed for very constrained maps

    return grid, elements_data


def get_unique_token_for_item(grid, prefix="I"):
    """Generates a unique incremental token for items like monsters (M-1, M-2)."""
    max_num = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cell_val = grid[r][c]
            if cell_val.startswith(prefix + "-"):
                try:
                    num = int(cell_val.split('-')[1])
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass # Not a valid number
    return f"{prefix}-{max_num + 1}"

def place_item_on_grid(grid, item_symbol, item_details=None, room_name=None, map_type=None):
    """
    Places a single item on the grid in an empty (or specified) location.
    Args:
        grid (list of list of str): The map grid.
        item_symbol (str): The symbol to place (e.g., 'T', 'X', 'M-1').
        item_details (dict, optional): Dictionary with details about the item (e.g., {'type': 'gold'}).
        room_name (str, optional): The name of the room/clearing to place the item in. If None or "Random", places randomly.
        map_type (str, optional): The type of map (e.g., "Dungeon", "Forest"). Used for smarter placement.
    Returns:
        tuple: (grid, success_boolean, placed_coords)
    """
    h, w = len(grid), len(grid[0])
    placed = False
    attempts = 0
    placed_coords = None

    target_coords_pool = []
    if room_name and room_name != "Random" and st.session_state.map_elements_data:
        element_data = st.session_state.map_elements_data.get(room_name, {})
        target_coords_pool = element_data.get('coords', [])
        if not target_coords_pool:
            st.warning(f"Area '{room_name}' not found for placing item '{item_symbol}'. Attempting random placement.")
            room_name = "Random" # Fallback to random if area not found

    while not placed and attempts < MAX_PLACEMENT_ATTEMPTS:
        if room_name == "Random" or not room_name:
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
        else:
            if not target_coords_pool: # If room_name was invalid/empty coords
                r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            else:
                r, c = random.choice(target_coords_pool)

        # Determine valid base tiles for placement based on map type
        valid_base_tiles = []
        if map_type in ["Dungeon", "Town"]:
            valid_base_tiles = [ROOM_FLOOR, HALLWAY, ENTRANCE, EMPTY_SPACE]
        elif map_type == "Forest":
            valid_base_tiles = [FOREST_CLEARING, FOREST_LIGHT, EMPTY_SPACE, ENTRANCE]
            # Traps can also be hidden in dense forest
            if item_symbol == TRAP_SYMBOL:
                valid_base_tiles.append(FOREST_DENSE)
        elif map_type in ["Sea", "Desert", "Mountain", "Underdark"]:
            valid_base_tiles = [EMPTY_SPACE, ROOM_FLOOR, HALLWAY, FOREST_CLEARING, FOREST_LIGHT, FOREST_DENSE, ENTRANCE, WALL]
            # For mountain/underdark, items can be in cave walls, for sea/desert in "empty" water/sand
            # Exclude only explicit voids
            valid_base_tiles = [tile for tile in valid_base_tiles if tile != VOID_SYMBOL]
        else: # Default to general walkable tiles
            valid_base_tiles = [ROOM_FLOOR, HALLWAY, FOREST_CLEARING, EMPTY_SPACE, ENTRANCE, DOOR]

        # Check if the cell is suitable
        if grid[r][c] in valid_base_tiles:
            # Do not place on doors if possible, but allow if no other options in area for non-door items
            if grid[r][c] == DOOR and (item_symbol == TREASURE_SYMBOL or item_symbol == TRAP_SYMBOL or item_symbol.startswith(MONSTER_SYMBOL_PREFIX)):
                attempts += 1
                continue
            
            # Check if the cell is already occupied by a different item type (M-, T, X, V)
            is_occupied_by_item = False
            if grid[r][c].startswith(MONSTER_SYMBOL_PREFIX) or \
               grid[r][c] == TREASURE_SYMBOL or \
               grid[r][c] == TRAP_SYMBOL or \
               grid[r][c] == VOID_SYMBOL:
                is_occupied_by_item = True
            
            if not is_occupied_by_item:
                grid[r][c] = item_symbol
                placed = True
                placed_coords = (r, c)
        attempts += 1

    if not placed:
        st.warning(f"Could not place '{item_symbol}' in '{room_name}' after {MAX_PLACEMENT_ATTEMPTS} attempts.")
    return grid, placed, placed_coords

def place_void_on_grid(grid, void_width, void_height, void_name, map_elements_data, area_name=None):
    """
    Places a rectangular void/pit on the grid.
    Args:
        grid (list of list of str): The map grid.
        void_width (int): Width of the void.
        void_height (int): Height of the void.
        void_name (str): Name of the void (e.g., "Pit 1").
        map_elements_data (dict): Dictionary of named map elements (rooms, clearings, etc.) and their coords.
        area_name (str, optional): The name of the area (room, clearing) to place the void in. If None or "Random", places randomly.
    Returns:
        tuple: (grid, success_boolean, placed_coords_list)
    """
    h, w = len(grid), len(grid[0])
    placed = False
    attempts = 0
    placed_coords_list = [] # Store all coordinates of the void

    target_coords_pool = []
    if area_name and area_name != "Random" and map_elements_data:
        element_data = map_elements_data.get(area_name, {})
        target_coords_pool = element_data.get('coords', [])
        if not target_coords_pool:
            st.warning(f"Area '{area_name}' not found for placing void '{void_name}'. Attempting random placement.")
            area_name = "Random"

    while not placed and attempts < MAX_PLACEMENT_ATTEMPTS:
        if area_name == "Random" or not area_name:
            # Random starting point for the void, prioritizing floor or hallway
            potential_start_coords = []
            for r_idx in range(h - void_height + 1): # Corrected range to allow full void placement
                for c_idx in range(w - void_width + 1): # Corrected range to allow full void placement
                    # Check if the entire proposed void area is suitable (floor or hallway)
                    is_suitable_area = True
                    for r_check in range(r_idx, r_idx + void_height):
                        # FIX: Changed c_check + void_width to c_idx + void_width
                        for c_check in range(c_idx, c_idx + void_width):
                            if grid[r_check][c_check] not in [ROOM_FLOOR, HALLWAY, FOREST_CLEARING, EMPTY_SPACE, ENTRANCE, DOOR]:
                                is_suitable_area = False
                                break
                        if not is_suitable_area:
                            break
                    if is_suitable_area:
                        potential_start_coords.append((r_idx, c_idx))
            
            if not potential_start_coords:
                st.warning(f"No suitable random area found for void '{void_name}' of size {void_width}x{void_height}.")
                return grid, False, [] # No place to put it
            
            start_r, start_c = random.choice(potential_start_coords)

        else: # Specific area placement
            if not target_coords_pool:
                attempts += 1
                continue

            # Find a suitable top-left corner within the target area
            suitable_area_coords = []
            for r_el, c_el in target_coords_pool:
                # Calculate potential top-left corner for the void
                # This logic tries to fit the void around the element coordinate
                start_r_cand = r_el - random.randint(0, void_height - 1)
                start_c_cand = c_el - random.randint(0, void_width - 1)

                # Ensure proposed void is within map bounds and is composed of suitable tiles
                if 0 <= start_r_cand < h - void_height + 1 and \
                   0 <= start_c_cand < w - void_width + 1:
                    is_suitable_area = True
                    for r_check in range(start_r_cand, start_r_cand + void_height):
                        # FIX: Changed c_check + void_width to start_c_cand + void_width
                        for c_check in range(start_c_cand, start_c_cand + void_width):
                            if grid[r_check][c_check] not in [ROOM_FLOOR, HALLWAY, FOREST_CLEARING, EMPTY_SPACE, ENTRANCE, DOOR]:
                                is_suitable_area = False
                                break
                        if not is_suitable_area:
                            break
                    if is_suitable_area:
                        suitable_area_coords.append((start_r_cand, start_c_cand))
            
            if not suitable_area_coords:
                st.warning(f"No suitable void placement found within '{area_name}' for '{void_name}'.")
                return grid, False, []
            
            start_r, start_c = random.choice(suitable_area_coords)


        # At this point, start_r, start_c should be valid for the void
        # Place the void
        current_placed_void_coords = []
        for r_fill in range(start_r, start_r + void_height):
            for c_fill in range(start_c, start_c + void_width): 
                grid[r_fill][c_fill] = VOID_SYMBOL
                current_placed_void_coords.append((r_fill, c_fill))
        placed = True
        placed_coords_list = current_placed_void_coords
        
        attempts += 1

    if not placed:
        st.warning(f"Could not place void '{void_name}' ({void_width}x{void_height}) after {MAX_PLACEMENT_ATTEMPTS} attempts.")
    return grid, placed, placed_coords_list

def place_all_items_on_map(grid, map_elements_data, map_type):
    """
    Places all configured monsters, treasures, traps, and voids onto the map grid.
    Args:
        grid (list of list of str): The map grid.
        map_elements_data (dict): Dictionary of named map elements (rooms, clearings, etc.) and their coords.
        map_type (str): The type of map (e.g., "Dungeon", "Forest").
    Returns:
        list of list of str: The updated map grid with items placed.
    """
    current_grid = [row[:] for row in grid] # Create a copy to modify
    st.session_state.item_locations_details = {} # Clear previous item details

    # 👹 Place Monsters
    for config in st.session_state.monsters_config:
        amount = config.get('amount', 0)
        monster_type = config.get('monster_type', 'None')
        area = config.get('area', 'Random')
        
        if amount > 0 and monster_type != "None":
            st.info(f"Attempting to place {amount} {monster_type} monsters in {area}...")
            for _ in range(amount):
                monster_token = get_unique_token_for_item(current_grid, "M")
                current_grid, placed, placed_coords = place_item_on_grid(current_grid, monster_token, 
                                                           item_details={'type': monster_type}, 
                                                           room_name=area, map_type=map_type)
                if placed:
                    st.session_state.item_locations_details[placed_coords] = {
                        'symbol': monster_token, 
                        'description': f"Monster: {monster_type}"
                    }
                else:
                    st.warning(f"Failed to place a '{monster_type}' monster in '{area}'.")

    # 💰 Place Treasures
    for config in st.session_state.treasures_config:
        amount = config.get('amount', 0)
        treasure_type = config.get('treasure_type', '')
        amount_type = config.get('amount_type', 'pieces')
        area = config.get('area', 'Random')

        if amount > 0 and treasure_type:
            st.info(f"Attempting to place {amount} {treasure_type} ({amount_type}) treasures in {area}...")
            for _ in range(amount):
                current_grid, placed, placed_coords = place_item_on_grid(current_grid, TREASURE_SYMBOL, 
                                                           item_details={'type': treasure_type, 'amount_type': amount_type, 'amount': amount}, 
                                                           room_name=area, map_type=map_type)
                if placed:
                    st.session_state.item_locations_details[placed_coords] = {
                        'symbol': TREASURE_SYMBOL, 
                        'description': f"Treasure: {treasure_type} ({amount} {amount_type})"
                    }
                else:
                    st.warning(f"Failed to place a '{treasure_type}' treasure in '{area}'.")

    # ☠️ Place Traps
    for config in st.session_state.traps_config:
        amount = config.get('amount', 0)
        trap_type = config.get('trap_type', '')
        area = config.get('area', 'Random')

        if amount > 0 and trap_type:
            st.info(f"Attempting to place {amount} {trap_type} traps in {area}...")
            for _ in range(amount):
                current_grid, placed, placed_coords = place_item_on_grid(current_grid, TRAP_SYMBOL, 
                                                           item_details={'type': trap_type}, 
                                                           room_name=area, map_type=map_type)
                if placed:
                    st.session_state.item_locations_details[placed_coords] = {
                        'symbol': TRAP_SYMBOL, 
                        'description': f"Trap: {trap_type}"
                    }
                else:
                    st.warning(f"Failed to place a '{trap_type}' trap in '{area}'.")

    # ⬛ Place Voids/Pits
    for config in st.session_state.voids_config:
        name = config.get('name', 'Void')
        width = config.get('width', 2)
        height = config.get('height', 2)
        area = config.get('area', 'Random')

        st.info(f"Attempting to place void/pit '{name}' ({width}x{height}) in {area}...")
        current_grid, placed, placed_coords_list = place_void_on_grid(current_grid, width, height, name, map_elements_data, area)
        if placed:
            for coord in placed_coords_list:
                st.session_state.item_locations_details[coord] = {
                    'symbol': VOID_SYMBOL, 
                    'description': f"Void/Pit: {name} ({width}x{height})"
                }
        else:
            st.warning(f"Failed to place void/pit '{name}' in '{area}'.")

    return current_grid

def generate_map_grid(width, height, map_type, num_elements):
    grid = [[EMPTY_SPACE for _ in range(width)] for _ in range(height)]
    
    elements_data = {}
    placed_elements_count = 0

    if map_type in ["Dungeon", "Mountain", "Underdark", "Town"]: # Town also uses rooms/buildings
        grid, elements_data, placed_elements_count = _place_dungeon_elements(grid, num_elements, MIN_ROOM_WIDTH, MAX_ROOM_WIDTH, MIN_ROOM_HEIGHT, MAX_ROOM_HEIGHT, ROOM_FLOOR)
        if placed_elements_count > 1:
            grid = _connect_elements(grid, elements_data, ROOM_FLOOR)
    elif map_type == "Forest":
        grid = _generate_forest_terrain(grid, width, height)
        # Place clearings as elements
        grid, elements_data, placed_elements_count = _place_dungeon_elements(grid, num_elements, MIN_ROOM_WIDTH, MAX_ROOM_WIDTH, MIN_ROOM_HEIGHT, MAX_ROOM_HEIGHT, FOREST_CLEARING)
        if placed_elements_count > 1:
             grid = _connect_elements(grid, elements_data, HALLWAY) # Connect clearings with paths
    elif map_type in ["Desert", "Sea"]:
        # For these, we just place 'elements' which can be oases, islands, ruins etc.
        grid, elements_data, placed_elements_count = _place_dungeon_elements(grid, num_elements, MIN_ROOM_WIDTH, MAX_ROOM_WIDTH, MIN_ROOM_HEIGHT, MAX_ROOM_HEIGHT, ROOM_FLOOR)
        # No specific connection for these yet, could add paths/bridges later if needed

    # Place entrance after all main elements are placed and connected
    grid, elements_data = _place_entrance(grid, map_type, elements_data)

    return grid, elements_data

def draw_grid(grid, map_elements_data=None, show_grid_lines=True):
    if grid is None:
        return
    h, w = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(w * 0.4, h * 0.4))

    # Define colors for various map elements
    colors = {
        EMPTY_SPACE: '#ffffff',
        WALL: '#808080',
        DOOR: '#a0522d',
        ENTRANCE: '#90ee90',
        ROOM_FLOOR: '#d2b48c',
        HALLWAY: '#cd853f',
        FOREST_DENSE: '#228b22', # Ensure this is still the correct color after changing the symbol to '&'
        FOREST_LIGHT: '#6b8e23',
        FOREST_CLEARING: '#9acd32',
        TREASURE_SYMBOL: '#ffd700',
        TRAP_SYMBOL: '#ff0000',
        VOID_SYMBOL: '#4b0082',
        # Add these new entries with their direct color codes:
        PLAYER_SYMBOL: '#0000ff',    # Blue for player symbol
        ITEM_SYMBOL: '#ff8c00',      # Dark orange for generic items
        FOREST_TRAIL: '#8b4513',     # Saddle brown for forest trails
        EXIT_SYMBOL: '#8a2be2',      # Blue violet for exit symbol
        RIVER_TILE: '#00bfff',       # Deep sky blue for rivers
    }
    monster_base_color = '#ff4500'

    # Set edge color for grid lines
    edge_color = 'black' if show_grid_lines else 'none'

    # Draw grid cells
    for r in range(h):
        for c in range(w):
            cell_val = grid[r][c]
            color = colors.get(cell_val, EMPTY_SPACE) # Default to empty space color

            # Special handling for monster symbols (M-1, M-2, etc.)
            if cell_val.startswith(MONSTER_SYMBOL_PREFIX):
                color = monster_base_color
            
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor=edge_color, linewidth=0.5))

            # Add text labels for specific symbols
            if cell_val in [DOOR, ENTRANCE, TREASURE_SYMBOL, TRAP_SYMBOL, VOID_SYMBOL] or cell_val.startswith(MONSTER_SYMBOL_PREFIX):
                ax.text(c + 0.5, r + 0.5, cell_val, ha='center', va='center', color='black', fontsize=8, weight='bold')

    # Draw borders for named map elements (rooms, clearings, etc.)
    if map_elements_data:
        for element_name, data in map_elements_data.items():
            if data['coords']:
                all_r = [coord[0] for coord in data['coords']]
                all_c = [coord[1] for coord in data['coords']]
                min_r, max_r = min(all_r), max(all_r)
                min_c, max_c = min(all_c), max(all_c)
                
                # Draw a bounding box for the element
                rect = plt.Rectangle((min_c, min_r), max_c - min_c + 1, max_r - min_r + 1,
                                     fill=False, edgecolor='blue', linewidth=1.5)
                ax.add_patch(rect)
                # Optionally add text label for the element (e.g., Room 1, Clearing A)
                ax.text(min_c + 0.5, min_r + 0.5, element_name, ha='left', va='top', 
                        color='blue', fontsize=7, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0) # Invert y-axis to match grid[row][col]
    ax.set_xticks(np.arange(w + 1))
    ax.set_yticks(np.arange(h + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='both', length=0) # Hide tick marks
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free memory

def save_dungeon_to_html(grid, file_path, map_type, item_locations_details=None):
    h, w = len(grid), len(grid[0])
    base_css = """
    body { font-family: monospace; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; background-color: #333; }
    .grid-container { display: grid; grid-template-columns: repeat(%d, 25px); grid-template-rows: repeat(%d, 25px); border: 1px solid black; background-color: #ffffff; }
    .grid-cell { width: 25px; height: 25px; display: flex; justify-content: center; align-items: center; font-size: 12px; border: 0.5px solid #ccc; color: black; }
    .empty { background-color: #ffffff; } .wall { background-color: #808080; color: white; } .door { background-color: #a0522d; color: white; }
    .entrance { background-color: #90ee90; color: black; } .room-floor { background-color: #d2b48c; } .hallway { background-color: #cd853f; }
    .forest-dense { background-color: #228b22; color: white; } .forest-light { background-color: #6b8e23; } .clearing { background-color: #9acd32; }
    .treasure { background-color: #ffd700; color: black; font-weight: bold; } .trap { background-color: #ff0000; color: white; font-weight: bold; }
    .void { background-color: #4b0082; color: white; font-weight: bold; }
    .monster { background-color: #ff4500; color: white; font-weight: bold; }
    .river { background-color: #4682B4; } /* Steel Blue */
    .stream { background-color: #87CEEB; } /* Sky Blue */
    """ % (w, h)
    map_type_css = ""
    if map_type == "Sea": map_type_css = ".empty, .clearing { background-color: #ADD8E6; } .wall, .forest-dense { background-color: #4682B4; } .forest-light { background-color: #87CEEB; }"
    elif map_type == "Desert": map_type_css = ".empty, .clearing { background-color: #F5DEB3; } .wall, .forest-dense { background-color: #A0522D; } .forest-light { background-color: #D2B48C; }"
    elif map_type == "Mountain": map_type_css = ".empty { background-color: #A9A9A9; } .wall { background-color: #696969; } .room-floor { background-color: #8B4513; } .hallway { background-color: #BDB76B; }"
    elif map_type == "Underdark": map_type_css = ".empty { background-color: #36454F; } .wall { background-color: #2F4F4F; } .room-floor { background-color: #6A5ACD; } .hallway { background-color: #8A2BE2; }"
    elif map_type == "Town": map_type_css = ".empty { background-color: #A3C1AD; } .room-floor { background-color: #D3D3D3; } .wall { background-color: #8B4513; } .hallway { background-color: #696969; }" # Placeholder town colors
    elif map_type == "Forest": # Updated Forest CSS for new tiles
        map_type_css = ".empty { background-color: #90EE90; } .wall { background-color: #228B22; } .room-floor, .clearing { background-color: #6B8E23; }"
        map_type_css += " .river { background-color: #4682B4; } .stream { background-color: #87CEEB; }"
    
    html_content = f"<!DOCTYPE html><html><head><title>Generated Map</title><style>{base_css}{map_type_css}</style></head><body><div class=\"grid-container\">"
    
    # Ensure item_locations_details is a dict for lookup
    if item_locations_details is None:
        item_locations_details = {}

    # Use itertools.product for a more Pythonic and slightly more efficient loop
    for r_idx, c_idx in itertools.product(range(h), range(w)):
        cell_val = grid[r_idx][c_idx]
        css_class, display_text = "", cell_val
        
        # Determine CSS class based on cell value
        if cell_val == EMPTY_SPACE: css_class = "empty"
        elif cell_val == WALL: css_class = "wall"
        elif cell_val == DOOR: css_class = "door"
        elif cell_val == ENTRANCE: css_class = "entrance"
        elif cell_val == ROOM_FLOOR: css_class = "room-floor"
        elif cell_val == HALLWAY: css_class = "hallway"
        elif cell_val == FOREST_DENSE: css_class = "forest-dense"
        elif cell_val == FOREST_LIGHT: css_class = "forest-light"
        elif cell_val == FOREST_CLEARING: css_class = "clearing"
        elif cell_val == TREASURE_SYMBOL: css_class = "treasure"
        elif cell_val == TRAP_SYMBOL: css_class = "trap"
        elif cell_val == VOID_SYMBOL: css_class = "void"
        elif cell_val == RIVER_TILE: css_class = "river" # New
        elif cell_val == STREAM_TILE: css_class = "stream" # New
        elif cell_val.startswith(MONSTER_SYMBOL_PREFIX): css_class = "monster"
        else: css_class = "empty" # Fallback

        # --- MODIFICATION START ---
        # Add title attribute for hover text
        title_text = ""
        if (r_idx, c_idx) in item_locations_details:
            item_info = item_locations_details[(r_idx, c_idx)]
            
            # Customize the tooltip based on item type
            if item_info.get('type') == 'monster' and 'name' in item_info:
                title_text = f"Monster: {item_info['name']}"
            elif item_info.get('type') == 'treasure' and 'name' in item_info:
                title_text = f"Treasure: {item_info['name']}"
            elif item_info.get('type') == 'trap' and 'name' in item_info:
                title_text = f"Trap: {item_info['name']}"
            elif item_info.get('type') == 'void' and 'name' in item_info:
                title_text = f"Void: {item_info['name']}"
            elif 'description' in item_info: # Fallback if 'name' isn't available but 'description' is
                title_text = item_info['description']
            
            # Optional: Add coordinates to the tooltip if you still want them
            # if title_text: # Only add if title_text is not empty
            #     title_text += f" ({r_idx}, {c_idx})"

        # Escape HTML special characters in title_text to prevent issues
        safe_title_text = html.escape(title_text)

        html_content += f'<div class="grid-cell {css_class}" title="{safe_title_text}">{display_text}</div>'
        # --- MODIFICATION END ---
        
    html_content += "</div></body></html>"
    with open(file_path, "w") as f:
        f.write(html_content)
    # Removed st.success from here, as Streamlit calls should be in the app's main flow
    # st.success(f"Map saved as HTML: {file_path}")

# --- Streamlit App ---

# --- GLOBAL SESSION STATE INITIALIZATION ---
# This block ensures essential data and variables are set up when the app first loads

# Initialize monster data
if "monsters_df" not in st.session_state:
    st.session_state.monsters_df = load_monsters() # Load monsters once, globally and cached

# Initialize campaign data
if "campaigns" not in st.session_state:
    st.session_state.campaigns = load_campaigns()

# Initialize the currently selected campaign (or None if no campaigns yet)
if "current_campaign_index" not in st.session_state:
    # If there are campaigns, default to the first one (index 0). Otherwise, -1 means no campaign selected.
    st.session_state.current_campaign_index = 0 if st.session_state.campaigns else -1

# This will hold the details of the actively selected campaign for editing
if "selected_campaign_details" not in st.session_state:
    if st.session_state.current_campaign_index != -1:
        st.session_state.selected_campaign_details = st.session_state.campaigns[st.session_state.current_campaign_index]
    else:
        # If no campaigns exist, initialize with empty details for a 'New Campaign' state
        st.session_state.selected_campaign_details = {
            "Campaign Name": "",
            "Game System": "",
            "DM Name": "",
            "Player Names": "", # Will store as a single string for now for high-level overview
            "Description": "",
            "Status": "Active",
            "Lore": "",
            "players": [], # List of dictionaries for detailed player characters
            "npcs": [], # List of dictionaries for NPCs
            "monsters_in_campaign": [], # List of dictionaries for campaign-specific monsters
            "plot_lines": [] # List of dictionaries for plots/quests
        }

# Initialize map generation parameters and other configs
if "map_grid" not in st.session_state:
    st.session_state.map_grid = None
if "map_elements_data" not in st.session_state:
    st.session_state.map_elements_data = {}
if "generated_map_type" not in st.session_state:
    st.session_state.generated_map_type = "Dungeon" # This was a default, ensure it's set
if "base_map_grid_clean" not in st.session_state:
    st.session_state.base_map_grid_clean = None
if "base_map_elements_data_clean" not in st.session_state:
    st.session_state.base_map_elements_data_clean = {}

# Initialize placement config lists
if "monsters_config" not in st.session_state:
    st.session_state.monsters_config = []
if "treasures_config" not in st.session_state:
    st.session_state.treasures_config = []
if "traps_config" not in st.session_state:
    st.session_state.traps_config = []
if "voids_config" not in st.session_state:
    st.session_state.voids_config = []
if "item_locations_details" not in st.session_state:
    st.session_state.item_locations_details = {}

# Initialize current encounter DataFrame (if not already handled by load_encounter_data within the tab logic)
if 'current_encounter_df' not in st.session_state:
    # Define required columns for the encounter DataFrame
    required_encounter_columns = ["Name", "Type", "Initiative", "Max HP", "Current HP", "AC", "Conditions"]
    st.session_state.current_encounter_df = pd.DataFrame(columns=required_encounter_columns)

# --- Branding --- 
# Use st.columns to place logos side-by-side
# Adjust the column ratios (e.g., [1, 1] for equal width, or [0.7, 1.3] if one logo needs more space)
logo_col1, logo_col2 = st.columns([1, 1]) 

with logo_col1:
    st.image("https://raw.githubusercontent.com/Mugmugmug81/keeping_up/main/1000210637.png", 
             caption="2Ones", 
             width=200) # Set a fixed width for the first logo
    
with logo_col2:
    st.image("https://raw.githubusercontent.com/Mugmugmug81/keeping_up/main/1000210636.png", 
             caption="Dungeon Master!", 
             width=200) # Set a fixed width for the second logo

st.markdown("<br>", unsafe_allow_html=True) # Add some space below logos

# --- Main App Title and Subtitle (always visible) ---
st.markdown("<h1 style='text-align: center; color: #8B4513; font-family: Georgia, serif;'>⚔️ The Dungeon Master's Codex (WIP) 📜</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #A0522D; font-family: sans-serif;'>Your All-in-One Companion for Epic Adventures! (WIP)</h3>", unsafe_allow_html=True)

st.markdown("---") # Visual separator below the main app header

# The rest of your code (selected_tab, if selected_tab == "Home", etc.) remains below this block.

# --- Sidebar Navigation ---
st.sidebar.title("Choose Dungeon Master!")
selected_tab = st.sidebar.radio(
    "Build!",
    options=["Bestiary", "Map Generator", "Map Library", "Campaign Manager", "Encounter Command Center"],
    index=0,
    key="main_navigation_radio_unique" 
)

# === Conditional Tab Rendering ===
if selected_tab == "Map Generator":
    st.header("Create Thy World")

    # --- Initialize Map Generation Parameters in Session State (Add this near the top of your script, or before this tab) ---
    if "map_type" not in st.session_state:
        st.session_state.map_type = "Dungeon" # Default value
    if "map_width" not in st.session_state:
        st.session_state.map_width = 20 # Default value
    if "map_height" not in st.session_state:
        st.session_state.map_height = 20 # Default value
    if "num_elements" not in st.session_state:
        st.session_state.num_elements = 1 # Default value

    # --- Your existing Save Map Section (no changes needed here for the defaulting issue) ---
    st.subheader("Save Current Map")
    if st.session_state.map_grid is not None:
        map_name = st.text_input("Enter map name to save:", value=f"Map_{st.session_state.generated_map_type}_{len(os.listdir(MAPS_FOLDER)) + 1}", key="map_name_input_top")
        if st.button("Save Map", key="save_map_btn_top") and map_name:
            file_path = os.path.join(MAPS_FOLDER, f"{map_name}.html")
            save_dungeon_to_html(st.session_state.map_grid, file_path, st.session_state.generated_map_type, st.session_state.item_locations_details)
            st.success(f"Map saved as HTML: {file_path}")
            with open(file_path, "rb") as f:
                st.download_button(
                    label="Download Map HTML",
                    data=f,
                    file_name=f"{map_name}.html",
                    mime="text/html",
                    key="download_map_html_btn"
                )
    else:
        st.info("Generate a map below to enable saving.")
    st.markdown("---")

    # --- IMPORTANT: Re-evaluate this 'monster_names' block ---
    # This block is still deriving 'monster_names' from 'all_monsters.keys()',
    # which gives column names, not monster names.
    # If this 'monster_names' is used anywhere *outside* the Monster Configuration
    # (where we previously fixed it), it needs to be updated here too.
    # If it's *only* used in the Monster Configuration section you provided earlier,
    # then this whole block here might be redundant and can be removed.
    all_monsters = load_monsters() # Assuming this loads into a DataFrame
    if not all_monsters.empty:
        # This part is still incorrect for getting monster names from a DataFrame
        # You'd typically want: monster_name_options = sorted(all_monsters['Name'].dropna().unique().tolist())
        monster_names = sorted(list(all_monsters.keys())) # This will give column names
        if "None" not in monster_names:
            monster_names.insert(0, "None")
    else:
        monster_names = ["None"]
    # --- End of monster_names re-evaluation ---


    with st.expander("Map Generation Settings", expanded=True):
        # Assign widget outputs directly to session_state
        # Use session_state values for 'value' or 'index'
        
        # Determine the correct index for st.radio
        map_type_options = ["Dungeon", "Forest", "Desert", "Sea", "Mountain", "Underdark", "Town"]
        current_map_type_index = map_type_options.index(st.session_state.map_type) if st.session_state.map_type in map_type_options else 0

        st.session_state.map_type = st.radio(
            "Map Type", 
            map_type_options, # Use the list of options
            horizontal=True, 
            key="map_type_radio",
            index=current_map_type_index # Set initial selection from session state
        )
        st.session_state.map_width = st.slider(
            "Map Width", 
            min_value=MIN_MAP_REQUIRED_WIDTH, max_value=50, 
            value=st.session_state.map_width, # Set initial value from session state
            key="map_width_slider"
        )
        st.session_state.map_height = st.slider(
            "Map Height", 
            min_value=MIN_MAP_REQUIRED_HEIGHT, max_value=50, 
            value=st.session_state.map_height, # Set initial value from session state
            key="map_height_slider"
        )
        st.session_state.num_elements = st.slider(
            "Number of Rooms/Locations/Buildings", 
            min_value=0, max_value=MAX_ELEMENTS, 
            value=st.session_state.num_elements, # Set initial value from session state
            key="num_elements_slider"
        )
        
        # New: Button to generate map
        if st.button("Generate Base Map", key="generate_map_btn_settings"):
            # Use values from session_state for the generation logic
            if st.session_state.map_width < MIN_MAP_REQUIRED_WIDTH or st.session_state.map_height < MIN_MAP_REQUIRED_HEIGHT:
                st.error(f"Map dimensions must be at least {MIN_MAP_REQUIRED_WIDTH}x{MIN_MAP_REQUIRED_HEIGHT} to ensure room placement.")
            else:
                # Reset item configurations and details when generating a new base map
                st.session_state.monsters_config = []
                st.session_state.treasures_config = []
                st.session_state.traps_config = []
                st.session_state.voids_config = []
                st.session_state.item_locations_details = {} # Clear item details
                
                # Pass parameters from session_state to the generation function
                base_grid, elements_data = generate_map_grid(
                    st.session_state.map_width, 
                    st.session_state.map_height, 
                    st.session_state.map_type, 
                    st.session_state.num_elements
                )
                st.session_state.map_grid = base_grid
                st.session_state.map_elements_data = elements_data
                st.session_state.generated_map_type = st.session_state.map_type # Store the type that was generated

                # --- IMPORTANT FIX HERE: Store the clean base map and its elements ---
                st.session_state.base_map_grid_clean = [row[:] for row in base_grid] # Store a deep copy
                st.session_state.base_map_elements_data_clean = {k: v for k, v in elements_data.items()} # Store a copy

                st.success(f"Base {st.session_state.map_type} map generated! Now configure and place items.")
                st.rerun() # Rerun to update the map_elements_data in session_state and UI

    # --- Display/Name Map Elements ---
    if st.session_state.map_elements_data:
        with st.expander("Name Map Elements (Rooms/Clearings/Buildings)", expanded=True):
            st.info("You can rename the detected elements below. Changes will apply when items are placed on the map.")
            new_map_elements_data = {}
            sorted_element_names = sorted(st.session_state.map_elements_data.keys())

            for original_name in sorted_element_names:
                data = st.session_state.map_elements_data[original_name]
                new_name = st.text_input(f"Rename '{original_name}' (Cells: {len(data['coords'])})", 
                                         value=original_name, key=f"rename_element_{original_name}")
                new_map_elements_data[new_name] = data
            
            if set(new_map_elements_data.keys()) != set(st.session_state.map_elements_data.keys()):
                    st.session_state.map_elements_data = new_map_elements_data
                    st.success("Map element names updated. Place items on map to see changes reflected.")
    else:
        st.info("Generate a map to see and name its elements.")
    st.markdown("---")

    # --- Item Placement Configurations ---
    st.subheader("Item Placement Configurations")

    # Get available areas (rooms, clearings, entrance, or "Random")
    area_options = sorted(list(st.session_state.map_elements_data.keys()) + ["Random"]) if st.session_state.map_elements_data else ["Random"]

    # New: Clear Item Configurations Button
    if st.button("🗑️ Clear All Item Configurations", key="clear_all_items_btn", help="Clears all monster, treasure, trap, and void configurations."):
        st.session_state.monsters_config = []
        st.session_state.treasures_config = []
        st.session_state.traps_config = []
        st.session_state.voids_config = []
        st.session_state.item_locations_details = {}
        st.success("All item configurations cleared!")
        st.rerun()

# 👹 Monster Configuration
    st.markdown("### 👹 Monsters")
    with st.container(border=True):
        # Initialize monsters_config if it doesn't exist
        if "monsters_config" not in st.session_state:
            st.session_state.monsters_config = []

        # Prepare the list of monster names for the selectbox
        # Ensure monsters_df is loaded from the Bestiary tab and is not empty
        if "monsters_df" not in st.session_state or st.session_state.monsters_df.empty:
            # If no monsters loaded or bestiary is empty, only 'None' is available
            monster_name_options = ["None"]
        else:
            # Get actual monster names from the 'Name' column of the DataFrame
            # Use .dropna() to remove any NaN values (empty cells) and .unique() for distinct names
            monster_name_options = sorted(st.session_state.monsters_df['Name'].dropna().unique().tolist())
            monster_name_options.insert(0, "None") # Ensure 'None' is always the first option

        if st.button("➕ Add Monster Group", key="add_monster_group_btn"):
            st.session_state.monsters_config.append({'amount': 1, 'monster_type': 'None', 'area': 'Random'})
        
        for i, config in enumerate(st.session_state.monsters_config):
            col_m1, col_m2, col_m3, col_m4 = st.columns([1, 2, 2, 0.5])
            with col_m1:
                config['amount'] = st.number_input(f"Amount", min_value=0, value=config.get('amount', 1), key=f"monster_amount_{i}")
            with col_m2:
                current_monster_type = config.get('monster_type', 'None')
                # Use monster_name_options generated above
                config['monster_type'] = st.selectbox(
                    f"Monster Type", 
                    monster_name_options, # Use the correctly populated list
                    index=monster_name_options.index(current_monster_type) if current_monster_type in monster_name_options else 0, 
                    key=f"monster_type_{i}"
                )
            with col_m3:
                current_area = config.get('area', "Random")
                config['area'] = st.selectbox(f"Placement Area", area_options, 
                                                index=area_options.index(current_area) if current_area in area_options else 0, 
                                                key=f"monster_area_{i}")
            with col_m4:
                st.write("") # Spacer for alignment
                if st.button("🗑️", key=f"remove_monster_{i}"):
                    st.session_state.monsters_config.pop(i)
                    st.rerun() # Rerun to update the list after removal

    # 💰 Treasure Configuration
    st.markdown("### 💰 Treasure") # Renamed to Treasure
    with st.container(border=True):
        if st.button("➕ Add Treasure Item", key="add_treasure_item_btn"):
            st.session_state.treasures_config.append({'amount': 1, 'amount_type': 'pieces', 'treasure_type': '', 'area': 'Random'})
        
        for i, config in enumerate(st.session_state.treasures_config):
            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns([1, 2, 2, 2, 0.5])
            with col_t1:
                config['amount'] = st.number_input(f"Amount", min_value=0, value=config.get('amount', 1), key=f"treasure_amount_{i}")
            with col_t2:
                config['amount_type'] = st.selectbox(f"Amount Type", ["pieces", "total value"], 
                                                    index=["pieces", "total value"].index(config.get('amount_type', 'pieces')), 
                                                    key=f"treasure_amount_type_{i}")
            with col_t3:
                config['treasure_type'] = st.text_input(f"Treasure Type (e.g., gold, jewels)", value=config.get('treasure_type', ''), key=f"treasure_type_{i}")
            with col_t4:
                current_area = config.get('area', "Random")
                config['area'] = st.selectbox(f"Placement Area", area_options, 
                                            index=area_options.index(current_area) if current_area in area_options else 0, 
                                            key=f"treasure_area_{i}")
            with col_t5:
                st.write("") # Spacer for alignment
                if st.button("🗑️", key=f"remove_treasure_{i}"):
                    st.session_state.treasures_config.pop(i)
                    st.rerun()

    # ☠️ Trap Configuration
    st.markdown("### ☠️ Traps")
    with st.container(border=True):
        if st.button("➕ Add Trap", key="add_trap_btn"):
            st.session_state.traps_config.append({'amount': 1, 'trap_type': '', 'area': 'Random'})
        
        for i, config in enumerate(st.session_state.traps_config):
            col_x1, col_x2, col_x3, col_x4 = st.columns([1, 3, 2, 0.5])
            with col_x1:
                config['amount'] = st.number_input(f"Amount", min_value=0, value=config.get('amount', 1), key=f"trap_amount_{i}")
            with col_x2:
                config['trap_type'] = st.text_input(f"Trap Type (e.g., pit, dart, poison)", value=config.get('trap_type', ''), key=f"trap_type_{i}")
            with col_x3:
                current_area = config.get('area', "Random")
                config['area'] = st.selectbox(f"Placement Area", area_options, 
                                            index=area_options.index(current_area) if current_area in area_options else 0, 
                                            key=f"trap_area_{i}")
            with col_x4:
                st.write("") # Spacer for alignment
                if st.button("🗑️", key=f"remove_trap_{i}"):
                    st.session_state.traps_config.pop(i)
                    st.rerun()

    # ⬛ Void/Pit Configuration (New Section)
    st.markdown("### ⬛ Voids / Pits")
    with st.container(border=True):
        if st.button("➕ Add Void/Pit", key="add_void_btn"):
            st.session_state.voids_config.append({'name': 'New Void', 'width': 2, 'height': 2, 'area': 'Random'})
        
        for i, config in enumerate(st.session_state.voids_config):
            col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns([2, 1, 1, 2, 0.5])
            with col_v1:
                config['name'] = st.text_input(f"Void/Pit Name", value=config.get('name', f'Void {i+1}'), key=f"void_name_{i}")
            with col_v2:
                config['width'] = st.number_input(f"Width", min_value=1, value=config.get('width', 2), key=f"void_width_{i}")
            with col_v3:
                config['height'] = st.number_input(f"Height", min_value=1, value=config.get('height', 2), key=f"void_height_{i}")
            with col_v4:
                current_area = config.get('area', "Random")
                config['area'] = st.selectbox(f"Placement Area", area_options, 
                                            index=area_options.index(current_area) if current_area in area_options else 0, 
                                            key=f"void_area_{i}")
            with col_v5:
                st.write("") # Spacer for alignment
                if st.button("🗑️", key=f"remove_void_{i}"):
                    st.session_state.voids_config.pop(i)
                    st.rerun()


    # Map Generator Button (moved to Item Placement section to emphasize this is after map generation)
    if st.button("Place Items on Map", key="place_items_on_map_btn"):
        if st.session_state.base_map_grid_clean is None or st.session_state.base_map_elements_data_clean is None:
            st.warning("Please generate a base map first using 'Generate Base Map' in the settings above to establish the clean map structure.")
        else:
            # Use the stored clean base grid and element data as the starting point
            initial_grid_for_items = [row[:] for row in st.session_state.base_map_grid_clean]
            # Use the *current* map_elements_data which includes user-renamed elements
            current_map_elements_for_items = st.session_state.map_elements_data 

            final_grid = place_all_items_on_map(initial_grid_for_items, current_map_elements_for_items, st.session_state.generated_map_type)
            st.session_state.map_grid = final_grid # Update the grid with placed items
            st.success(f"Items placed on {st.session_state.generated_map_type} map!")
            st.rerun()


    if st.session_state.map_grid is not None:
        st.subheader("Generated Map")
        
        # Grid Overlay Toggle
        show_grid_lines_toggle = st.checkbox("Show Grid Lines", value=True, key="show_grid_lines_toggle")
        
        # Pass the toggle state to draw_grid
        draw_grid(st.session_state.map_grid, st.session_state.map_elements_data, show_grid_lines_toggle)


elif selected_tab == "Map Library":
    st.header("📁 Map Library")
    st.write("Browse and load your saved maps here.")
    
    saved_maps = glob.glob(os.path.join(MAPS_FOLDER, "*.html"))
    if not saved_maps:
        st.info("No maps saved yet. Generate a map first!")
    else:
        map_files = [os.path.basename(f) for f in saved_maps]
        selected_map_file = st.selectbox("Select a map to load:", map_files, key="map_library_selector")

        if selected_map_file:
            map_full_path = os.path.join(MAPS_FOLDER, selected_map_file)
            try:
                with open(map_full_path, "r", encoding="utf-8") as f: # Added encoding for robustness
                    map_html_content = f.read()
                
                # Encode the HTML content to Base64
                b64_html = base64.b64encode(map_html_content.encode("utf-8")).decode("utf-8")
                
                # Construct the data URI
                data_uri = f"data:text/html;base64,{b64_html}"

                st.markdown(
                    f'<iframe src="{data_uri}" width="100%" height="600px" style="border:none;"></iframe>',
                    unsafe_allow_html=True
                )
                st.info(f"Loaded map: {selected_map_file}")

                if st.button(f"🗑️ Delete {selected_map_file}", key=f"delete_map_{selected_map_file}"):
                    os.remove(map_full_path)
                    st.success(f"Deleted {selected_map_file}")
                    st.rerun() # Refresh the list of maps

            except Exception as e:
                st.error(f"Error loading map: {e}")


# Assuming your tab definitions are something like this (adjust as needed if you have more tabs):
# bestiary_tab, map_gen_tab, encounter_tab = st.tabs(["Bestiary", "Map Generator", "Encounter Command Center"])

# Within your main Streamlit app layout:
elif selected_tab == "Bestiary": # Assuming this is how you structure your tabs
    st.header("📚 Monster Bestiary")
    st.write("Here you can view, add, edit, and delete monsters for your campaigns.")

        # Use st.data_editor for interactive table editing
    edited_monsters_df = st.data_editor(
        st.session_state.monsters_df,
        column_config={
            "Name": st.column_config.TextColumn("Name", required=True),
            "Type": st.column_config.TextColumn("Type"),
            "CR": st.column_config.TextColumn("Challenge Rating"), # Keep as Text for now due to "1/2", "1/4" etc.
            "HP": st.column_config.NumberColumn("Hit Points", min_value=0, step=1), # Changed to NumberColumn
            "AC": st.column_config.NumberColumn("Armor Class", min_value=0, step=1), # Changed to NumberColumn
            "Speed": st.column_config.TextColumn("Speed"),
            "Saves": st.column_config.TextColumn("Saving Throws"),
            "Skills": st.column_config.TextColumn("Skills"),
            "Damage Immunities": st.column_config.TextColumn("Damage Immunities"),
            "Damage Resistances": st.column_config.TextColumn("Damage Resistances"),
            "Damage Vulnerabilities": st.column_config.TextColumn("Damage Vulnerabilities"),
            "Condition Immunities": st.column_config.TextColumn("Condition Immunities"),
            "Senses": st.column_config.TextColumn("Senses"),
            "Languages": st.column_config.TextColumn("Languages"),
            "Challenge": st.column_config.TextColumn("Challenge"), 
            "Proficiency Bonus": st.column_config.NumberColumn("Proficiency Bonus", min_value=0, step=1), # Changed to NumberColumn
            "Actions": st.column_config.TextColumn("Actions"),
            "Legendary Actions": st.column_config.TextColumn("Legendary Actions"),
            "Reactions": st.column_config.TextColumn("Reactions"),
            "Description": st.column_config.TextColumn("Description")
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="bestiary_editor",
        on_change=update_bestiary_df_callback
    )

    st.info("Monster changes are saved automatically as you edit.")
    st.markdown("---")
    st.subheader("💡 How to Use:")
    st.write("1. **To Search, Add, Download, or Enlarge the Table:** Utilize the menu at the top of the table.")
    st.write("2. **View & Edit:** Click directly on any cell in the table to modify its content.")
    st.write("3. **Delete Monster:** Select the row to be deleted, then click on Delete at the top right of the table.")
    st.write("4. **Save Changes:** After making any additions, edits, or deletions, **click the 'Save Bestiary Changes' button** to permanently update your database")

elif selected_tab == "Campaign Manager":
    st.header("📓 Campaign Manager")
    st.write("Manage your campaigns, including high-level details, players, NPCs, monsters, and plot lines.")

    # --- Campaign Selection / Creation ---
    st.subheader("Select or Create Campaign")
    campaign_names = [campaign["Campaign Name"] for campaign in st.session_state.campaigns] if st.session_state.campaigns else []
    
    # Add a "New Campaign" option if no campaigns exist or user wants to create one
    display_options = ["Create New Campaign..."] + campaign_names

    # Determine initial selection for the selectbox
    initial_select_index = 0
    if st.session_state.current_campaign_index != -1 and st.session_state.current_campaign_index < len(campaign_names):
        # If a campaign is currently selected and valid, find its index in display_options
        try:
            initial_select_index = display_options.index(st.session_state.campaigns[st.session_state.current_campaign_index]["Campaign Name"])
        except ValueError:
            # Fallback if somehow not found, or if campaign was just deleted
            initial_select_index = 0
    
    selected_campaign_display_name = st.selectbox(
        "Choose an existing campaign or create a new one:",
        options=display_options,
        index=initial_select_index,
        key="campaign_selector"
    )

    if selected_campaign_display_name == "Create New Campaign...":
        # Reset selected campaign details for a new, empty campaign
        st.session_state.selected_campaign_details = {
            "Campaign Name": "",
            "Game System": "",
            "DM Name": "",
            "Player Names": "",
            "Description": "",
            "Status": "Active",
            "Lore": "",
            "players": [],
            "npcs": [],
            "monsters_in_campaign": [],
            "plot_lines": []
        }
        st.session_state.current_campaign_index = -1 # Indicate no specific campaign selected yet
        st.info("Fill in the details below to create a new campaign.")
    else:
        # User selected an existing campaign, find its index and update session state
        try:
            campaign_index = campaign_names.index(selected_campaign_display_name)
            if st.session_state.current_campaign_index != campaign_index:
                st.session_state.current_campaign_index = campaign_index
                st.session_state.selected_campaign_details = st.session_state.campaigns[campaign_index]
                st.rerun() # Rerun to load selected campaign details into the forms
        except ValueError:
            st.error("Selected campaign not found. Please select another or create a new one.")
            st.session_state.current_campaign_index = -1
            # Reset to 'Create New Campaign...' state if selection fails
            st.session_state.selected_campaign_details = {
                "Campaign Name": "", "Game System": "", "DM Name": "", "Player Names": "",
                "Description": "", "Status": "Active", "Lore": "",
                "players": [], "npcs": [], "monsters_in_campaign": [], "plot_lines": []
            }


    # --- Campaign Details Editor ---
    st.subheader("Campaign Details")
    with st.form("campaign_details_form", clear_on_submit=False):
        # Top-level campaign attributes
        st.session_state.selected_campaign_details["Campaign Name"] = st.text_input(
            "Campaign Name", value=st.session_state.selected_campaign_details.get("Campaign Name", ""), key="campaign_name"
        )
        st.session_state.selected_campaign_details["Game System"] = st.text_input(
            "Game System", value=st.session_state.selected_campaign_details.get("Game System", ""), key="game_system"
        )
        st.session_state.selected_campaign_details["DM Name"] = st.text_input(
            "DM Name", value=st.session_state.selected_campaign_details.get("DM Name", ""), key="dm_name"
        )
        st.session_state.selected_campaign_details["Player Names"] = st.text_input(
            "Player Names (comma-separated for overview)", value=st.session_state.selected_campaign_details.get("Player Names", ""), key="player_names_overview"
        )
        st.session_state.selected_campaign_details["Status"] = st.selectbox(
            "Status", options=["Active", "On Hold", "Completed", "Archived"],
            index=["Active", "On Hold", "Completed", "Archived"].index(st.session_state.selected_campaign_details.get("Status", "Active")),
            key="campaign_status"
        )
        st.session_state.selected_campaign_details["Description"] = st.text_area(
            "Description", value=st.session_state.selected_campaign_details.get("Description", ""), key="campaign_description"
        )
        st.session_state.selected_campaign_details["Lore"] = st.text_area(
            "Lore & Background", value=st.session_state.selected_campaign_details.get("Lore", ""), key="campaign_lore"
        )

        submitted = st.form_submit_button("💾 Save Campaign Details")
        if submitted:
            # Check if creating a new campaign or updating existing
            if st.session_state.current_campaign_index == -1: # It's a new campaign
                if st.session_state.selected_campaign_details["Campaign Name"]:
                    # Add the new campaign to the list
                    st.session_state.campaigns.append(st.session_state.selected_campaign_details)
                    st.session_state.current_campaign_index = len(st.session_state.campaigns) - 1
                    save_campaigns(st.session_state.campaigns)
                    st.success(f"New campaign '{st.session_state.selected_campaign_details['Campaign Name']}' created and saved!")
                    st.rerun() # Rerun to refresh the selector and detail forms
                else:
                    st.warning("Please enter a 'Campaign Name' to create a new campaign.")
            else: # Updating an existing campaign
                st.session_state.campaigns[st.session_state.current_campaign_index] = st.session_state.selected_campaign_details
                save_campaigns(st.session_state.campaigns)
                st.success(f"Campaign '{st.session_state.selected_campaign_details['Campaign Name']}' details updated!")
                st.rerun() # Rerun to reflect changes immediately in the selector if name changed


    st.markdown("---")

    # --- Nested Data Editors for Players, NPCs, Monsters, Plot Lines ---
    if st.session_state.current_campaign_index != -1 or st.session_state.selected_campaign_details.get("Campaign Name"):
        # Only show these if a campaign is selected or a new one is being filled out
        st.subheader("Campaign Elements")

        # Players
        # --- Add DM/Main Player ---
        st.markdown("###### Add Player")
        col_dm_player_1, col_dm_player_2 = st.columns([3, 1])
        with col_dm_player_1:
            dm_player_name_input = st.text_input(
                "Enter Player Name:",
                value=st.session_state.get("dm_player_name_input_default", ""), # Persist input
                key="dm_player_name_input"
            )
        with col_dm_player_2:
            st.write("") # Spacer for alignment
            if st.button("➕ Add Player", key="add_dm_player_btn"):
                if dm_player_name_input:
                    new_player = {
                        "Name": dm_player_name_input,
                        "Character Name": "",
                        "Race": "",
                        "Class": "",
                        "Notes": ""
                    }
                    # Ensure the players list exists, then append
                    if "players" not in st.session_state.selected_campaign_details:
                        st.session_state.selected_campaign_details["players"] = []
                    
                    # Prevent duplicate entries if the name already exists
                    current_player_names = [p.get("Name") for p in st.session_state.selected_campaign_details["players"]]
                    if dm_player_name_input not in current_player_names:
                        st.session_state.selected_campaign_details["players"].append(new_player)
                        st.session_state.dm_player_name_input_default = dm_player_name_input # Store for persistence
                        st.success(f"'{dm_player_name_input}' added to players!")
                        st.rerun() # Rerun to update the data editor
                    else:
                        st.warning(f"'{dm_player_name_input}' is already in the players list.")
                else:
                    st.warning("Please enter a name for the DM/Main Player.")
        st.markdown("---") # Separator
        st.markdown("##### 👥 Players")
        players_data = st.session_state.selected_campaign_details.get("players", [])
        edited_players_df = st.data_editor(
            pd.DataFrame(players_data) if players_data else pd.DataFrame(columns=DEFAULT_PLAYER_COLUMNS),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Player Name", required=True),
                "Character Name": st.column_config.TextColumn("Character Name"),
                "Race": st.column_config.TextColumn("Race"),
                "Class": st.column_config.TextColumn("Class"),
                "Notes": st.column_config.TextColumn("Notes")
            },
            hide_index=True,
            key="players_editor", # This key is correct for the callback
            # --- PLACE THE ON_CHANGE CALLBACK HERE ---
            on_change=lambda: update_campaign_sublist_callback("players") # Pass "players" as the key
        )

        # --- UPDATE THIS INFO MESSAGE ---
        st.info("Player changes are saved automatically as you edit.")


        # NPCs
        st.markdown("##### 🤝 NPCs")
        DEFAULT_NPC_COLUMNS = ["Name", "Role", "Location", "Status", "Notes"]
        npcs_data = st.session_state.selected_campaign_details.get("npcs", [])
        edited_npcs_df = st.data_editor(
            pd.DataFrame(npcs_data) if npcs_data else pd.DataFrame(columns=DEFAULT_NPC_COLUMNS),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("NPC Name", required=True),
                "Role": st.column_config.TextColumn("Role"),
                "Location": st.column_config.TextColumn("Location"),
                "Status": st.column_config.TextColumn("Status"),
                "Notes": st.column_config.TextColumn("Notes")
            },
            hide_index=True,
            key="npcs_editor", # This key is already correct
            # --- ADD THE ON_CHANGE CALLBACK HERE ---
            on_change=lambda: update_campaign_sublist_callback("npcs") # Pass "npcs" as the key
        )

        st.info("NPC changes are saved automatically as you edit.")

        # Monsters in Campaign
        st.markdown("##### 👹 Campaign Monsters")
        # --- Add Monster from Bestiary ---
        st.markdown("###### Add from Bestiary")
        col_add_mon1, col_add_mon2 = st.columns([3, 1])

        # Prepare monster options for the selectbox
        bestiary_monster_names = []
        if not st.session_state.monsters_df.empty:
            bestiary_monster_names = sorted(st.session_state.monsters_df['Name'].dropna().unique().tolist())
        
        with col_add_mon1:
            selected_bestiary_monster = st.selectbox(
                "Select a monster from your Bestiary:",
                options=["--- Select a Monster ---"] + bestiary_monster_names,
                key="select_bestiary_monster_to_add"
            )
        
        with col_add_mon2:
            st.write("") # Spacer for alignment
            if st.button("➕ Add Monster", key="add_selected_bestiary_monster"):
                if selected_bestiary_monster and selected_bestiary_monster != "--- Select a Monster ---":
                    # Find the monster details in the bestiary DataFrame
                    monster_details = st.session_state.monsters_df[
                        st.session_state.monsters_df['Name'] == selected_bestiary_monster
                    ].iloc[0] # Get the first (and likely only) matching row

                    new_campaign_monster = {
                        "Name": monster_details.get("Name", selected_bestiary_monster),
                        "Type": monster_details.get("Type", "N/A"),
                        "CR": monster_details.get("CR", "N/A"),
                        "Notes": "" # Start with empty notes for campaign-specific context
                    }
                    
                    # Ensure the list exists, then append
                    if "monsters_in_campaign" not in st.session_state.selected_campaign_details:
                        st.session_state.selected_campaign_details["monsters_in_campaign"] = []
                    
                    st.session_state.selected_campaign_details["monsters_in_campaign"].append(new_campaign_monster)
                    st.success(f"'{selected_bestiary_monster}' added to campaign monsters!")
                    st.rerun() # Rerun to update the data editor with the new entry
                else:
                    st.warning("Please select a monster from the bestiary to add.")
        st.markdown("---") # Separator
        
        DEFAULT_CAMPAIGN_MONSTER_COLUMNS = ["Name", "Type", "CR", "Notes"]
        campaign_monsters_data = st.session_state.selected_campaign_details.get("monsters_in_campaign", [])
        edited_campaign_monsters_df = st.data_editor(
            pd.DataFrame(campaign_monsters_data) if campaign_monsters_data else pd.DataFrame(columns=DEFAULT_CAMPAIGN_MONSTER_COLUMNS),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Name": st.column_config.TextColumn("Monster Name", required=True),
                "Type": st.column_config.TextColumn("Type"),
                "CR": st.column_config.TextColumn("CR"),
                "Notes": st.column_config.TextColumn("Notes")
            },
            hide_index=True,
            key="monsters_in_campaign_editor", # This key is already correct
            # --- ADD THE ON_CHANGE CALLBACK HERE (inside the data_editor call) ---
            on_change=lambda: update_campaign_sublist_callback("monsters_in_campaign") # IMPORTANT: Pass "monsters_in_campaign"
        )

        st.info("Campaign Monster changes are saved automatically as you edit.")


        # Plot Lines
        st.markdown("##### 📜 Plot Lines / Quests")
        DEFAULT_PLOT_LINE_COLUMNS = ["Title", "Status", "Synopsis", "Key NPCs", "Locations"]
        plot_lines_data = st.session_state.selected_campaign_details.get("plot_lines", [])
        edited_plot_lines_df = st.data_editor(
            pd.DataFrame(plot_lines_data) if plot_lines_data else pd.DataFrame(columns=DEFAULT_PLOT_LINE_COLUMNS),
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("Title", required=True),
                "Status": st.column_config.SelectboxColumn("Status", options=["Active", "Completed", "On Hold", "Abandoned"]),
                "Synopsis": st.column_config.TextColumn("Synopsis"),
                "Key NPCs": st.column_config.TextColumn("Key NPCs"),
                "Locations": st.column_config.TextColumn("Locations")
            },
            hide_index=True,
            key="plot_lines_editor", # This key is already correct
            # --- ADD THE ON_CHANGE CALLBACK HERE (inside the data_editor call) ---
            on_change=lambda: update_campaign_sublist_callback("plot_lines") # IMPORTANT: Pass "plot_lines"
        )
        # --- REMOVE THIS LINE: It's now redundant because the on_change callback handles it ---
        # st.session_state.selected_campaign_details["plot_lines"] = edited_plot_lines_df.to_dict(orient="records")
        # --- UPDATE THIS INFO MESSAGE ---
        st.info("Plot Line changes are saved automatically as you edit.")

        st.markdown("---")

        # Delete Campaign Button
        if st.session_state.current_campaign_index != -1 and st.button("🗑️ Delete Current Campaign", key="delete_campaign_btn"):
            campaign_to_delete_name = st.session_state.campaigns[st.session_state.current_campaign_index]["Campaign Name"]
            # Remove the campaign from the list
            st.session_state.campaigns.pop(st.session_state.current_campaign_index)
            save_campaigns(st.session_state.campaigns)
            
            # Reset selected campaign details and index
            st.session_state.current_campaign_index = 0 if st.session_state.campaigns else -1
            if st.session_state.current_campaign_index != -1:
                st.session_state.selected_campaign_details = st.session_state.campaigns[st.session_state.current_campaign_index]
            else:
                st.session_state.selected_campaign_details = {
                    "Campaign Name": "", "Game System": "", "DM Name": "", "Player Names": "",
                    "Description": "", "Status": "Active", "Lore": "",
                    "players": [], "npcs": [], "monsters_in_campaign": [], "plot_lines": []
                }
            st.success(f"Campaign '{campaign_to_delete_name}' deleted!")
            st.rerun() # Rerun to refresh the UI and selector
    else:
        st.info("Create a new campaign above to manage its elements.")

elif selected_tab == "Encounter Command Center":
    st.header("🧮 Encounter Command Center")
    st.write("Manage your current combat encounters, initiative, HP, and conditions.")
    st.markdown("---")

    # Ensure DEFAULT_ENCOUNTER_COLUMNS is defined globally at the top of your script
    # (e.g., DEFAULT_ENCOUNTER_COLUMNS = ["Name", "Type", ...])

    # Ensure current_encounter_df is loaded (or initialized) for this tab
    if 'current_encounter_df' not in st.session_state or st.session_state.current_encounter_df.empty:
        st.session_state.current_encounter_df = load_encounter_data() # load_encounter_data is defined in V5_1.py
        # If still empty after loading, ensure default columns are set
        if st.session_state.current_encounter_df.empty:
            st.session_state.current_encounter_df = pd.DataFrame(columns=DEFAULT_ENCOUNTER_COLUMNS)

    st.subheader("Add Participants")
    
    # --- Add Monsters from Bestiary (Your existing code) ---
    st.markdown("###### Add Monsters from Bestiary")
    col_mon_1, col_mon_2 = st.columns([3, 1])
    
    bestiary_monster_names = []
    if 'monsters_df' in st.session_state and not st.session_state.monsters_df.empty:
        bestiary_monster_names = sorted(st.session_state.monsters_df['Name'].dropna().unique().tolist())

    with col_mon_1:
        selected_monster_for_encounter = st.selectbox(
            "Select monster(s) to add:",
            options=["--- Select Monster ---"] + bestiary_monster_names,
            key="add_monster_to_encounter_select"
        )
    with col_mon_2:
        st.write("") # Spacer
        if st.button("➕ Add Monster", key="add_monster_to_encounter_btn"):
            if selected_monster_for_encounter and selected_monster_for_encounter != "--- Select Monster ---":
                if 'monsters_df' in st.session_state and not st.session_state.monsters_df.empty:
                    monster_details = st.session_state.monsters_df[
                        st.session_state.monsters_df['Name'] == selected_monster_for_encounter
                    ].iloc[0]

                    new_participant = {
                        "Name": monster_details.get("Name", selected_monster_for_encounter),
                        "Type": "Monster",
                        "Initiative": 0, # Default initiative
                        "Max HP": monster_details.get("HP", 1),
                        "Current HP": monster_details.get("HP", 1),
                        "AC": monster_details.get("AC", 10),
                        "Conditions": ""
                    }
                    st.session_state.current_encounter_df = pd.concat([
                        st.session_state.current_encounter_df,
                        pd.DataFrame([new_participant], columns=DEFAULT_ENCOUNTER_COLUMNS)
                    ], ignore_index=True)
                    
                    st.success(f"'{selected_monster_for_encounter}' added to encounter!")
                    st.rerun()
                else:
                    st.warning("Bestiary data not loaded or empty. Cannot add monster.")
            else:
                st.warning("Please select a monster to add to the encounter.")
    
    st.markdown("---")

    # --- Add Players/NPCs from Current Campaign ---
    st.markdown("###### Add Players/NPCs from Current Campaign")
    if st.session_state.current_campaign_index != -1 and st.session_state.selected_campaign_details.get("Campaign Name"):
        campaign_players = st.session_state.selected_campaign_details.get("players", [])
        campaign_npcs = st.session_state.selected_campaign_details.get("npcs", [])

        participant_options = ["--- Select Player/NPC ---"]
        for p in campaign_players:
            participant_options.append(f"Player: {p.get('Character Name', p.get('Name', 'Unknown'))}")
        for n in campaign_npcs:
            participant_options.append(f"NPC: {n.get('Name', 'Unknown')}")
        
        col_char_1, col_char_2 = st.columns([3, 1])

        with col_char_1:
            selected_campaign_participant = st.selectbox(
                "Select player or NPC to add:",
                options=participant_options,
                key="add_campaign_char_to_encounter_select"
            )
        
        with col_char_2:
            st.write("") # Spacer
            if st.button("➕ Add Player/NPC", key="add_campaign_char_to_encounter_btn"):
                if selected_campaign_participant and selected_campaign_participant != "--- Select Player/NPC ---":
                    participant_type, participant_name = selected_campaign_participant.split(": ", 1)
                    
                    new_participant = {
                        "Name": participant_name,
                        "Type": participant_type,
                        "Initiative": 0, # Default initiative
                        "Max HP": 0, 
                        "Current HP": 0,
                        "AC": 0,
                        "Conditions": ""
                    }

                    st.session_state.current_encounter_df = pd.concat([
                        st.session_state.current_encounter_df,
                        pd.DataFrame([new_participant], columns=DEFAULT_ENCOUNTER_COLUMNS)
                    ], ignore_index=True)
                    
                    st.success(f"'{participant_name}' ({participant_type}) added to encounter!")
                    st.rerun()
                else:
                    st.warning("Please select a player or NPC to add.")
    else:
        st.info("Select a campaign in the 'Campaign Manager' tab to add players and NPCs from it.")
    
    st.markdown("---")

    # --- Main Encounter Data Editor (This is the ONE instance to keep) ---
    st.subheader("Encounter Participants")
    # Sort by initiative before displaying, so the editor always shows sorted data
    st.session_state.current_encounter_df['Initiative'] = pd.to_numeric(st.session_state.current_encounter_df['Initiative'], errors='coerce').fillna(0)
    st.session_state.current_encounter_df = st.session_state.current_encounter_df.sort_values(
        by="Initiative", ascending=False
    ).reset_index(drop=True)

    edited_df = st.data_editor(
        st.session_state.current_encounter_df,
        column_config={
            "Name": st.column_config.TextColumn("Name", required=True), 
            "Type": st.column_config.SelectboxColumn("Type", options=["Player", "Monster", "NPC"], required=True),
            "Initiative": st.column_config.NumberColumn("Initiative", min_value=0, help="Higher goes first"), 
            "Max HP": st.column_config.NumberColumn("Max HP"),
            "Current HP": st.column_config.NumberColumn("Current HP"), 
            "AC": st.column_config.NumberColumn("AC"), 
            "Conditions": st.column_config.TextColumn("Conditions")
        },
        hide_index=True, 
        key="encounter_editor", # Changed key to "encounter_editor" to avoid past conflict if any
        on_change=update_encounter_df_callback,
        num_rows="dynamic"
    )
    
    # --- Clear Encounter Button and Logic ---
    col_enc1, col_enc2 = st.columns(2)
    with col_enc1:
        if st.button("Clear Encounter", key="clear_encounter_btn"):
            st.session_state.current_encounter_df = pd.DataFrame(columns=DEFAULT_ENCOUNTER_COLUMNS)
            # Ensure 'Conditions' column is string type for consistency
            if "Conditions" in st.session_state.current_encounter_df.columns:
                st.session_state.current_encounter_df["Conditions"] = st.session_state.current_encounter_df["Conditions"].astype(str)
            save_encounter_data(st.session_state.current_encounter_df) # Save the empty state
            st.success("Encounter cleared!")
            st.rerun() # Rerun to reflect the cleared table

    # --- AUTO-SAVE INFO MESSAGE ---
    st.info("Encounter participants are saved automatically as you edit.")
    # --- END AUTO-SAVE INFO MESSAGE ---

    # --- Removed old redundant st.data_editor and old 'Add Participants' button ---
    # The data editor handles additions and edits. Separate 'Add Participants' button is not needed
    # unless it triggers a more complex, non-editor-based addition logic.
    # The previous code block you provided had a second data_editor and a button
    # which are now removed to prevent duplication and key errors.
    
    # Remove the saved CSV file if it exists
    if os.path.exists(ENCOUNTER_DATA_PATH):
        os.remove(ENCOUNTER_DATA_PATH)
        st.info("Saved encounter file removed.")
    else:
        st.info("No participants in the current encounter. Add them manually or generate a map with monsters.")
    
    # Create an empty DataFrame with correct columns and dtypes for display
    empty_encounter_df = pd.DataFrame(columns=DEFAULT_ENCOUNTER_COLUMNS)
    if "Conditions" in empty_encounter_df.columns:
        empty_encounter_df["Conditions"] = empty_encounter_df["Conditions"].astype(str)

    edited_encounter_df = st.data_editor(empty_encounter_df, num_rows="dynamic", use_container_width=True, # Use the new empty_encounter_df
        column_config={
            "Name": st.column_config.TextColumn("Name", required=True), 
            "Type": st.column_config.SelectboxColumn("Type", options=["Player", "Monster", "NPC"], required=True),
            "Initiative": st.column_config.NumberColumn("Initiative", min_value=0, help="Higher goes first"), 
            "Max HP": st.column_config.NumberColumn("Max HP"),
            "Current HP": st.column_config.NumberColumn("Current HP"), 
            "AC": st.column_config.NumberColumn("AC"), 
            "Conditions": st.column_config.TextColumn("Conditions")
        },
        hide_index=True, 
        key="encounter_data_editor", # IMPORTANT: Ensure this key is "encounter_data_editor" for consistency with the callback
        on_change=update_encounter_df_callback) # Ensure on_change is here as well

    # --- ADD THE AUTO-SAVE INFO MESSAGE HERE ---
    st.info("Encounter participants are saved automatically as you edit.")
    # --- END AUTO-SAVE INFO MESSAGE ---

    if st.button("Add Participants", key="add_encounter_participants_btn"):
            # The st.data_editor with on_change=update_encounter_df_callback handles saving changes directly.
            # This button can now be used for other logic, like perhaps adding new rows if the data_editor
            # itself isn't sufficient for complex additions, or just for confirming input
            # from other fields, without explicitly re-saving the entire DataFrame.
            st.success("Participants added/changes saved via auto-save.")
            # If you have *other* logic here that manipulates the DataFrame and needs saving,
            # you'd keep a save_encounter_data call, but based on current pattern, likely not.