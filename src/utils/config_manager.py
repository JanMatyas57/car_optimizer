import json
import os
import glob

class ConfigManager:
    # define the directory for storing vehicle profiles
    VEHICLE_DIR = os.path.join("src", "core", "vehicles")

    @staticmethod
    def _ensure_dir():
        """creates the directory if it doesn't exist."""
        if not os.path.exists(ConfigManager.VEHICLE_DIR):
            os.makedirs(ConfigManager.VEHICLE_DIR)

    @staticmethod
    def get_available_vehicles() -> list:
        """returns a list of saved vehicle names (without .json extension)."""
        ConfigManager._ensure_dir()
        files = glob.glob(os.path.join(ConfigManager.VEHICLE_DIR, "*.json"))
        names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        return sorted(names)

    @staticmethod
    def load_vehicle(name: str, default_config: dict) -> dict:
        """loads a specific vehicle's config and merges it with defaults."""
        ConfigManager._ensure_dir()
        filepath = os.path.join(ConfigManager.VEHICLE_DIR, f"{name}.json")
        
        merged_config = default_config.copy()
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                merged_config.update(saved_config)
            except Exception as e:
                print(f"failed to load vehicle '{name}': {e}. using defaults.")

    
            
        return merged_config

    @staticmethod
    def save_vehicle(name: str, config_dict: dict):
        """saves the current config dictionary to a specific vehicle's json file."""
        ConfigManager._ensure_dir()
        filepath = os.path.join(ConfigManager.VEHICLE_DIR, f"{name}.json")
        try:
            config_to_save = config_dict.copy()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4)
        except Exception as e:
            print(f"failed to save vehicle '{name}': {e}")