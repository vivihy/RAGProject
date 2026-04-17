import json
import os
from dotenv import load_dotenv

load_dotenv()

def load_full_config(json_path="configs/optimizer_config_template.json"):
    with open(json_path, 'r') as f:
        return json.load(f)


def load_search_space(json_path="configs/optimizer_config_template.json"):
    config = load_full_config(json_path)
    return config["search_space"]


def get_llm_config(model_alias, json_path="configs/optimizer_config_template.json"):
    config = load_full_config(json_path)
    registry = config.get("model_registry", {})
    if model_alias not in registry:
        raise ValueError(f"Unknown model alias: {model_alias}")

    cfg = registry[model_alias]
    result = {
        "provider": cfg["provider"],
        "api_base": os.getenv(cfg.get("api_base_env", ""), cfg.get("api_base_default", "")),
        "model": os.getenv(cfg.get("model_name_env", ""), cfg.get("model_name_default", ""))
    }
    if "api_key_env" in cfg:
        result["api_key"] = os.getenv(cfg["api_key_env"])
        if not result["api_key"]:
            raise ValueError(f"Missing env {cfg['api_key_env']} for {model_alias}")
    return result