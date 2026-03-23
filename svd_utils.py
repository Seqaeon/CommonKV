import json

def get_rank(model_path):
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path).to_dict()
    except Exception:
        with open(f"{model_path}/config.json", "r") as f:
            config = json.load(f)

    head_wise_ranks = config.get("head_wise_ranks", {})
    return head_wise_ranks
