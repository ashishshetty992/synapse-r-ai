# phase3-neurons/app/config_api.py
from fastapi import APIRouter
import json
from .active_cfg import active_shaping_path, active_fewshot_path

router = APIRouter(prefix="/config", tags=["config"])

@router.get("/effective")
def effective_config():
    return {
        "active": {
            "shaping": active_shaping_path(),
            "fewshot": active_fewshot_path(),
        }
    }

