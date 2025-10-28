# model3.py
"""
Model 3: Combined Safety Score
Provides compute_safety_score(...) which combines:
 - ingredient-based analysis (safe / caution / unsafe / unknown)
 - nutritional evaluation (either as a dict of nutrient values OR as lists of pros/cons)

The function is defensive and works with varying shapes returned by model2.analyze_product.
"""
from typing import List, Dict, Optional

def _compute_ingredient_score(ingredient_analysis: List[Dict]) -> float:
    """
    Convert ingredient statuses into a 0-10 score.
    ingredient_analysis: list of {"name": ..., "status": "safe"|"caution"|"unsafe"|"unknown", ...}
    """
    if not ingredient_analysis:
        return 5.0 

    mapping = {'safe': 1, 'caution': 0, 'unsafe': -1, 'unknown': 0}
    raw = 0
    for ing in ingredient_analysis:
        raw += mapping.get(str(ing.get("status", "")).lower(), 0)

    n = len(ingredient_analysis)
    normalized = ((raw + n) / (2 * n)) * 10
    return round(normalized, 2)


def _compute_nutrition_score_from_values(nutrition_info: Dict[str, float]) -> float:
    """
    nutrition_info expected to be dict with numeric values, e.g. {"sugar_g": 10, "sodium_mg": 200, "protein_g": 5, "total_fat_g": 8}
    Produces a 0-10 style score centered at 5 with penalties/bonuses.
    """
    if not nutrition_info:
        return 5.0
    sugar = float(nutrition_info.get("sugar_g", 0) or 0)
    sodium = float(nutrition_info.get("sodium_mg", 0) or 0)
    fat = float(nutrition_info.get("total_fat_g", 0) or 0)
    protein = float(nutrition_info.get("protein_g", 0) or 0)

    base = 5.0
    sugar_penalty = min(sugar / 10.0, 2.5)       
    sodium_penalty = min(sodium / 500.0, 2.0)    
    fat_penalty = min(fat / 10.0, 1.5)           
    protein_bonus = min(protein / 5.0, 2.0)      

    adj = -sugar_penalty - sodium_penalty - fat_penalty + protein_bonus
    score = max(0.0, min(10.0, base + adj))
    return round(score, 2)


def _compute_nutrition_score_from_pros_cons(nutrition_pros: List[str], nutrition_cons: List[str]) -> float:
    """
    If you only have pros/cons text lists (from model2), convert to score:
    - each 'pro' adds a small positive
    - each 'con' subtracts more
    """
    base = 5.0
    base += len(nutrition_pros) * 0.5
    base -= len(nutrition_cons) * 0.7
    return round(max(0.0, min(10.0, base)), 2)


def compute_safety_score(
    ingredient_analysis: List[Dict],
    nutrition_info: Optional[Dict] = None,
    nutrition_pros: Optional[List[str]] = None,
    nutrition_cons: Optional[List[str]] = None,
    weight_ingredient: float = 0.6,
    weight_nutrition: float = 0.4
) -> float:
    """
    Public function your app should call.

    Parameters:
    - ingredient_analysis: list of ingredient analysis dicts (from model2)
    - nutrition_info: optional dict of numeric nutrition fields (preferred)
    - nutrition_pros / nutrition_cons: alternative input (text lists) if numeric dict not available
    - weight_ingredient / weight_nutrition: combine weights (sum not strictly enforced; they will be normalized)

    Returns:
    - final safety score (0.0 - 10.0)
    """
    ingr_score = _compute_ingredient_score(ingredient_analysis)

    if nutrition_info:
        nutri_score = _compute_nutrition_score_from_values(nutrition_info)
    else:
        pros = nutrition_pros or []
        cons = nutrition_cons or []
        nutri_score = _compute_nutrition_score_from_pros_cons(pros, cons)

    w_ing = float(weight_ingredient)
    w_nut = float(weight_nutrition)
    total_w = w_ing + w_nut if (w_ing + w_nut) != 0 else 1.0
    w_ing /= total_w
    w_nut /= total_w

    final = (ingr_score * w_ing) + (nutri_score * w_nut)
    final = max(0.0, min(10.0, final))
    return round(final, 2)

def compute_safety_score_from_model2_output(model2_output: Dict, weight_ingredient=0.6, weight_nutrition=0.4) -> float:
    """
    Accepts model2 output (dict returned by analyze_product) and extracts
    required fields automatically.
    """
    ingredient_analysis = model2_output.get("ingredient_analysis", [])
    nutrition_info = model2_output.get("nutrition", None) or model2_output.get("nutrition_info", None) or None
    pros = model2_output.get("nutrition_pros", [])
    cons = model2_output.get("nutrition_cons", [])
    return compute_safety_score(
        ingredient_analysis=ingredient_analysis,
        nutrition_info=nutrition_info,
        nutrition_pros=pros,
        nutrition_cons=cons,
        weight_ingredient=weight_ingredient,
        weight_nutrition=weight_nutrition
    )
