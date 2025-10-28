import pandas as pd
import re
from rapidfuzz import fuzz

def load_data():
    ingredient_safety_df = pd.read_csv("data/ingredients_safety_merged.csv")
    nutrition_df = pd.read_csv("data/nutrition_standards.csv")
    ingredient_safety_df.columns = ingredient_safety_df.columns.str.strip().str.lower()
    nutrition_df.columns = nutrition_df.columns.str.strip().str.lower()
    ingredients_list = ingredient_safety_df['ingredient_name'].tolist()
    return ingredient_safety_df, nutrition_df, ingredients_list

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def match_ingredient(ingredient: str, ingredients_list: list, threshold=80):
    ingr_clean = clean_text(ingredient)
    best_match = None
    highest_score = 0
    for ref in ingredients_list:
        ref_clean = clean_text(ref)
        score = fuzz.ratio(ingr_clean, ref_clean)
        if score > highest_score:
            highest_score = score
            best_match = ref
    if highest_score >= threshold:
        return best_match
    for ref in ingredients_list:
        ref_words = clean_text(ref).split()
        for word in ingr_clean.split():
            if word in ref_words:
                return ref
    return ingredient

def detect_ingredients(ingredients_input, ingredient_safety_df, ingredients_list):
    results = []
    for ingr in ingredients_input:
        canonical_name = match_ingredient(ingr, ingredients_list)
        safety_row = ingredient_safety_df[ingredient_safety_df['ingredient_name'].str.lower() == canonical_name.lower()]
        if not safety_row.empty:
            status = safety_row.iloc[0]['safety_status']
            reason = safety_row.iloc[0]['reason_for_unsafety']
        else:
            status = "unknown"
            reason = "No information available"
        results.append({
            "name": canonical_name,
            "status": status,
            "reason": reason
        })
    order = {"unsafe": 0, "caution": 1, "safe": 2, "unknown": 3}
    results = sorted(results, key=lambda x: order.get(x['status'], 3))
    return results

def evaluate_nutrition(nutrition_input, nutrition_df):
    pros, cons = [], []
    for _, row in nutrition_df.iterrows():
        nutrient = row['nutrient']
        if nutrient not in nutrition_input or pd.isna(nutrition_input[nutrient]):
            continue
        value = nutrition_input[nutrient]
        if isinstance(value, str):
            value = float(re.findall(r"[\d\.]+", value)[0])
        low = float(row['low_threshold'])
        high = float(row['high_threshold'])
        note = row['note']
        if value < low:
            cons.append(f"Low {nutrient} ({note})")
        elif value > high:
            cons.append(f"High {nutrient} ({note})")
        else:
            pros.append(f"Balanced {nutrient}")
    return pros, cons

def analyze_product(product_json, ingredient_safety_df, nutrition_df, ingredients_list):
    ingredients_input = product_json.get("ingredients", [])
    nutrition_input = product_json.get("nutrition", {})
    product_name = product_json.get("product_name", "Unnamed Product")
    ingredient_analysis = detect_ingredients(ingredients_input, ingredient_safety_df, ingredients_list)
    nutrition_pros, nutrition_cons = evaluate_nutrition(nutrition_input, nutrition_df)
    result = {
        "product_name": product_name,
        "ingredient_analysis": ingredient_analysis,
        "nutrition_pros": nutrition_pros,
        "nutrition_cons": nutrition_cons
    }
    return result