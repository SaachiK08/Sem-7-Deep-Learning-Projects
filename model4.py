"""
Model 4: Product Alternative Lookup (Simple Logic)

Provides:
- load_alternatives(): Loads the product lookup CSV.
- suggest_alternative(): Checks safety score, then finds a specific 
                         alternative from the loaded data.
"""
import pandas as pd

def load_alternatives(csv_path="data\product_alternatives.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: Alternatives file not found at {csv_path}.")
        print("Creating a small example DataFrame to continue.")
        data = {
            "Product": ["Coca_Cola", "Maggi_Masala_Noodles", "Parle_G"],
            "Category": ["Carbonated Soft Drink", "Instant Noodles", "Glucose Biscuit"],
            "Alternative Safe Product": [
                "Sparkling Water with a slice of lemon/lime", 
                "Whole Wheat/Oats Noodles with homemade seasoning",
                "Whole Wheat Digestive Biscuits"
            ]
        }
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame(columns=["Product", "Alternative Safe Product"])
    if "Product" not in df.columns or "Alternative Safe Product" not in df.columns:
        print("Error: The CSV must have 'Product' and 'Alternative Safe Product' columns.")
        return pd.DataFrame(columns=["Product", "Alternative Safe Product"])
    df['lookup_key'] = df['Product'].str.lower().str.replace(' ', '_')

    df.set_index('lookup_key', inplace=True)
    
    return df

def suggest_alternative(product_name: str, safety_score: float, alternatives_df: pd.DataFrame) -> str | None:
    """
    Suggests an alternative for a given product name *only if* the
    safety score is below the threshold (7.0).

    Args:
        product_name: The name of the product (e.g., "Coca Cola").
        safety_score: The score from Model 3 (e.g., 3.5).
        alternatives_df: The loaded DataFrame from load_alternatives().

    Returns:
        The alternative product string, or None if not needed or not found.
    """
    if safety_score >= 7.0:
        return None
    if alternatives_df is None or alternatives_df.empty:
        return None
    lookup_key = product_name.lower().replace(' ', '_')
    
    try:
        alternative = alternatives_df.loc[lookup_key, 'Alternative Safe Product']
        return str(alternative)
    except KeyError:
        return None
    except Exception as e:
        print(f"Error during lookup for '{lookup_key}': {e}")
        return None

