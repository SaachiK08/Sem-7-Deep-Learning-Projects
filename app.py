"""
NutriScan AI - Streamlit Application
"""
import streamlit as st
from PIL import Image
import os
import model1
import model2
import model3
import model4 
st.set_page_config(page_title="NutriScan AI", layout="wide")

@st.cache_resource
def load_model4_db():
    """
    Loads the alternatives CSV file into a pandas DataFrame.
    """
    print("--- Loading Model 4 Alternatives Database ---")
    return model4.load_alternatives("data\product_alternatives.csv")

# Load Model 1
@st.cache_resource
def load_model1_data():
    """
    Loads the product recognition model and data.
    """
    print("--- Loading Model 1 Product Recognition Model ---")
    return model1.load_model_and_data()

# Load Model 2
@st.cache_resource
def load_model2_data():
    """
    Loads the ingredient and nutrition databases.
    """
    print("--- Loading Model 2 Ingredient/Nutrition Databases ---")
    return model2.load_data()

# --- Load all data on startup ---
alternatives_db = load_model4_db()
model1_model, model1_data, model1_labels = load_model1_data()
model2_safety_df, model2_nutrition_df, model2_ingredients_list = load_model2_data()

st.title("NutriScan AI: Your Personal Food Scanner")
st.write("Upload an image of a food product to get a detailed analysis of its ingredients, nutritional value, and a safety score.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Image.', use_column_width=True)

    with col2:
        st.write("Analyzing... please wait.")

        # Save temp file for model1 to process
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        product_info = None
        analysis_result = None
        safety_score = None

        # ---------------- MODEL 1 ----------------
        with st.expander("Step 1: Product Identification", expanded=True):
            try:
                product_info = model1.get_product_info_from_image(
                    temp_image_path, 
                    model1_model, 
                    model1_data, 
                    model1_labels
                )
                if product_info:
                    st.success(f"**Product Identified:** {product_info['product_name']} (Confidence: {product_info['prediction_confidence']:.2%})")
                else:
                    st.error("Could not identify the product or find it in our database.")
            except Exception as e:
                st.error(f"An error occurred in Model 1: {e}")
                product_info = None

        # ---------------- MODEL 2 ----------------
        if product_info:
            with st.expander("Step 2: Ingredient and Nutrition Analysis", expanded=True):
                try:
                    analysis_result = model2.analyze_product(
                        product_info, 
                        model2_safety_df, 
                        model2_nutrition_df, 
                        model2_ingredients_list
                    )
                    st.subheader("Ingredient Analysis")
                    for ing in analysis_result['ingredient_analysis']:
                        icon = "🟢" if ing['status'] == 'safe' else "🟡" if ing['status'] == 'caution' else "🔴" if ing['status'] == 'unsafe' else "⚪"
                        st.markdown(f"{icon} **{ing['name']}**: {ing['status'].capitalize()} - {ing['reason']}")

                    st.subheader("Nutritional Evaluation")
                    st.markdown("#### Pros")
                    if analysis_result['nutrition_pros']:
                        for pro in analysis_result['nutrition_pros']:
                            st.markdown(f"{pro}")
                    else:
                        st.markdown("_No specific pros identified._")

                    st.markdown("#### Cons")
                    if analysis_result['nutrition_cons']:
                        for con in analysis_result['nutrition_cons']:
                            st.markdown(f"{con}")
                    else:
                        st.markdown("_No specific cons identified._")

                except Exception as e:
                    st.error(f"An error occurred in Model 2: {e}")
                    analysis_result = None

        # ---------------- MODEL 3 ----------------
        if analysis_result:
            with st.expander("Step 3: Overall Safety Score", expanded=True):
                try:
                    safety_score = model3.compute_safety_score_from_model2_output(
                        analysis_result,
                        weight_ingredient=0.7, 
                        weight_nutrition=0.3
                    )

                    st.metric(label="Overall Product Safety Score", value=f"{safety_score}/10")
                    
                    if safety_score < 4.0:
                        st.error("This product is rated 'Unsafe' due to its ingredients and/or nutritional profile.")
                    elif safety_score < 7.0:
                        st.warning("This product is rated 'Moderate'. Consume in moderation.")
                    else:
                        st.success("This product is rated 'Safe'.")

                except Exception as e:
                    st.error(f"An error occurred in Model 3: {e}")
                    safety_score = None

        # ---------------- MODEL 4 ----------------
        if product_info and safety_score is not None:
            with st.expander("Step 4: Healthier Alternatives", expanded=True):
                try:
                    alternative_string = model4.suggest_alternative(
                        product_name=product_info['product_name'],
                        safety_score=safety_score,
                        alternatives_df=alternatives_db
                    )

                    if alternative_string:
                        st.write("Here is a healthier alternative you might consider:")
                        st.success(f"**Suggestion:** {alternative_string}")
                    else:
                        if safety_score >= 7.0:
                            st.info("This product is rated 'Safe', so no alternative is suggested.")
                        else:
                            st.info(f"This product is rated 'Moderate' or 'Unsafe', but no simple alternative was found for '{product_info['product_name']}' in our database.")
                
                except Exception as e:
                    st.error(f"An error occurred in Model 4: {e}")

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)