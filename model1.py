import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras import layers, models        
import numpy as np
import pandas as pd
import json
import os

WEIGHTS_PATH = "data/product_classifier.weights.h5" 
DATA_PATH = "data/product_ingredients.xlsx"
LABELS_PATH = "data/class_labels.json"
IMG_SIZE = (224, 224)
NUM_CLASSES = 10

def create_model_architecture(num_classes):
    """Defines the MobileNetV2-based model structure."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def load_model_and_data():
    """Builds model, loads weights, loads data, and loads class labels."""
    model = create_model_architecture(NUM_CLASSES)
    
    try:
        model.load_weights(WEIGHTS_PATH)
    except FileNotFoundError:
        raise RuntimeError(f"Error: Weights file not found at {WEIGHTS_PATH}. Did you run the training script and save weights using model.save_weights('product_classifier_weights.h5')?")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights. Original error: {e}")
        
    data = pd.read_excel(DATA_PATH)
    
    try:
        with open(LABELS_PATH, 'r') as f:
            raw_labels = json.load(f)
            class_labels = {int(k): v for k, v in raw_labels.items()}
    except FileNotFoundError:
        print(f"Warning: {LABELS_PATH} not found. Using hardcoded labels.")
        class_labels = {
            "0": "Amul_Butter", "1": "Bingo_Mad_Angles", "2": "Cadbury_Dairy_Milk", "3": "Coca_Cola", "4": "Kurkure_Masala_Munch", "5": "Maggi_Masala_Noodles", "6": "Oreo_Original", "7": "Parle_G", "8": "Top_Ramen_Curry_Noodles", "9": "Yippee_Noodles"
        }

    return model, data, class_labels

def predict_product(image_path, model, class_labels):
    """Predicts the product class and confidence from an image."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    
    return class_labels[class_idx], confidence

def format_product_info(row, confidence):
    """Formats the product row and confidence into the required JSON structure."""
    product_json = {
        "product_id": row.get("product_id", None),
        "product_name": row.get("product_name", None),
        "product_type": row.get("category", None),
        "ingredients": [ing.strip() for ing in str(row.get("ingredients", "")).split(",")],
        "nutrition": {
            "energy_kcal": row.get("energy_kcal", None),
            "carbs_g": row.get("carbs_g", None),
            "total_sugar_g": row.get("total_sugar_g", None),
            "added_sugar_g": row.get("added_sugar_g", None),
            "protein_g": row.get("protein_g", None),
            "total_fat_g": row.get("total_fat_g", None),
            "sat_fat_g": row.get("sat_fat_g", None),
            "trans_fat_g": row.get("trans_fat_g", None),
            "sodium_mg": row.get("sodium_mg", None),
        },
        "prediction_confidence": round(float(confidence), 4)
    }
    return product_json

def get_product_info_from_image(image_path, model, data, class_labels):
    """Main function to predict, lookup, and format product data."""
    
    product_name, conf = predict_product(image_path, model, class_labels)
    product_info = data[data["product_name"] == product_name]
    
    if not product_info.empty:
        row = product_info.iloc[0].to_dict()
        product_json = format_product_info(row, conf)
        return product_json
    else:
        return {
            "error": "Product not found in database",
            "predicted_name": product_name,
            "confidence": round(float(conf), 4)
        }