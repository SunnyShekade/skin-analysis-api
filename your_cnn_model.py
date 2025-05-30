import numpy as np
import cv2  # OpenCV for image processing
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F

# Load your pre-trained model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Example model, replace with your own
num_classes = 35  # Ensure this matches your model's output
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.eval()  # Set the model to evaluation mode

# Define the image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image):
    image.seek(0)
    # Load the image and preprocess it for your model
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image could not be decoded. Please upload a valid image.")
    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0)  # Add batch dimension
    print("Preprocessed image shape:", img.shape)  # Debugging: Print image shape
    return img

def analyze(image):
    img = preprocess_image(image)  # Preprocess the image

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(img)  # Run the image through your CNN model
        print("Model output shape:", outputs.shape)  # Debugging: Print output shape
        
        if outputs.numel() == 0:
            raise ValueError("Model output is empty. Please check the input image and model.")
        
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to convert logits to probabilities
        print("Probabilities shape:", probabilities.shape)  # Debugging: Print probabilities shape
        print("Probabilities:", probabilities)  # Debugging: Print probabilities

    # Ensure there are enough classes
    if probabilities.shape[1] != num_classes:  # Check against num_classes
        raise ValueError("Model output does not have the expected number of classes.")

    # Access the probabilities correctly
    try:
        tone_index = torch.argmax(probabilities[0][:3]).item()  # First 3 classes for tone
        acne_level_index = torch.argmax(probabilities[0][3:7]).item()  # Next 4 classes for acne level
        blackheads_index = torch.argmax(probabilities[0][7:11]).item()  # Next 4 classes for blackheads
        dark_circles_index = torch.argmax(probabilities[0][11:15]).item()  # Next 4 classes for dark circles
        skin_type_index = torch.argmax(probabilities[0][15:19]).item()  # Next 4 classes for skin type
        hair_quality_index = torch.argmax(probabilities[0][19:22]).item()  # Next 3 classes for hair quality
        hydration_level_index = torch.argmax(probabilities[0][22:25]).item()  # Next 3 classes for hydration
        sensitivity_index = torch.argmax(probabilities[0][25:28]).item()  # Next 3 classes for sensitivity
        wrinkles_index = torch.argmax(probabilities[0][28:32]).item()  # Next 4 classes for wrinkles
        pore_size_index = torch.argmax(probabilities[0][32:35]).item()  # Next 3 classes for pore size
    except Exception as e:
                raise ValueError(f"Error while calculating argmax: {str(e)}")

    # Assuming the model outputs probabilities for various skin metrics
    skin_metrics = {
        'tone': map_skin_tone(tone_index, probabilities[0][:3]),
        'acne_level': map_acne_level(acne_level_index, probabilities[0][3:7]),
        'blackheads': map_blackheads(blackheads_index, probabilities[0][7:11]),
        'dark_circles': map_dark_circles(dark_circles_index, probabilities[0][11:15]),
        'skin_type': map_skin_type(skin_type_index, probabilities[0][15:19]),
        'hair_quality': map_hair_quality(hair_quality_index, probabilities[0][19:22]),
        'hydration_level': map_hydration(hydration_level_index, probabilities[0][22:25]),
        'sensitivity': map_sensitivity(sensitivity_index, probabilities[0][25:28]),
        'wrinkles': map_wrinkles(wrinkles_index, probabilities[0][28:32]),
        'pore_size': map_pore_size(pore_size_index, probabilities[0][32:35]),
    }

    return skin_metrics


def calculate_percentage(probabilities, index, category_count):
    # Convert probabilities to a list of floats
    probabilities = [p.item() for p in probabilities]
    
    # Normalize probabilities to sum to 1
    total_prob = sum(probabilities)
    if total_prob > 0:
        normalized_probs = [p / total_prob for p in probabilities]
    else:
        normalized_probs = [0] * len(probabilities)  # Handle case where all probabilities are zero

    # Calculate the percentage based on normalized probabilities
    percentage = normalized_probs[index] * 100
    
    # Ensure a minimum percentage of 10%
    if percentage < 10:
        percentage = 10
    
    return round(percentage)

def map_category_with_dynamic_percentage(index, probabilities, categories):
    category = categories[index]
    percentage = calculate_percentage(probabilities, index, len(categories))
    return category, f"{percentage}%"

# Update mapping functions to use the new dynamic percentage calculation
def map_skin_tone(index, probabilities):
    categories = ['Light', 'Medium', 'Dark']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_acne_level(index, probabilities):
    categories = ['None', 'Mild', 'Moderate', 'Severe']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_blackheads(index, probabilities):
    categories = ['None', 'Few', 'Moderate', 'Many']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_dark_circles(index, probabilities):
    categories = ['None', 'Mild', 'Moderate', 'Severe']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_skin_type(index, probabilities):
    categories = ['Oily', 'Dry', 'Combination', 'Normal']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_hair_quality(index, probabilities):
    categories = ['Poor', 'Average', 'Good']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_hydration(index, probabilities):
    categories = ['Dehydrated', 'Normal', 'Well-Hydrated']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_sensitivity(index, probabilities):
    categories = ['Low', 'Medium', 'High']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_wrinkles(index, probabilities):
    categories = ['None', 'Few', 'Moderate', 'Many']
    return map_category_with_dynamic_percentage(index, probabilities, categories)

def map_pore_size(index, probabilities):
    categories = ['Small', 'Medium', 'Large']
    return map_category_with_dynamic_percentage(index, probabilities, categories)