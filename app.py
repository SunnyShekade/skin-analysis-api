from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import your_cnn_model  # Import your CNN model here
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/analyze', methods=['POST'])
def analyze_skin():
    if 'image' not in request.files:
        logging.error('No image uploaded')
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']

    if not image.content_type.startswith('image/'):
        logging.error('Uploaded file is not an image')
        return jsonify({'error': 'Uploaded file is not an image'}), 400

    try:
        skin_metrics = your_cnn_model.analyze(image)  # Process the image with your model
    except Exception as e:
        logging.error(f'Error analyzing skin: {str(e)}')
        return jsonify({'error': str(e)}), 500  # Return error if analysis fails

    return jsonify(skin_metrics)

    # Return a detailed JSON response with all skin metrics
    return jsonify({
        'skinTone': skin_metrics['tone'][0],  # Label
        'skinTonePercentage': f"{skin_metrics['tone'][1]:.2f}%",  # Percentage formatted
        'acneLevel': skin_metrics['acne_level'][0],  # Label
        'acneLevelPercentage': f"{skin_metrics['acne_level'][1]:.2f}%",  # Percentage formatted
        'blackheads': skin_metrics['blackheads'][0],  # Label
        'blackheadsPercentage': f"{skin_metrics['blackheads'][1]:.2f}%",  # Percentage formatted
        'darkCircles': skin_metrics['dark_circles'][0],  # Label
        'darkCirclesPercentage': f"{skin_metrics['dark_circles'][1]:.2f}%",  # Percentage formatted
        'skinType': skin_metrics['skin_type'][0],  # Label
        'skinTypePercentage': f"{skin_metrics['skin_type'][1]:.2f}%",  # Percentage formatted
        'hairQuality': skin_metrics['hair_quality'][0],  # Label
        'hairQualityPercentage': f"{skin_metrics['hair_quality'][1]:.2f}%",  # Percentage formatted
        'hydrationLevel': skin_metrics['hydration_level'][0],  # Label
        'hydrationLevelPercentage': f"{skin_metrics['hydration_level'][1]:.2f}%",  # Percentage formatted
        'sensitivity': skin_metrics['sensitivity'][0],  # Label
        'sensitivityPercentage': f"{skin_metrics['sensitivity'][1]:.2f}%",  # Percentage formatted
        'wrinkles': skin_metrics['wrinkles'][0],  # Label
        'wrinklesPercentage': f"{skin_metrics['wrinkles'][1]:.2f}%",  # Percentage formatted
        'poreSize': skin_metrics['pore_size'][0],  # Label
        'poreSizePercentage': f"{skin_metrics['pore_size'][1]:.2f}%",  # Percentage formatted
        'recommendedProducts': skin_metrics.get('recommendedProducts', [])
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port))
