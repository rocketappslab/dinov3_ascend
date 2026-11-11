from flask import Flask, request, jsonify
from flask_cors import CORS
import easyocr
import os
import tempfile
from werkzeug.utils import secure_filename


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize EasyOCR (loads model once)
reader = easyocr.Reader(['ch_sim','en'], model_storage_directory="/home/ei/workspace/rocket/Project/nanfang_capsule/src/data/easyor_model", gpu=True, download_enabled=False)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/ocr', methods=['POST'])
def ocr_api():
    try:
        data = request.get_json() or {}
        image_path = data.get('image_path')

        if not image_path:
            return jsonify({'error': 'No image_path provided'}), 400

        # EasyOCR accepts file path strings directly
        result = reader.readtext(image_path)

        ocr_results = []
        for (bbox, text, confidence) in result:
            serializable_bbox = [[float(coord) for coord in point] for point in bbox]
            ocr_results.append({
                'text': text,
                'confidence': float(confidence),
                'bbox': serializable_bbox
            })

        return jsonify({
            'success': True,
            'results': ocr_results,
            'count': len(ocr_results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Run HTTPS server
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=4443,        # HTTPS default port
        debug=False      # Set to True for development only
    )
