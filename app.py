from flask import Flask, request, render_template
import os
from detect import detect_image, detect_video, get_detected_classes

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = 'static/videos'
IMAGE_FOLDER = 'static/images'

# Ensure all folders exist
os.makedirs(os.path.join(UPLOAD_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'videos'), exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['image']
    model_choice = request.form.get('model', 'both')
    image_path = os.path.join(UPLOAD_FOLDER, 'images', file.filename)
    file.save(image_path)

    output_path_v5 = os.path.join(IMAGE_FOLDER, f"v5_{file.filename}")
    output_path_v8 = os.path.join(IMAGE_FOLDER, f"v8_{file.filename}")

    metrics = {}
    result_file_v5 = result_file_v8 = None

    if model_choice == 'v5':
        metrics = detect_image(image_path, output_path_v5, None)
        result_file_v5 = output_path_v5
        detected_classes = get_detected_classes(image_path, model_choice)
    elif model_choice == 'v8':
        metrics = detect_image(image_path, None, output_path_v8)
        result_file_v8 = output_path_v8
        detected_classes = get_detected_classes(image_path, model_choice)
    else:  # both
        metrics = detect_image(image_path, output_path_v5, output_path_v8)
        result_file_v5 = output_path_v5
        result_file_v8 = output_path_v8
        detected_classes = get_detected_classes(image_path, model_choice)

    return render_template('index.html',
                           result_type="image",
                           result_file_v5=result_file_v5,
                           result_file_v8=result_file_v8,
                           detected_classes=detected_classes,
                           metrics=metrics)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['video']
    model_choice = request.form.get('model', 'both')
    video_path = os.path.join(UPLOAD_FOLDER, 'videos', file.filename)
    file.save(video_path)

    output_path_v5 = os.path.join(VIDEO_FOLDER, f"v5_{file.filename}")
    output_path_v8 = os.path.join(VIDEO_FOLDER, f"v8_{file.filename}")

    result = {}
    result_file_v5 = result_file_v8 = None

    if model_choice == 'v5':
        result = detect_video(video_path, output_path_v5, None)
        result_file_v5 = output_path_v5
    elif model_choice == 'v8':
        result = detect_video(video_path, None, output_path_v8)
        result_file_v8 = output_path_v8
    else:
        result = detect_video(video_path, output_path_v5, output_path_v8)
        result_file_v5 = output_path_v5
        result_file_v8 = output_path_v8

    return render_template('index.html',
                           result_type="video",
                           result_file_v5=result_file_v5,
                           result_file_v8=result_file_v8,
                           detected_classes=result.get("classes", []),
                           metrics=result)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
