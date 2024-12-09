from flask import Flask, render_template, request, jsonify
import os
from logistic_regression import search_images

app = Flask(__name__)

# Ensure results directory exists
os.makedirs("uploads", exist_ok=True)

# Route for the main page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle search queries
@app.route("/search", methods=["POST"])
def search():
    # Retrieve text and image queries
    text_query = request.form.get("text-query", "")
    query_type = request.form.get("query-type")
    weight = float(request.form.get("query-weight", 0.5))

    image_file = request.files.get("image-query")
    image_path = None

    if image_file:
        image_path = os.path.join("uploads", image_file.filename)
        image_file.save(image_path)

    # Perform search
    results = search_images(text_query, image_path, query_type, weight)

    # Return JSON response
    return jsonify(results)


from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def serve_uploaded_image(filename):
    return send_from_directory('uploads', filename)


if __name__ == "__main__":
    app.run(debug=True)
