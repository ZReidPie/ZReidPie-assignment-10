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
    text_query = request.form.get("text-query", "")
    query_type = request.form.get("query-type")
    weight = float(request.form.get("query-weight", 0.5))
    use_pca = request.form.get("use-pca", "false") == "true"  # Checkbox or toggle in the frontend
    k = int(request.form.get("pca-k", 10)) if use_pca else None  # User-specified k value

    if use_pca and (k is None or k <= 0 or k > 512):  # Assuming embeddings have 512 dimensions
        return jsonify({"error": f"Invalid value for Number of Principal Components (k). Ensure 1 <= k <= 512."})



    image_file = request.files.get("image-query")
    image_path = None
    if image_file:
        image_path = os.path.join("uploads", image_file.filename)
        image_file.save(image_path)

    results = search_images(text_query, image_path, query_type, weight, use_pca, k)
    return jsonify(results)



from flask import send_from_directory

@app.route('/uploads/<path:filename>')
def serve_uploaded_image(filename):
    return send_from_directory('uploads', filename)


if __name__ == "__main__":
    app.run(debug=True)
