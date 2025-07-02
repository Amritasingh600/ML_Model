# app.py

from flask import Flask, render_template, request
from backend import predict_tumor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        # Collect 30+ inputs from the form
        input_features = []
        for feature in [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
            "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
            "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
            "perimeter_se", "area_se", "smoothness_se", "compactness_se",
            "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
            "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst",
            "symmetry_worst", "fractal_dimension_worst"
        ]:
            input_features.append(float(request.form[feature]))
        
        # Call your backend prediction
        prediction = predict_tumor(input_features)
        
        result = f'Tumor is likely: {prediction}'
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
