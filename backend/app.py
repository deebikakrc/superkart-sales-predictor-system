import joblib
import pandas as pd
from flask import Flask, request, jsonify
# Initialize Flask app with a name
superkart_api = Flask("SuperKart sales report predictor system")

# Load the trained model
model = joblib.load("superkart_model.joblib")

# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Total Sales Predictor System"

# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    product_features_json = request.get_json()

    # Extract relevant features from the input data
    extracted_product_features_json = {
    'Product_Weight': product_features_json['Product_Weight'],
    'Product_Sugar_Content': product_features_json['Product_Sugar_Content'],
    'Product_Allocated_Area': product_features_json['Product_Allocated_Area'],
    'Product_MRP': product_features_json['Product_MRP'],
    'Store_Size': product_features_json['Store_Size'],
    'Store_Location_City_Type': product_features_json['Store_Location_City_Type'],
    'Store_Type': product_features_json['Store_Type'],
    'Product_Id_char': product_features_json['Product_Id_char'],
    'Store_Age_Years': product_features_json['Store_Age_Years'],
    'Product_Type_Category': product_features_json['Product_Type_Category']
}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([extracted_product_features_json])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Total_Sales_Predicted': prediction})

# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for the batch data
    predictions = model.predict(input_data).tolist()

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return output_dict


# Run the Flask app in debug mode
if __name__ == '__main__':
    superkart_api.run(debug=True)
