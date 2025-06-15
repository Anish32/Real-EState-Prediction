
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import FunctionTransformer 
from sklearn.pipeline import Pipeline 
pipeline_filename = 'global_pipeline.pkl'
country_list_filename = 'country_list.pkl'
states_dict_filename = 'states_by_country.pkl'
cities_dict_filename = 'cities_by_state.pkl'
feature_importance_filename = 'feature_importance.pkl'
sector_list_filename = 'sector_list.pkl'
mae_filename = 'mae_global.pkl' 
pipeline = None
country_list = []
states_by_country = {}
cities_by_state = {}
feature_importance_dict = {}
sector_list = []
global_mae = None 
categorical_features = ['Country', 'State', 'City', 'Sector', 'Servant Room', 'Storeroom',
                        'Furnishing Type', 'Luxury', 'Floor Category']
# Define numerical features list
numerical_features = ['total_sqft', 'bath', 'BHK', 'Property Age', 'Water Source Proximity',
                      'School Rating Proximity']

# Define the custom function used in the pipeline
def to_category_dtype(X):
    X_copy = X.copy()
    for col in categorical_features:
        if col in X_copy.columns:
            # Use .copy() to avoid SettingWithCopyWarning
            X_copy[col] = X_copy[col].astype('category')
    return X_copy

# --- Load Saved Data ---
try:
    # Load the pipeline
    if os.path.exists(pipeline_filename):
        # Pass custom objects/functions to joblib.load
        pipeline = joblib.load(pipeline_filename, mmap_mode=None)
        st.success("Model loaded successfully!")
    else:
        st.error(f"Model file '{pipeline_filename}' not found. Please train and save the model first.")

    # Load MAE
    if os.path.exists(mae_filename):
        global_mae = joblib.load(mae_filename)
        st.info(f"Global Mean Absolute Error loaded: {global_mae:.2f}")
    else:
        st.warning(f"MAE file '{mae_filename}' not found. Cannot provide an estimated price range based on MAE.")
        global_mae = None # Set MAE to None if file not found
        # Provide a default or handle this case in the prediction display


    # Load location data
    if os.path.exists(country_list_filename):
        country_list = joblib.load(country_list_filename)
        if not country_list:
            st.warning("Country list loaded but is empty.")
            country_list = ['No Countries Found']
    else:
        st.warning(f"Country list file '{country_list_filename}' not found.")
        country_list = ['Loading Failed']

    if os.path.exists(states_dict_filename):
        states_by_country = joblib.load(states_dict_filename)
        if not states_by_country:
             st.warning("States dictionary loaded but is empty.")
             states_by_country = {'No States Found': ['Select Country']}
    else:
         st.warning(f"States dictionary file '{states_dict_filename}' not found.")
         states_by_country = {'Loading Failed': ['State 1']}

    if os.path.exists(cities_dict_filename):
        cities_by_state = joblib.load(cities_dict_filename)
        if not cities_by_state:
             st.warning("Cities dictionary loaded but is empty.")
             cities_by_state = {'No Cities Found': ['Select State']}
    else:
         st.warning(f"Cities dictionary file '{cities_dict_filename}' not found.")
         cities_by_state = {'State 1': ['City 1']}

    # Load sector list
    if os.path.exists(sector_list_filename):
        sector_list = joblib.load(sector_list_filename)
        if not sector_list:
            st.warning("Sector list loaded but is empty.")
            sector_list = ['No Sectors Found']
    else:
         st.warning(f"Sector list file '{sector_list_filename}' not found.")
         sector_list = ['Loading Failed']

    # Load feature importance
    if os.path.exists(feature_importance_filename):
        feature_importance_dict = joblib.load(feature_importance_filename)
        if not feature_importance_dict:
            st.info("Feature importance data is empty.")
            feature_importance_dict = {} # Ensure it's an empty dict if file is empty
    else:
         st.warning(f"Feature importance file '{feature_importance_filename}' not found.")
         feature_importance_dict = {} # Ensure it's an empty dict if file is missing


except Exception as e:
    st.error(f"An error occurred while loading model or data files: {e}")
    pipeline = None
    country_list = ['Error Loading']
    states_by_country = {'Error Loading': ['Check Files']}
    cities_by_state = {'Check Files': ['Error Loading']}
    feature_importance_dict = {}
    sector_list = ['Error Loading']
    global_mae = None


# Streamlit App Title
st.title("House Price Prediction")

st.write("Select the property details and location and land sector type to get a price prediction based on the trained model.")

# Check if model loaded successfully before proceeding
if pipeline is None:
    st.info("Model could not be loaded. Please ensure the training script ran successfully and generated the required .pkl files.")
else:
    # --- User Input Section ---
    st.sidebar.header("Property Details and Location")

    # Select Country
    selected_country = st.sidebar.selectbox("Select Country", country_list)

    # Select State based on selected Country
    state_list = states_by_country.get(selected_country, ['Select a Country first'])
    selected_state = st.sidebar.selectbox("Select State", state_list)

    # Select City based on selected State
    city_list = cities_by_state.get(selected_state, ['Select a State first'])
    selected_city = st.sidebar.selectbox("Select City", city_list)

    # Select Sector
    selected_sector = st.sidebar.selectbox("Select Land Sector Type", sector_list)

    total_sqft = st.sidebar.number_input("Total Square Feet", min_value=100, max_value=100000, value=1200)
    bhk = st.sidebar.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=20, value=3)
    bath = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=20, value=2)
    property_age = st.sidebar.number_input("Property Age (Years)", min_value=0, max_value=100, value=5)
    servant_room = st.sidebar.selectbox("Servant Room", ["Yes", "No"])
    storeroom = st.sidebar.selectbox("Storeroom", ["Yes", "No"])
    furnishing_type = st.sidebar.selectbox("Furnishing Type", ["Furnished", "Semi-Furnished", "Unfurnished"])
    luxury = st.sidebar.selectbox("Luxury", ["High", "Medium", "Low", "None"])
    floor_category = st.sidebar.selectbox("Floor Category", ["Ground Floor", "Low Rise", "Mid Rise", "High Rise"])

    # Add conditional dummy features input
    water_source_proximity = st.sidebar.number_input("Water Source Proximity (1-10, Lower is better, 0 if not agricultural)", min_value=0.0, max_value=10.0, value=0.0)
    school_rating_proximity = st.sidebar.number_input("School Rating Proximity (1-5, Lower is better, 0 if not residential)", min_value=0.0, max_value=5.0, value=0.0)


    if st.sidebar.button("Predict Price"):
        # Check if valid selections were made
        if 'Loading Failed' in [selected_country, selected_state, selected_city, selected_sector] or \
           'Error Loading' in [selected_country, selected_state, selected_city, selected_sector] or \
           'No Countries Found' in selected_country or \
           selected_state == 'Select a Country first' or \
           selected_city == 'Select a State first' or \
           'No States Found' in selected_state or \
           'No Cities Found' in selected_city or \
           'No Sectors Found' in selected_sector:
             st.warning("Please select a valid Country, State, City, and Land Sector Type before predicting.")
        else:
            # Create a DataFrame from user input
            # Ensure columns match the training data used to train the pipeline
            input_data = pd.DataFrame([[selected_country, selected_state, selected_city, selected_sector,
                                        total_sqft, bath, bhk, property_age, servant_room, storeroom,
                                        furnishing_type, luxury, floor_category,
                                        water_source_proximity, school_rating_proximity]],
                                      columns=['Country', 'State', 'City', 'Sector',
                                               'total_sqft', 'bath', 'BHK', 'Property Age', 'Servant Room',
                                               'Storeroom', 'Furnishing Type', 'Luxury', 'Floor Category',
                                               'Water Source Proximity', 'School Rating Proximity'])

            # Debug: Show input data structure
            st.write("Input Data for Prediction:")
            st.dataframe(input_data)
            st.write(f"Input Data Types:\n{input_data.dtypes}")


            try:
                # Make prediction using the loaded pipeline
                predicted_price = pipeline.predict(input_data)[0]

                # Display the predicted price and range using rate terminology
                st.subheader("Estimated Price Rate")
                st.write(f"Predicted Rate: ₹{predicted_price:,.2f} Lakhs")

                if global_mae is not None:
                    lower_bound = predicted_price - global_mae
                    upper_bound = predicted_price + global_mae
                    st.write(f"Estimated Rate Range (± MAE): ₹{max(0, lower_bound):,.2f} Lakhs to ₹{upper_bound:,.2f} Lakhs")
                else:
                    st.info("Estimated price rate range is not available because MAE was not loaded.")


                # --- Generate Data for Line Chart (Varying total_sqft) ---
                st.subheader("Estimated Price Rate vs. Total Square Feet")
                num_points = 50 # Number of points for the line chart
                # Create a range of total_sqft values around the user input
                sqft_min = max(100, total_sqft * 0.5) # Ensure minimum is at least 100
                sqft_max = total_sqft * 1.5
                sqft_range = np.linspace(sqft_min, sqft_max, num_points)

                # Create DataFrames for prediction, varying total_sqft
                plot_data = []
                for sqft in sqft_range:
                     # Create a copy of the input data and modify total_sqft
                     temp_input_data = input_data.copy()
                     temp_input_data['total_sqft'] = sqft

                     # Predict price for this sqft value
                     try:
                         predicted_price_for_sqft = pipeline.predict(temp_input_data)[0]
                         plot_data.append({'total_sqft': sqft, 'predicted_price': predicted_price_for_sqft})
                     except Exception as e:
                         # Handle potential errors during prediction for a range of values
                         # st.warning(f"Could not predict for sqft={sqft:.2f}: {e}") # Comment out this warning to avoid spamming
                         plot_data.append({'total_sqft': sqft, 'predicted_price': np.nan}) # Add NaN to plot

                plot_df = pd.DataFrame(plot_data).dropna() # Drop rows with NaN predictions

                # Debug: Show plot data info
                st.write("Plot Data Info:")
                st.write(f"Shape: {plot_df.shape}")
                st.write(f"Head:\n{plot_df.head()}")


                if not plot_df.empty:
                     # Create the line chart
                     fig, ax = plt.subplots(figsize=(10, 6))
                     sns.lineplot(x='total_sqft', y='predicted_price', data=plot_df, ax=ax)
                     ax.set_title('Predicted Price Rate vs. Total Square Feet')
                     ax.set_xlabel("Total Square Feet")
                     ax.set_ylabel("Estimated Price Rate (Lakh ₹)") # Adjusted label
                     ax.grid(True)
                     st.pyplot(fig)
                else:
                    st.warning("Could not generate data for the price vs. sqft plot. Check if predictions failed for the generated sqft range.")


                # --- Display Feature Importance ---
                if feature_importance_dict:
                    st.subheader("Feature Importance (Top 10)")
                    # Sort feature importance and get top 10
                    sorted_importance = sorted(feature_importance_dict.items(), key=operator.itemgetter(1), reverse=True)[:10]
                    importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])

                    # Debug: Show importance data info
                    st.write("Feature Importance Data Info:")
                    st.write(f"Shape: {importance_df.shape}")
                    st.write(f"Head:\n{importance_df.head()}")


                    if not importance_df.empty:
                        # Create a bar plot in the app
                        fig, ax = plt.subplots(figsize=(8, 5))
                        # Modified sns.barplot to use hue and legend=False
                        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis', hue='Feature', legend=False)
                        ax.set_title('Top 10 Feature Importances')
                        ax.set_xlabel('Importance Score')
                        ax.set_ylabel('Feature')
                        st.pyplot(fig) # Display the plot in Streamlit
                    else:
                        st.info("Feature importance data is empty or contains no relevant features.")

                else:
                    st.info("Feature importance data not available or empty.")


            except ValueError as ve:
                 st.error(f"Error during prediction: {ve}. This might be due to unseen categorical values. Please ensure the selected location and sector were present in the training data or handle unknown values in your pipeline (which handle_unknown='ignore' should help with, but other data issues are possible).")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
