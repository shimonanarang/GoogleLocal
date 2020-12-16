# GoogleLocal

Directions:

Downloading data:
Visit https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local and download Places Data (276mb), User Data (178mb), and Review Data (1.4gb).

Dataset and Preprocessing:
1. Place the files in the load_and_preprocess_data folder.
2. Run 'load_and_preprocess_review_data.py' to get restaurants.pickle. This contains preprocessed restaurant reviews in pandas dataframe format. 
3. Run 'dataset_and_EDA.ipynb' to get restaurants_places.pickle' which contains the merged data for reviews, places and user data. This is our final data. 

Models:
1. All models are self sufficient. Please place 'restaurants_places.pickle' in the models folder before executing. 
2. To run AutoEncoder Collaborative Filter, run AutoEnc CF/main.py. It uses data in .dat file and a reference data sample for Google Local Dataset is uploaded


# References
1. https://github.com/gtshs2/Autorec







