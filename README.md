# disaster_response

### Summary:
This project is a ML Classifier that creates labels for messages of natural disasters in order to make easier and more effective the response.

It includes the pre-trained model and the output is shown in a web app in which the user can input a message and see the response.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
# disaster_response
