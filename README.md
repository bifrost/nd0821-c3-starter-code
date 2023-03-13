
# Environment Set up
* Download and install conda if you don’t have it already.
    * Create a new environment
    * conda create -n [envname] "python=3.8.16"
	* Install requirements: pip install -r requirements.txt

# Tests
* To runthe tests
* pytest -rA -W error::UserWarning

# Model
* call train_model.py script to train the model
* python starter/starter/train_model.py

# Rest API local
* To startup the app local
* uvicorn main:app --reload
* URL: http://127.0.0.1:8000
* Docs: http://127.0.0.1:8000/docs


# API Deployment
* Push the changes to github and merge, the new app will automatic be deployed to (render)[https://render.com/]
* URL: https://cansus.onrender.com
