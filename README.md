# Preliminary Steps:
Edit the environmental variable to include all required keys:
ROOT <- The root of this directory
CLIENT_ID, CLIENT_SECRET, USER_AGENT <- API keys and values required to use PRAW

# To train:
First run ./Training/inputs.py to generate and fill all folders in Data
Next, run ./Training/train_models.py to train the 4 models covered in the report

WARNING: Train_models is currently forced to GPU. Currently it will use approximately 16GB RAM and 20GB VRAM at its peak. 

# To test:
There are pretrained models in ./Models corresponding to the ones used to generate submission statistics.
Once you generate data, ./Test/benchmark.py can be run to see relevant metrics

# To predict:
Run ./Inference/predict.py and follow the input instructions. 
The predictions for each model will be printed out soon
