# Respiration Neural Network Model

Trains a neural network to predict respiration as a function of land use classification, day of year, air temperature, soil temperature, and modeled GPP.

> If python needs to be configured within the user scope, see [a7193b2bdd59e1b930efcb7facd6d12e](https://gist.github.com/benfasoli/a7193b2bdd59e1b930efcb7facd6d12e)

This uses the Keras API via Tensorflow with k-fold cross validation for hyperparameter optimization. Land use is the only categorical model feature and is one hot encoded prior to model training.

```bash
# Install dependencies
python3 -m venv env
source env/bin/activate

pip install -U -r requirements.txt

# Train model, takes ~10 minutes per defined hyperparameter set
./main.py
```
