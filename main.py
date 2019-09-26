#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import minmax_scale, OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


df = pd.read_csv('data/data.csv', infer_datetime_format=True)

# Sort data in time to reduce autocorrelation between train and test datasets
# when splitting into stratified subsets
df = df.sort_values(by=['date']).drop('date', axis=1)

# Subset variables for model training
df = df[['respiration', 'day_of_year', 't_air_era5', 't_soil_era5', 'gpp_model_mean', 'land_use']]


# Split columns to features and labels
features = df
labels = features.pop('respiration')

# Normalize each numeric feature column
for k in features.columns:
    if np.issubdtype(features.loc[:, k].dtype, np.number):
        features.loc[:, k] = minmax_scale(features.loc[:, k])

# Use one hot encoding for categorical features
# land_use = features.land_use.values.reshape(-1, 1)
# enc = OneHotEncoder(sparse=False).fit(land_use)
# land_use_ohe = pd.DataFrame(enc.transform(
#     land_use), columns=enc.categories_[0])
# features = pd.concat((features, land_use_ohe), axis=1)
features = features.drop('land_use', axis=1)


print(pd.concat((features, labels), axis=1))


# Split dataset into train/test features/labels
x_train, x_test, y_train, y_test = train_test_split(
    features.values,
    labels.values,
    shuffle=False,
    test_size=0.2
)


def build_model(n_layers: int = 2, n_units_per_layer: int = 64):
    """Instantiate model template
    
    Args:
        n_layers (int): number of dense interconnected layers prior to output
        n_units_per_layer (int): interconnected neurons in each dense layer
    Returns:
        keras.Sequential: compiled model object
    """
    model_layers = [layers.Dense(n_units_per_layer,
                                activation=tf.nn.relu,
                                input_shape=[x_train.shape[1]])]
    for _ in range(n_layers - 1):
        model_layers.append(layers.Dense(n_units_per_layer, activation=tf.nn.relu))
    
    model_layers.append(layers.Dense(1))

    model = keras.Sequential(model_layers)
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


# Tunable hyperparameters are fed to build_model(**kwargs) and optimized by
# minimizing mean absolute error.
model_params = [
    {'n_layers': 1, 'n_units_per_layer': 16},
    {'n_layers': 1, 'n_units_per_layer': 32},
    {'n_layers': 1, 'n_units_per_layer': 64},
    {'n_layers': 2, 'n_units_per_layer': 16},
    {'n_layers': 2, 'n_units_per_layer': 32},
    {'n_layers': 2, 'n_units_per_layer': 64},
    {'n_layers': 3, 'n_units_per_layer': 16},
    {'n_layers': 3, 'n_units_per_layer': 32},
    {'n_layers': 3, 'n_units_per_layer': 64}
]


def train_model(train_index, val_index):
    """Train model on subset of training data

    Args:
        train_index (np.array): index of rows used to train model
        val_index (np.array): rows used to calculate loss for early stopping
    Returns:
        tuple: (mae, rmse) error metrics from validation fold
    """
    model.fit(
        x_train[train_index],
        y_train[train_index],
        epochs=1000,
        validation_split=0.2,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)]
    )

    pred = model.predict(x_train[val_index]).flatten()
    e = pred - y_train[val_index]
    return (np.abs(e).mean(), np.sqrt(np.mean(e**2)))
    

n_folds = 5
skf = KFold(n_folds)
metrics = []

for kwargs in model_params:
    model = build_model(**kwargs)

    k_metrics = []
    for i, (train_index, val_index) in enumerate(skf.split(x_train, y_train)):
        print(f'Training model: {kwargs}, {i+1}/{n_folds}')
        mae, rmse = train_model(train_index, val_index)
        k_metrics.append((mae, rmse))

    k_metrics = np.array(k_metrics)
    metrics.append({
        'kwargs': kwargs,
        'mae': k_metrics[:, 0].mean(),
        'mae_sd': k_metrics[:, 0].std(),
        'rmse': k_metrics[:, 1].mean(),
        'rmse_sd': k_metrics[:, 1].std()
    })


print('Model uncertainties:')
for x in metrics: print(x)


# Choose hyperparameter set based on minimum MAE value
minimum = None
for x in metrics:
    if not minimum or x['mae'] < minimum:
        kwargs = x['kwargs']


print(f'Model selection: {kwargs}')
model = build_model(**kwargs)
history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    ])


# Plot train and validation error as a function of epoch
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error')
plt.plot(hist['epoch'], hist['mean_absolute_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
         label='Val Error')
plt.legend()

plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.plot(hist['epoch'], hist['mean_squared_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_squared_error'],
         label='Val Error')
plt.legend()
plt.savefig('train_error.png', dpi=300)
plt.close('all')


pred = model.predict(x_test).flatten()
plt.scatter(y_test, pred, alpha=0.1)
plt.xlabel('Known Values')
plt.ylabel('Model Predictions')
plt.axis('equal')
plt.axis('square')
plt.savefig('test_error.png', dpi=300)
plt.close('all')


loss, mae, mse = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {loss}')
print(f'Test MAE: {mae}')
print(f'Test MSE: {mse}')
print(f'Test r-square: {np.corrcoef(pred, y_test)[0, 1]**2}')


path = 'model.h5'
model.save(path)
print(f'Model saved to: {path}')
