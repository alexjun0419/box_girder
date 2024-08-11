#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras_tuner import HyperModel, RandomSearch
from tensorflow.keras.utils import plot_model
from IPython.display import Image
import keras_tuner as kt


# In[26]:


# Load the data
ColumnsData_10828 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_10828.xlsx")
ColumnsData_10879 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_10879.xlsx")
ColumnsData_10924 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_10924.xlsx")
ColumnsData_10986 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_10986.xlsx")
ColumnsData_11024 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_11024.xlsx")

GMData_10828 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_10828.xlsx")
GMData_10879 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_10879.xlsx")
GMData_10924 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_10924.xlsx")
GMData_10986 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_10986.xlsx")
GMData_11024 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_11024.xlsx")

AllLocations = pd.read_csv("~/Downloads/SoCal Data Science Program Team 3/SharedData/AllLocations.csv")

def clean_columns_data(data):
    data['CID'] = data['CID'].astype('category')
    for col in ['HCol', 'PcFcAg', 'NBar', 'DBar', 'RR', 'DCol', 'Ag', 'NPiles', 'HSize', 'HSpc', 'RP']:
        data[col] = data[col].astype('category')
    data['GMID'] = data['GMID'].str.replace(r'GM(\d{1})$', r'GM0\1').astype('category')
    return data

def clean_gm_data(data):
    data['TID'] = data['TID'].astype('category')
    data['RP'] = data['RP'].astype('category')
    data['RPID'] = data['RPID'].astype('category')
    data['EventID'] = data['EventID'].astype('category')
    data['GMID'] = data['GMID'].str.replace(r'GM(\d{1})$', r'GM0\1').astype('category')
    return data

def clean_locations(data):
    data['Grid'] = data['Grid'].astype('category')
    return data

# Clean the datasets
columns_10828 = clean_columns_data(ColumnsData_10828)
columns_10879 = clean_columns_data(ColumnsData_10879)
columns_10924 = clean_columns_data(ColumnsData_10924)
columns_10986 = clean_columns_data(ColumnsData_10986)
columns_11024 = clean_columns_data(ColumnsData_11024)


gm_10828 = clean_gm_data(GMData_10828)
gm_10879 = clean_gm_data(GMData_10879)
gm_10924 = clean_gm_data(GMData_10924)
gm_10986 = clean_gm_data(GMData_10986)
gm_11024 = clean_gm_data(GMData_11024)
locations = clean_locations(AllLocations)

# Joining GM and Column Data
def join_and_select_columns(columns_data, gm_data):
    merged_data = columns_data.merge(gm_data, how='left', left_on=['T1', 'RP', 'GMID'], right_on=['Time', 'RP', 'GMID'])
    selected_columns = [
        'CID', 'T1', 'TID', 'GMID', 'RP', 'RPID', 'EventID', 'M', 'Rrup', 'Rx', 'Ztor', 'Rake', 'Dip', 'Z1', 'W', 'Az', 'PGA', 'PGV', 'CAV', 'AI', 'D575', 'D595', 
        'HCol', 'PcFcAg', 'WTop', 'NBar', 'DBar', 'RR', 'DCol', 'Ag', 'fcc', 'fcu', 'ecc', 'ecu', 'HSize', 'HSpc', 'Ke', 'Rho', 'Du', 'Dy', 'Desa', 'DIesa', 'Sa', 
        'DIrp', 'DI', 'Dmax', 'Dxmax', 'Dymax'
    ]
    return merged_data[selected_columns]

col_gm_10828 = join_and_select_columns(columns_10828, gm_10828)
col_gm_10879 = join_and_select_columns(columns_10879, gm_10879)
col_gm_10924 = join_and_select_columns(columns_10924, gm_10924)
col_gm_10986 = join_and_select_columns(columns_10986, gm_10986)
col_gm_11024 = join_and_select_columns(columns_11024, gm_11024)



# Remove the DI value which are out of ranges
def remove_out_of_range_di(data):
    return data[(data['DI'] >= 0) & (data['DI'] <= 1)]

col_gm_10828_r = remove_out_of_range_di(col_gm_10828)
col_gm_10879_r = remove_out_of_range_di(col_gm_10879)
col_gm_10924_r = remove_out_of_range_di(col_gm_10924)
col_gm_10986_r = remove_out_of_range_di(col_gm_10986)
col_gm_11024_r = remove_out_of_range_di(col_gm_11024)



# Combine all bay area
col_gm_combined_bay = pd.concat([col_gm_10828_r, col_gm_10879_r, col_gm_10924_r, col_gm_10986_r, col_gm_11024_r], ignore_index=True)


# Ensure 'DI' is included in the selected variables
selected_vars = col_gm_combined_bay[['RP','DI','AI','Ke','M', 'Rx', 'Az', 'Dip', 'PGV', 'PGA', 'D575', 'CAV', 'HCol', 'PcFcAg', 'RR', 'DCol', 'DIrp', 'Sa', 'T1', 'DIesa', 'fcu', 'Dy']]

# Split the data into training and testing sets
train_data, test_data = train_test_split(selected_vars, test_size=0.2, random_state=123)

# Separate predictors and response
x_train = train_data[['RP','AI','Ke','M', 'Rx', 'Az', 'Dip', 'PGV', 'PGA', 'D575', 'CAV', 'HCol', 'PcFcAg', 'RR', 'DCol', 'DIrp', 'Sa', 'T1', 'DIesa', 'fcu', 'Dy']]
y_train = train_data['DI']
x_test = test_data[['RP','AI','Ke','M', 'Rx', 'Az', 'Dip', 'PGV', 'PGA', 'D575', 'CAV', 'HCol', 'PcFcAg', 'RR', 'DCol', 'DIrp', 'Sa', 'T1', 'DIesa', 'fcu', 'Dy']]
y_test = test_data['DI']

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[27]:


# Define a hypermodel class
class MyHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_dim=x_train_scaled.shape[1]))
        
        for i in range(hp.Int('num_layers', 1, 5)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
        
        model.add(Dense(1, activation= 'sigmoid'))
        
        model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
        return model


# In[28]:


# Initialize the tuner
tuner = RandomSearch(
    MyHyperModel(),
    objective='val_mean_squared_error',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='helloworld'
)

# Perform hyperparameter tuning
tuner.search(x_train_scaled, y_train, epochs=50, validation_data=(x_test_scaled, y_test))

tuner.search_space_summary()
tuner.results_summary()


# In[29]:


# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model
history = model.fit(x_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(x_test_scaled, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print(f"Test Loss: {loss}")


# Calculate RMSE
y_pred = model.predict(x_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")


# In[30]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


# In[31]:


# Evaluate the model
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print(f"Test Loss: {loss}")

# Calculate RMSE
y_pred = model.predict(x_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")


# In[42]:


# Make predictions on the test data
predictions = model.predict(x_test_scaled)

# Create a DataFrame with actual and predicted values
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(predictions_df['Actual'], predictions_df['Predicted'], alpha=0.6, edgecolors='w', linewidth=0.5, color='Green')
plt.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         color='red', linestyle='dashed')
plt.xlabel('Actual DI')
plt.ylabel('Predicted DI')
#plt.gcf().patch.set_facecolor('none')
#plt.gca().patch.set_facecolor('none')

plt.show()


# In[33]:


#Neural Network Structure
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

Image(filename='model_structure.png')


# In[43]:


import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Make predictions on the test data
predictions = model.predict(x_test_scaled)

# Create a DataFrame with actual and predicted values
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})

# Calculate the correlation coefficient and R-squared value
correlation = np.corrcoef(predictions_df['Actual'], predictions_df['Predicted'])[0, 1]
r_squared = r2_score(predictions_df['Actual'], predictions_df['Predicted'])

# Plot actual vs. predicted values with correlation and R-squared annotation
plt.figure(figsize=(10, 6))

# Plot scatter plot
plt.scatter(predictions_df['Predicted'], predictions_df['Actual'], alpha=0.6, edgecolors='w', linewidth=0.5)

# Plot the diagonal line
plt.plot([predictions_df['Predicted'].min(), predictions_df['Predicted'].max()],
         [predictions_df['Predicted'].min(), predictions_df['Predicted'].max()],
         color='red', linestyle='dashed', linewidth=2)

# Annotate the plot with correlation and R-squared

# Add titles and labels
plt.xlabel('Predicted DI', fontsize=14)
plt.ylabel('Actual DI', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()


# In[36]:





# In[ ]:




