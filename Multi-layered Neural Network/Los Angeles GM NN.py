#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


# Load the data
ColumnsData_13016 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_13016.xlsx")
ColumnsData_13104 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_13104.xlsx")
ColumnsData_13168 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/ColumnsData/ColumnsData_13168.xlsx")

GMData_13016 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_13016.xlsx")
GMData_13104 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_13104.xlsx")
GMData_13168 = pd.read_excel("~/Downloads/SoCal Data Science Program Team 3/SharedData/GMData/GMData_13168.xlsx")

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
columns_13016 = clean_columns_data(ColumnsData_13016)
columns_13104 = clean_columns_data(ColumnsData_13104)
columns_13168 = clean_columns_data(ColumnsData_13168)

gm_13016 = clean_gm_data(GMData_13016)
gm_13104 = clean_gm_data(GMData_13104)
gm_13168 = clean_gm_data(GMData_13168)
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

col_gm_13016 = join_and_select_columns(columns_13016, gm_13016)
col_gm_13104 = join_and_select_columns(columns_13104, gm_13104)
col_gm_13168 = join_and_select_columns(columns_13168, gm_13168)


# Remove the DI value which are out of ranges
def remove_out_of_range_di(data):
    return data[(data['DI'] >= 0) & (data['DI'] <= 1)]

col_gm_13016_r = remove_out_of_range_di(col_gm_13016)
col_gm_13104_r = remove_out_of_range_di(col_gm_13104)
col_gm_13168_r = remove_out_of_range_di(col_gm_13168)


# Combine all los angeles
col_gm_combined_la = pd.concat([col_gm_13016_r, col_gm_13104_r, col_gm_13168_r], ignore_index=True)


# Ensure 'DI' is included in the selected variables
selected_vars = col_gm_combined_la[['DI','AI', 'PGV', 'PGA', 'D575', 'CAV', 'HCol', 'PcFcAg', 'RR', 'DCol', 'DIrp', 'Sa', 'T1', 'DIesa', 'fcu', 'Dy','Ke']]

# Split the data into training and testing sets
train_data, test_data = train_test_split(selected_vars, test_size=0.2, random_state=123)

# Separate predictors and response
x_train = train_data[['AI', 'PGV', 'PGA', 'D575', 'CAV', 'HCol', 'PcFcAg', 'RR', 'DCol', 'DIrp', 'Sa', 'T1', 'DIesa', 'fcu', 'Dy','Ke']]
y_train = train_data['DI']
x_test = test_data[['AI', 'PGV', 'PGA', 'D575', 'CAV', 'HCol', 'PcFcAg', 'RR', 'DCol', 'DIrp', 'Sa', 'T1', 'DIesa', 'fcu', 'Dy','Ke']]
y_test = test_data['DI']

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[5]:


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


# In[6]:


# Initialize the tuner
tuner = RandomSearch(
    MyHyperModel(),
    objective='val_mean_squared_error',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir_2',
    project_name='helloworld'
)

# Perform hyperparameter tuning
tuner.search(x_train_scaled, y_train, epochs=50, validation_data=(x_test_scaled, y_test))

tuner.search_space_summary()
tuner.results_summary()


# In[7]:


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


# In[8]:


y_pred = model.predict(x_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")


# In[9]:


# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()


# In[10]:


# Evaluate the model
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print(f"Test Loss: {loss}")

# Calculate RMSE
y_pred = model.predict(x_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse}")


# In[11]:


# Make predictions on the test data
predictions = model.predict(x_test_scaled)

# Create a DataFrame with actual and predicted values
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(predictions_df['Actual'], predictions_df['Predicted'], alpha=0.5)
plt.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         color='red', linestyle='dashed')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual DI')
plt.ylabel('Predicted DI')
plt.show()


# In[12]:


#Neural Network Structure
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

Image(filename='model_structure.png')


# In[13]:


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
plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', fontsize=12, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
plt.text(0.05, 0.90, f'R-squared: {r_squared:.2f}', fontsize=12, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

# Add titles and labels
plt.xlabel('Predicted DI', fontsize=14)
plt.ylabel('Actual DI', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()


# In[ ]:




