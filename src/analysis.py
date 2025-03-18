import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Used ChatGBT to help me with the format of plotting my data and making it legible

#Loads CSV files
df_10 = pd.read_csv('/repo/ez-diffusion-model/data/results_n10.csv')
df_40 = pd.read_csv('/repo/ez-diffusion-model/data/results_n40.csv')
df_400 = pd.read_csv('/repo/ez-diffusion-model/data/results_n4000.csv')

#Calculates the bias and squared error
df_10['v_bias'] = df_10['v_true'] - df_10['v_est']
df_10['a_bias'] = df_10['a_true'] - df_10['a_est']
df_10['t_bias'] = df_10['t_true'] - df_10['t_est']
df_10['v_squared_error'] = (df_10['v_true'] - df_10['v_est'])**2
df_10['a_squared_error'] = (df_10['a_true'] - df_10['a_est'])**2
df_10['t_squared_error'] = (df_10['t_true'] - df_10['t_est'])**2

df_40['v_bias'] = df_40['v_true'] - df_40['v_est']
df_40['a_bias'] = df_40['a_true'] - df_40['a_est']
df_40['t_bias'] = df_40['t_true'] - df_40['t_est']
df_40['v_squared_error'] = (df_40['v_true'] - df_40['v_est'])**2
df_40['a_squared_error'] = (df_40['a_true'] - df_40['a_est'])**2
df_40['t_squared_error'] = (df_40['t_true'] - df_40['t_est'])**2

df_400['v_bias'] = df_400['v_true'] - df_400['v_est']
df_400['a_bias'] = df_400['a_true'] - df_400['a_est']
df_400['t_bias'] = df_400['t_true'] - df_400['t_est']
df_400['v_squared_error'] = (df_400['v_true'] - df_400['v_est'])**2
df_400['a_squared_error'] = (df_400['a_true'] - df_400['a_est'])**2
df_400['t_squared_error'] = (df_400['t_true'] - df_400['t_est'])**2

#Computes average bias and squared error for each N
bias_10 = df_10['v_bias'].mean()
bias_40 = df_40['v_bias'].mean()
bias_400 = df_400['v_bias'].mean()

squared_error_10 = df_10['v_squared_error'].mean()
squared_error_40 = df_40['v_squared_error'].mean()
squared_error_400 = df_400['v_squared_error'].mean()

a_squared_error_10 = df_10['a_squared_error'].mean()
a_squared_error_40 = df_40['a_squared_error'].mean()
a_squared_error_400 = df_400['a_squared_error'].mean()

t_squared_error_10 = df_10['t_squared_error'].mean()
t_squared_error_40 = df_40['t_squared_error'].mean()
t_squared_error_400 = df_400['t_squared_error'].mean()

#Prints results
print(f"Bias for N=10: {bias_10}")
print(f"Bias for N=40: {bias_40}")
print(f"Bias for N=400: {bias_400}")
print(f"Squared Error for N=10: {squared_error_10}")
print(f"Squared Error for N=40: {squared_error_40}")
print(f"Squared Error for N=400: {squared_error_400}")
print(f"A Squared Error for N=10: {a_squared_error_10}")
print(f"A Squared Error for N=40: {a_squared_error_40}")
print(f"A Squared Error for N=400: {a_squared_error_400}")
print(f"T Squared Error for N=10: {t_squared_error_10}")
print(f"T Squared Error for N=40: {t_squared_error_40}")
print(f"T Squared Error for N=400: {t_squared_error_400}")

#Plots bias and squared error across N
n_values = [10, 40, 400]
bias_values = [bias_10, bias_40, bias_400]
squared_error_values = [squared_error_10, squared_error_40, squared_error_400]

plt.figure(figsize=(12, 8))

#Plots Bias
plt.subplot(2, 2, 1)
plt.plot(n_values, bias_values, marker='o', linestyle='-', color='r')
plt.title('Bias vs Sample Size (N)')
plt.xlabel('Sample Size (N)')
plt.ylabel('Bias')

#Plots Squared Error (v)
plt.subplot(2, 2, 2)
plt.plot(n_values, squared_error_values, marker='o', linestyle='-', color='b')
plt.title('Squared Error (v) vs Sample Size (N)')
plt.xlabel('Sample Size (N)')
plt.ylabel('Squared Error (v)')

#Plots Squared Error (a)
a_squared_error_values = [a_squared_error_10, a_squared_error_40, a_squared_error_400]
plt.subplot(2, 2, 3)
plt.plot(n_values, a_squared_error_values, marker='o', linestyle='-', color='g')
plt.title('Squared Error (a) vs Sample Size (N)')
plt.xlabel('Sample Size (N)')
plt.ylabel('Squared Error (a)')

#Plots Squared Error (t)
t_squared_error_values = [t_squared_error_10, t_squared_error_40, t_squared_error_400]
plt.subplot(2, 2, 4)
plt.plot(n_values, t_squared_error_values, marker='o', linestyle='-', color='purple')
plt.title('Squared Error (t) vs Sample Size (N)')
plt.xlabel('Sample Size (N)')
plt.ylabel('Squared Error (t)')

plt.tight_layout()
plt.show()
