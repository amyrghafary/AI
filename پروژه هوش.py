#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[2]:


# تابع برای ایجاد داده‌ها با تعداد نمونه و پارامترهای مشخص
def generate_linear_data(num_samples, noise_std):
    np.random.seed(42)  # برای تولید داده‌های قابل تکرار
    X = 2 * np.random.rand(num_samples, 1)
    y = 4 + 3 * X + np.random.normal(0, noise_std, size=(num_samples, 1))
    return X, y

# تعداد نمونه‌ها و انحراف استاندارد نویز مورد نظر
num_samples_list = [5, 25, 100, 1000]
noise_std = 1.0

# تولید و نمایش داده‌ها برای هر تعداد نمونه
for num_samples in num_samples_list:
    X, y = generate_linear_data(num_samples, noise_std)
    print(f"\nNumber of Samples: {num_samples}")
    print("X:", X.flatten())
    print("y:", y.flatten())


# In[3]:


# تابع برای آموزش مدل و رسم نمودار
def train_and_plot_models(X, y, degrees, num_samples):
    plt.figure(figsize=(15, 4))

    for i, degree in enumerate(degrees, 1):
        # اضافه کردن توان‌ها به ویژگی‌های ورودی
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)

        # آموزش مدل رگرسیون خطی
        model = LinearRegression()
        model.fit(X_poly, y)

        # پیش‌بینی با استفاده از مدل
        X_new = np.linspace(0, 2, 100).reshape(100, 1)
        X_new_poly = poly_features.transform(X_new)
        y_pred = model.predict(X_new_poly)

        # رسم نمودار
        plt.subplot(1, len(degrees), i)
        plt.scatter(X, y, color='blue', label='Actual Data')
        plt.plot(X_new, y_pred, color='red', linewidth=2, label=f'Degree {degree} Model')
        plt.title(f'Model with Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()

    plt.tight_layout()
    plt.show()


# In[4]:


# تولید و آموزش مدل‌ها برای هر تعداد نمونه
for num_samples in num_samples_list:
    X, y = generate_linear_data(num_samples, noise_std)
    degrees = [1, 4, 16]
    train_and_plot_models(X, y, degrees, num_samples)


# In[5]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# تابع برای آموزش مدل و محاسبه دقت نهایی
def train_and_evaluate_models(X, y, degrees):
    accuracies = []

    for degree in degrees:
        # اضافه کردن توان‌ها به ویژگی‌های ورودی
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)

        # آموزش مدل رگرسیون خطی
        model = LinearRegression()
        model.fit(X_poly, y)

        # محاسبه دقت نهایی مدل
        y_pred = model.predict(X_poly)
        accuracy = mean_squared_error(y, y_pred)
        accuracies.append(accuracy)

    return accuracies

# تولید و آموزش مدل‌ها برای هر تعداد نمونه
accuracies_list = []
for num_samples in num_samples_list:
    X, y = generate_linear_data(num_samples, noise_std)
    degrees = [1, 4, 16]
    accuracies = train_and_evaluate_models(X, y, degrees)
    accuracies_list.append(accuracies)

# رسم نمودار دقت نهایی مدل‌ها
plt.figure(figsize=(10, 6))
for i, degree in enumerate(degrees):
    plt.plot(num_samples_list, [acc[i] for acc in accuracies_list], marker='o', label=f'Degree {degree}')

plt.title('Final Model Accuracy for Different Sample Sizes')
plt.xlabel('Number of Samples')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()


# In[ ]:




