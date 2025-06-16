import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate synthetic dataset with multicollinearity
np.random.seed(0)
X1 = np.random.rand(100) * 10
X2 = 0.8 * X1 + np.random.rand(100) * 2  # correlated with X1
X3 = np.random.rand(100) * 10
noise = np.random.normal(0, 2, 100)
Y = 3 + 2 * X1 + 1.5 * X2 + 0.5 * X3 + noise

# Create DataFrame
data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

# Explore Data
print("Correlation Matrix:")
print(data.corr())
sns.heatmap(data.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# Scatterplot Matrix
sns.pairplot(data)
plt.suptitle("Scatterplot Matrix", y=1.02)
plt.show()

# Split data
X = data[['X1', 'X2', 'X3']]
y = data['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Multiple Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate Model
print(f"R-squared: {r2_score(y_test, y_pred):.3f}")
print(f"Adjusted R-squared: {1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1):.3f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.3f}")

# Statsmodels for Full Summary and ANOVA
data = pd.concat([X, y], axis=1)
ols_model = smf.ols('Y ~ X1 + X2 + X3', data=data).fit()
anova_results = anova_lm(ols_model, typ=2)
print(anova_results)


# Model Diagnostics - Residual Plot
residuals = ols_model.resid
sns.residplot(x=ols_model.fittedvalues, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Check Multicollinearity (VIF)
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF for each feature:")
print(vif_data)