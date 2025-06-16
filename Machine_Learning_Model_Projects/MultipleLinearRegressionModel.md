# Multiple Linear Regression Interview Prep

## What is Multiple Linear Regression?

A regression technique that models the relationship between one dependent variable and two or more independent variables.

## General Linear Regression Model

$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon$

## Matrix Representation

* $\mathbf{Y} = \mathbf{X\beta} + \epsilon$
* $\mathbf{\beta} = (X^TX)^{-1}X^TY$

## Least Squares in Matrix Form

Minimizes $\|Y - X\beta\|^2$. Solution: $\hat{\beta} = (X^TX)^{-1}X^TY$

## Types of Predictive Variables

* Continuous
* Categorical (converted via indicator variables/dummies)

## What is an F-test?

Tests whether at least one explanatory variable has a non-zero coefficient.

## Coefficient of Multiple Determination

R-squared: Proportion of variance in dependent variable explained by all predictors.

## Adjusted R-squared

Adjusts R² for the number of predictors, penalizing overfitting.

## Scatterplots

Visual tool to observe the relationship between two variables.

## Correlation Matrix

Table showing pairwise correlation coefficients among variables.

## What is Multicollinearity?

Occurs when predictors are highly correlated, inflating variance of coefficient estimates.

## ANOVA Partitioning

Breaks total variation into regression and residual parts: $SST = SSR + SSE$

## Diagnostic and Remedial Measures

* Residual plots
* VIF (Variance Inflation Factor)
* Transformation or dropping variables to reduce multicollinearity

## Indicator Variables

Binary (0/1) variables used to represent categorical predictors.

## Model Selection Criteria

* R²: Goodness of fit
* Adjusted R²: Fit adjusted for number of predictors
* Mallows’ Cp: Compares models with true number of predictors
* AIC/BIC: Penalized likelihood measures
* PRESS: Prediction error sum of squares for model validation

## Building a Multiple Linear Regression Model

1. Data Preprocessing (encoding, cleaning)
2. Feature Selection
3. Fit model (using `sklearn` or `statsmodels`)
4. Evaluate using R², RMSE, and diagnostics
5. Interpret coefficients and plots
