# Linear Regression Interview Questions & Answers

---

### 1. What is Regression?

**Answer:** Regression is a statistical technique to model the relationship between a dependent variable and one or more independent variables.

---

### 2. Types of Regression

**Answer:** Simple Linear, Multiple Linear, Polynomial, Ridge, Lasso, Logistic (for classification).

---

### 3. What is Mean, Variance, and Standard Deviation?

**Answer:**

* **Mean:** Average of data
* **Variance:** Measure of data spread around the mean
* **Standard Deviation:** Square root of variance

---

### 4. Correlation and Causation

**Answer:** Correlation shows relationship; causation indicates that one variable directly affects another.

---

### 5. What are Observational and Experimental data?

**Answer:**

* **Observational:** No manipulation, data is observed
* **Experimental:** Controlled experiments with treatments applied

---

### 6. Formula for Regression

**Answer:** $y = \beta_0 + \beta_1 x + \varepsilon$

---

### 7. Building a Simple Linear Regression model

**Answer:** Use libraries like `scikit-learn` to fit the model, then predict and visualize results.

---

### 8. Understanding Interpolation and Extrapolation

**Answer:**

* **Interpolation:** Predicting within data range
* **Extrapolation:** Predicting beyond data range (less reliable)

---

### 9. What are Lurking Variables?

**Answer:** Hidden variables that affect both independent and dependent variables, possibly misleading correlation.

---

### 10. Derivation for Least Square Estimates

**Answer:** Minimize $\sum (y_i - \hat{y}_i)^2$ to derive $\hat{\beta}_0$ and $\hat{\beta}_1$

---

### 11. The Gauss Markov Theorem

**Answer:** States that under certain conditions, OLS estimators are the Best Linear Unbiased Estimators (BLUE).

---

### 12. Point estimators of Regression

**Answer:** Estimated coefficients $\hat{\beta}_0, \hat{\beta}_1$ from sample data.

---

### 13. Sampling distributions of Regression coefficients

**Answer:** Coefficients follow a sampling distribution used to construct confidence intervals and perform hypothesis testing.

---

### 14. F-Statistics

**Answer:** Tests whether the overall regression model is significant. Large F indicates at least one predictor is useful.

---

### 15. ANOVA Partitioning

**Answer:** Total variation (SST) is split into explained (SSR) and unexplained (SSE) components.

---

### 16. Coefficient of Determination (R-Squared)

**Answer:** Proportion of variance in the dependent variable explained by the model.

---

### 17. Diagnostic and Remedial Measures

**Answer:**

* **Check residuals** for normality, linearity, homoscedasticity
* **Use plots, VIF, Cook's distance** to identify issues
* **Apply transformations or different models** as remedies
