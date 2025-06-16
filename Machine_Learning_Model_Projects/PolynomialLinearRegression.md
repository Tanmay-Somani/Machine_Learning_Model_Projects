# Polynomial Regression & Data Preprocessing Interview Prep

## Learn to Build a Polynomial Regression Model from Scratch

* Extend linear regression by including polynomial terms: $Y = \beta_0 + \beta_1 X + \beta_2 X^2 + ... + \beta_n X^n + \epsilon$
* Use `PolynomialFeatures` in `sklearn` to create polynomial terms.

## What is a Distribution Plot?

* A visual representation of the distribution of data (e.g., histogram + KDE).
* Shows skewness, modality, and spread.

## What is a Boxplot?

* Displays the median, quartiles, and potential outliers of data.
* Useful for spotting asymmetry and extreme values.

## What is a Violin Plot?

* Combines boxplot and KDE (density) to show distribution and summary stats.
* Offers more insight into distribution shape.

## How to Detect Outliers?

* IQR Rule: Values outside $Q1 - 1.5*IQR, Q3 + 1.5*IQR$
* Z-score method: $|z| > 3$
* Visual: boxplots, scatterplots

## How to Treat Outliers?

* Removal
* Capping/winsorizing
* Transformation (log, sqrt)
* Imputation

## What is Pandas Imputer?

* Earlier used via `SimpleImputer`; `pandas` itself doesn't have a built-in imputer.
* Fill missing data with `.fillna()` or integrate with `sklearn` imputers.

## What is Iterative Imputer?

* Uses regression models iteratively to impute missing values.
* More accurate than mean/median imputers.
* Available in `sklearn.experimental`.

## What is a KNN Imputer?

* Imputes missing values using mean (or weighted average) of nearest neighbors.
* Good when similar samples have similar missing values.

## What is an LGBM Imputer?

* LightGBM used to model and impute missing data.
* Often custom implemented using regression over missing fields.

## What is Univariate Analysis?

* Analysis of a single variable using stats and plots (histograms, boxplots, etc).
* Used for understanding distribution, central tendency, and spread.

## What is Chatterjee Correlation?

* Measures correlation between an independent variable and the fitted values.
* Values close to 1 imply multicollinearity.
* Proposed to detect variable redundancy in regression.

## What is ANOVA?

* **Analysis of Variance**: Tests if means of multiple groups differ.
* Based on partitioning variance into between-group and within-group.

## Implementation of ANOVA

* Use `statsmodels`:

```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
model = ols('Y ~ C(group)', data=df).fit()
anova_results = anova_lm(model)
```

## What is Data Preprocessing?

* Steps to clean and prepare data:

  * Handling missing values
  * Encoding categorical features
  * Normalization/scaling
  * Outlier treatment

## What is AIC?

* **Akaike Information Criterion**: Model selection metric
* Lower AIC → better model (penalizes model complexity)
* $\text{AIC} = 2k - 2\ln(L)$, where k = #parameters, L = likelihood

## What is Likelihood?

* Probability of data given parameters
* Higher likelihood → better model fit
* Used in MLE (Maximum Likelihood Estimation)
