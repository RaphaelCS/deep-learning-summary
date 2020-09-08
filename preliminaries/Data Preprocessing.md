# Data Preprocessing

```python
all_features = pd.concat((train_data, test_data))
```



## Numerical Features

### Standardization

Rescaling features to zero mean and unit variance
$$
x \leftarrow \frac{x - \mu}{\sigma}
$$

<u>Reasons</u>:

1. It proves convenient for optimization.
2. Because we do not know *a priori* which features will be relevant, we do not want to penalize coefficients assigned to one feature more than on any other.

```python
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std())
)
```

<u>Usages</u>:

We use it when

1. Parameter-based models or distance-based models are all about feature normalization.

We do not use it when

1. The tree-based method does not require the normalization of features, such as random forest, bagging and boosting.



**Missing Values**: Replacing all missing values by the corresponding feature's mean

```python
# After standardizing the data all means vanish, hence we can set missing
# Values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```



  

## Discrete Values

### One-hot encoding

The value of the discrete feature is extended to the Euclidean space, and a value of the discrete feature corresponds to a point in the Euclidean space.

<u>Reasons</u>:

1. Make the calculation of **distance between features more reasonable**

   > Note: Most algorithms are based on vector space metrics.

2. Make the values of the variables without partial order relationship have no partial order 

3. Make them equidistant from the origin 

<u>Advantages</u>:

1. Solve the problem that the classifier is not good at processing attribute data
2. Expand features
3. Its value is only 0 and 1, and different types are stored in the vertical space 

<u>Disadvantages</u>:

1. When the number of categories is large, the feature space becomes very large

   > Note: In this case, PCA can generally be used to reduce the dimensionality. And the combination of one hot encoding+PCA is also very useful in practice.

<u>Usages</u>:

We use it when

1. The number of categories is not too large

We do not use it when

1. Some tree-based algorithms are not based on vector space metrics, and the values are just category symbols. That is, there is no partial order relation.

   > Note: For Decision Trees, the essence of one-hot is to increase the depth of the tree.

```python
# 'Dummy_na=True' considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
```





# Coding Skills

1. Vectorization for Speed