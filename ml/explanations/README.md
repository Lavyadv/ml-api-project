# Model explanations

## Plain-language interpretation
### Sample 0: predicted **setosa** (100.0%)
LIME's strongest local signals were `petal_width <= 0.30` and `petal_length <= 1.60`.
In plain business language: for this individual flower, those measurements are the main reasons the model favours **setosa**. This explanation describes this one prediction, not a universal rule for every flower.

### Sample 77: predicted **virginica** (77.0%)
LIME's strongest local signals were `1.30 < petal_width <= 1.80` and `sepal_length > 6.40`.
In plain business language: for this individual flower, those measurements are the main reasons the model favours **virginica**. This explanation describes this one prediction, not a universal rule for every flower.
