
################################################################################
# Imports
################################################################################

library(ggplot2)
library(caret)
library(caTools)
library(pROC)
library(corrplot)
library(fastDummies)
library(plotly)
library(class)
library(glmnet)

# Load the data
file_path <- "C:/Users/kfrei/OneDrive - Ostbayerische Technische Hochschule Regensburg/Desktop/HDA_project/data/leukemia.csv"
data <- read.csv(file_path)

################################################################################
# Basic Exploration of the Dataset
################################################################################

num_samples <- nrow(data); num_samples
num_features <- ncol(data); num_features

# names(data)
# head(data)

nan_count <- sum(sapply(data, function(x) sum(is.na(x))))
cat("Couting the NaN value on features:\n")
print(nan_count)

print("Distribution of the target variable (Y):")
print(table(data$Y))

ggplot(data, aes(x = factor(Y), fill = factor(Y))) +
  geom_bar() +
  labs(title = "Distribution of the Target Variable", x = "Diagnosis", y = "Count")

y = data$Y
x = data[,-c(1)]

set.seed(42)
train_index <- createDataPartition(y, p = 0.7, list = FALSE)
x_train <- x[train_index, ]
x_test <- x[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index] 

# Standardize the data
x_train <- data.frame(scale(x_train))
x_test <- data.frame(scale(x_test))

# Create DataFrame
df_train <- as.data.frame(x_train)
df_train$Y <- as.factor(y_train)
df_train$Y

df_test <- as.data.frame(x_test)
df_test$Y <- as.factor(y_test)
df_test$Y

################################################################################
# PCA
################################################################################

pca_train <- prcomp(x_train)
percentage_of_explained_variance <- 0.90

# Create a data frame for plotting
variance_df <- data.frame(
  PC = 1:length(pca_train$sdev),
  CumulativeVariance = cumsum(pca_train$sdev^2) / sum(pca_train$sdev^2)
)

# Plotting the cumulative variance
ggplot(variance_df, aes(x = PC, y = CumulativeVariance)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:length(pca_train$sdev)) +
  labs(title = "Cumulative Variance Explained by PCA Components",
       x = "Principal Component",
       y = "Cumulative Variance Explained") +
  geom_hline(yintercept = percentage_of_explained_variance, 
             linetype = "dashed", 
             color = "red") +
  theme_minimal()

# Capture percentage of variance
explained_variance <- cumsum(pca_train$sdev^2) / sum(pca_train$sdev^2)
num_components <- which.min(abs(explained_variance - percentage_of_explained_variance))
num_components

# Fixed number of PCs
# num_components <- 3

# Select the required number of principal components
x_pca_train <- pca_train$x[, 1:num_components]

# Apply PC-transformation learned from training data to test data
x_pca_test <- predict(pca_train, newdata = x_test)
x_pca_test <- x_pca_test[, 1:num_components]

# Combine PCA result with the target variable
df_pca_train <- as.data.frame(x_pca_train)
df_pca_train$Y <- as.factor(y_train)
df_pca_train$Y

df_pca_test <- as.data.frame(x_pca_test)
df_pca_test$Y <- as.factor(y_test)
df_pca_test$Y

################################################################################
# SELECT DATA (Raw or PCA)
################################################################################

df_train = df_pca_train
df_test = df_pca_test

# df_train = df_train
# df_test = df_test

################################################################################
# KNN
################################################################################

# Define a range of k values to test
k_values <- 1:7
accuracy_scores <- numeric(length(k_values))

# Find the best k value
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  test_pred <- knn(train = x_train, test = x_test, cl = y_train, k = k)
  cm <- table(y_test, test_pred)
  accuracy_scores[i] <- sum(diag(cm)) / sum(cm)
}

best_k <- k_values[which.max(accuracy_scores)]
cat("Best k:", best_k, "\n")
cat("Best accuracy:", max(accuracy_scores), "\n")
plot(k_values, accuracy_scores, type = "b", pch = 19, col = "blue", xlab = "Number of Neighbors k", ylab = "Accuracy")

# Train the model with the best k
test_pred <- knn(train = x_train, test = x_test, cl = y_train, k = best_k)

cm <- table(y_test, test_pred)
cat("Confusion Matrix for the best k:\n")
print(cm)

true_positives <- final_cm[2, 2]
false_positives <- final_cm[1, 2]
false_negatives <- final_cm[2, 1]
true_negatives <- final_cm[1, 1]

accuracy <- sum(diag(cm)) / sum(cm)
precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy:", accuracy, "\n")
cat("F1 Score:", f1_score, "\n")

################################################################################
# Logistic Regression
################################################################################

# Prepare matrix of predictors and response variable vector
x_matrix <- model.matrix(Y ~ . - 1, data = df_train) # -1 to exclude intercept
y_vector <- df_train$Y

# Fit regularized logistic regression model using glmnet
# alpha = 1 for lasso, alpha = 0 for ridge, values in between for elastic net
glmnet_model <- glmnet(x_matrix, as.numeric(y_vector) - 1, family = "binomial", alpha = 1)

# Optionally, use cross-validation to find the best lambda
cv_model <- cv.glmnet(x_matrix, as.numeric(y_vector) - 1, family = "binomial", type.measure = "class")
plot(cv_model)

# Best lambda value
best_lambda <- cv_model$lambda.min
cat("Best Lambda:", best_lambda, "\n")

# Predict using the model with the best lambda
predicted_probabilities <- predict(glmnet_model, s = best_lambda, newx = model.matrix(Y ~ . - 1, data = df_test), type = "response")
predicted_classes <- ifelse(predicted_probabilities > 0.5, 1, 0)

# Evaluate performance
conf_matrix <- table(Predicted = predicted_classes, Actual = df_test$Y)
print(conf_matrix)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy with best lambda:", accuracy, "\n")

################################################################################
# Model training and testing
################################################################################

# Train a logistic regression classifier with cross-validation
# control <- trainControl(method = "cv", number = 10, savePredictions = "final")
# model <- train(Y ~ ., data = train_data, method = "glm", trControl = control, family = "binomial")
lr_pca_models <- list()
alphas = seq(0.0, 1.0, by = 0.1)

for (alpha in alphas){
  lr_pca_model = cv.glmnet(X_pca, df_pca$Y, family="binomial", type.measure = "auc", alpha = alpha)
  lr_pca_models[[as.character(alpha)]] <- lr_pca_model
}

best_alpha_lasso <- alphas[which.max(sapply(lr_lasso_models, function(m) m$cvm[which.max(m$cvm)]))]
best_alpha_lasso

lr_lasso_model = lr_lasso_models[[as.character(best_alpha_lasso)]]
lr_lasso_model
coef(lr_lasso_model)

predict_lr_lasso = predict(lr_lasso_model, newx = x_test_lasso, type="response")
predict_lr_lasso

# Train a model with Lasso regularization
model <- train(Y ~ ., data = train_data, method = "glmnet", 
                     trControl = control, family = "binomial",
                     tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, length = 10)))

# Predict on the test set
y_pred <- predict(model, newdata = test_data)

# Evaluate the model's accuracy
accuracy <- sum(y_pred == test_data$Y) / nrow(test_data)
print(paste('Accuracy:', round(accuracy, 2)))

# Evaluate with additional metrics
confusionMatrix <- confusionMatrix(predict(model, newdata = test_data), test_data$Y)
print(confusionMatrix)

