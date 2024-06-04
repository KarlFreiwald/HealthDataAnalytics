
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
# Hyperparameters
################################################################################

cv_folds = 10
prob_threshold <- 0.5

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

# Select if you want to proceed with the raw data or with the PCA-reduced data
df_train = df_pca_train; df_test = df_pca_test
# df_train = df_train;     df_test = df_test

# Further Processing of the selected data
x_train <- as.matrix(df_train[, -ncol(df_train)])
y_train <- as.factor(df_train$Y)

x_test <- as.matrix(df_test[, -ncol(df_test)])
y_test <- as.factor(df_test$Y)

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

################################################################################
# Logistic Regression
################################################################################

alpha_values <- seq(0.0, 1.0, by = 0.1)
results <- list()

for (alpha in alpha_values) {
  cv_fit <- cv.glmnet(
    x_train, y_train, 
    family = "binomial", 
    type.measure = "class",
    alpha = alpha, 
    nfolds = cv_folds
    )
  results[[paste("Alpha", alpha)]] <- list(alpha = alpha, cv_fit = cv_fit, lambda_min = cv_fit$lambda.min)
}

# Find the alpha with the lowest CV error
best_result <- sapply(results, function(x) min(x$cv_fit$cvm))
best_alpha <- alpha_values[which.min(best_result)]

cat("Best alpha:", best_alpha, "\n")
# Re-run the model with the best alpha
best_cv_fit <- results[[paste("Alpha", best_alpha)]]$cv_fit
best_lambda <- best_cv_fit$lambda.min

plot(best_cv_fit)

# Fit the final model with the best alpha and lambda
final_model <- glmnet(
  x_train, y_train, 
  family = "binomial", 
  alpha = best_alpha, 
  lambda = best_lambda
  )

# Make predictions
test_pred_prob <- predict(final_model, newx = x_test, type = "response")
test_pred <- ifelse(pred_prob > prob_threshold, 1, 0)

################################################################################
# Model Evaluation
################################################################################

conf_matrix <- table(Predicted=y_test, Actual=test_pred)
print(conf_matrix)

true_positives <- conf_matrix[2, 2]
false_positives <- conf_matrix[1, 2]
false_negatives <- conf_matrix[2, 1]
true_negatives <- conf_matrix[1, 1]

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
precision <- true_positives / (true_positives + false_positives)
recall <- true_positives / (true_positives + false_negatives)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("Accuracy:", accuracy, "\n")
cat("F1 Score:", f1_score, "\n")
