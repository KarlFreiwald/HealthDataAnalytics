################################################################################
#####               CLASSIFICATION OF THE CANCER DATASET                   #####
################################################################################

# import dataset of classification

library(ggplot2)
library(glmnet)
library(caret)
library(caTools)
library(pROC)
library(corrplot)
library(leaps)
library(stats)
library(graphics)

# If the programm has difficulties finding the relative path, use the entire path for help
# setwd("C:/Users/kfrei/OneDrive - Ostbayerische Technische Hochschule Regensburg/Desktop/HDA_project")

dataset <- read.csv("data/cancer.csv")
summary(dataset)

################################################################################
#####               Basic exploration of the dataset                       #####
################################################################################

sample_size = nrow(dataset)
print(sample_size)

nan_count <- sapply(dataset, function(x) sum(is.na(x)))
print("Couting the NaN value on features:\n")
print(nan_count)

# Dataset (for an error maybe) contains column "X" full of Nan values -> remove
dataset$X <- NULL

nan_count <- sapply(dataset, function(x) sum(is.na(x)))
print("Couting the NaN value on features after delete X feature:\n")
print(nan_count)

# At this point there are no more Nan values in the dataset.

# Visualizing the target variable distribution
print("Distribution of the target variable (Y):")
print(table(dataset$diagnosis))
quartz(title = "Distribution of Target Value") 
ggplot(dataset, aes(x = factor(diagnosis), fill = factor(diagnosis))) +
  geom_bar() +
  labs(title = "Distribution of the Target Variable", x = "Diagnosis", y = "Count")


################################################################################
#####                          Preprocessing                               #####
################################################################################

# Split the data in X and Y
y = dataset$diagnosis
id = dataset$id
x = subset(dataset, select = -c(diagnosis,id))

# The Y value is a categorical value ("M" and "B" values)
# the solution is to transform the "M" value in 1 e "B" value in "0" value
# We use the ifelse function to make a conditional substitution
y <- ifelse(y == "M", 1, ifelse(y == "B", 0, NA))
y <- factor(y, levels = c("0", "1"))
print(y)

################################################################################
#####                 Features Selection on Matrix Correlation             #####
################################################################################

# Create correlation matrix using corrplot library
quartz(title = "Correlation Matrix")    # use quartz to change the name of window
correlation_matrix <- cor(x)
corrplot(correlation_matrix, method = "circle", type = "upper", order = "hclust"
         ,tl.cex = 0.7, diag = FALSE)

correlation_threshold <- 0.9
correlation_mask <- abs(correlation_matrix) > correlation_threshold

# Initialize a list to collect correlated features for each feature
correlated_features <- vector("list", length = nrow(correlation_matrix))
names(correlated_features) <- paste("Feature", 1:nrow(correlation_matrix))

# Populate the list with indices of correlated features
for (i in 1:(nrow(correlation_matrix) - 1)) {
  for (j in (i + 1):ncol(correlation_matrix)) {
    if (correlation_mask[i, j]) {
      correlated_features[[i]] <- c(correlated_features[[i]], j)
      correlated_features[[j]] <- c(correlated_features[[j]], i)  # Include for bidirectional correlation
    }
  }
}

# Print correlated features
cat("Highly Correlated Features:\n")
for (i in seq_along(correlated_features)) {
  if (length(correlated_features[[i]]) > 0) {
    cat(sprintf("Feature %3d and Features %s\n", i, paste(correlated_features[[i]], collapse=", ")))
  }
}

# Remove highly correlated features
variables_to_keep <- 1:ncol(x)  # keep the default features
for (pair in correlated_pairs) {
  variables_to_keep <- variables_to_keep[variables_to_keep != pair[2]]  # remove the second features in the list
}

x_ncorr <- x[, variables_to_keep]
variables_to_keep

################################################################################
# Best Subset Selection
################################################################################

regfit.full <- regsubsets(y ~ ., data = x_ncorr)
print(summary(regfit.full))

# Subset selection with a specified maximum number of variables
regfit.full <- regsubsets(y ~ ., data = x_ncorr, nvmax = 15)
reg.summary <- summary(regfit.full)
print(names(reg.summary))

# Analyze various statistics
cat("\nLocation of RSS min:", which.min(reg.summary$rss), "\n")
cat("Location of adj-RSq max:", which.max(reg.summary$adjr2), "\n")
cat("Location of Cp min:", which.min(reg.summary$cp), "\n")
cat("Location of BIC min:", which.min(reg.summary$bic), "\n")

# Plot RSS, adjusted R2, Cp, and BIC for all models
par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
points(which.min(reg.summary$rss), min(reg.summary$rss), col = "red", cex = 2, pch = 20)

plot(reg.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
points(which.max(reg.summary$adjr2), max(reg.summary$adjr2), col = "red", cex = 2, pch = 20)

plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
points(which.min(reg.summary$cp), min(reg.summary$cp), col = "red", cex = 2, pch = 20)

plot(reg.summary$bic, xlab = "Number of Variables", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), min(reg.summary$bic), col = "red", cex = 2, pch = 20)

# Additional plots for BSS evaluations
plot(regfit.full, scale = "r2")
plot(regfit.full, scale = "adjr2")
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")

# Get coefficients of the model with the minimum BIC
coef(regfit.full, which.min(reg.summary$bic))

# Get the matrix of included/excluded features for each model
model_matrix <- summary(regfit.full)$which
print(model_matrix)

# Find non-included features for the model with minimum BIC
optimal_model_index <- which.min(reg.summary$bic); optimal_model_index
included_features <- names(model_matrix)[model_matrix[optimal_model_index,]]
non_included_features <- setdiff(names(x_ncorr), included_features)

# Print the features that are included in the optimal model
cat("Included Features in the Optimal Model (Min BIC):\n\t", 
    paste(included_features, collapse="\n\t"), "\n")

# Print the features that are not included in the optimal model
cat("Non-included Features in the Optimal Model (Min BIC):\n\t", 
    paste(non_included_features, collapse="\n\t"), "\n")

################################################################################
# Split in train and test
################################################################################

# We chose to put in training set the data that have more variance then data in test set

# data_variance <- apply(x_ncorr,1,var)
# 
# sorted_indices <- order(data_variance, decreasing = TRUE)
# sorted_indices
# 
# length(sorted_indices)
# train_proportion <- 0.8
# 
# num_train <- ceiling(train_proportion*length(sorted_indices))
# 
# train_indices <- sorted_indices[1:num_train]
# test_indices <- sorted_indices[(num_train + 1):length(sorted_indices)]
# 
# x_train <- x_ncorr[train_indices, ]
# y_train <- y[train_indices]
# x_test <- x_ncorr[test_indices, ]
# y_test <- y[test_indices]

set.seed(123)
split <- sample.split(x_ncorr, SplitRatio = 0.7)
x_train <- subset(x_ncorr, split == TRUE)
x_test <- subset(x_ncorr, split == FALSE)
y_train <- subset(y,split == TRUE)
y_test <- subset(y,split == FALSE)

################################################################################
# Normalization
################################################################################

z_score_normalization <- function(x) {
  return((x - mean(x)) / sd(x))
}

# Normalize train and test set separately
x_train_norm = as.data.frame(lapply(x_train,z_score_normalization))
x_test_norm = as.data.frame(lapply(x_test,z_score_normalization))

quartz(title = "Scatter Plot of distribution before normalization")
ggplot(x_train, aes(x = radius_mean, y =seq_len(nrow(x_train)))) +
  geom_point() +
  labs(x = "Radius Mean",
       y = "Samples") +
  theme_minimal()

quartz(title = "Scatter Plot of distribution after normalization")
ggplot(x_train_norm, aes(x = radius_mean, y =seq_len(nrow(x_train)))) +
  geom_point() +
  labs(x = "Radius Mean",
       y = "Samples") +
  theme_minimal()

################################################################################
# Logistic Regression
################################################################################

train_control_lr <- trainControl(method="cv", number = 5)

# Train the model using glmnet for Lasso (L1) or Ridge (L2)
log_reg <- train(
  y_train ~ ., 
  data = cbind(x_train_norm, y_train),
  method = "glmnet",
  trControl = train_control_lr,
  family = "binomial"
)

# Predict and ROC Curve
predict_lr = predict(log_reg, newdata=x_test_norm, type = "prob")[,2]
predict_lr_ROC <- roc(y_test, as.numeric(predict_lr))
predict_lr_ROC
quartz(title="ROC Curve Logistic Regression")
plot(predict_lr_ROC)
auc_value_lr <- auc(predict_lr_ROC)
text(x = 0.6, y = 0.2, labels = paste("AUC =", round(auc_value_lr, 3)), col = "red")

predict_lr <- factor(ifelse(predict_lr > 0.5,1,0), levels = c("0", "1"))
predict_lr
y_test

# Confusion Matrix
conf_matrix_lr <- confusionMatrix(y_test, predict_lr)

cm_lr <- as.table(conf_matrix_lr$table)

cm <- as.data.frame(cm_lr)
colnames(cm) <- c("Reference", "Prediction", "Freq")

quartz("Confusion Matrix")
ggplot(cm, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Classification Result", x = "Prediction", y = "True Values") +
  theme_minimal()

# Calculate the correct and wrong prediction
correct_predictions_lr <- sum(diag(cm_lr))
total_predictions <- sum(cm_lr)
incorrect_predictions_lr <- total_predictions - correct_predictions_lr
cat("Correct Predictions:", correct_predictions_lr, "\n")
cat("Incorrect Predictions:", incorrect_predictions_lr, "\n")

# Compute the F1 Score
TP_lr <- cm_lr[2, 2]
FP_lr <- cm_lr[1, 2]
FN_lr <- cm_lr[2, 1]
TN_lr <- cm_lr[1, 1]

precision_lr <- TP_lr / (TP_lr + FP_lr)
recall_lr <- TP_lr / (TP_lr + FN_lr)
f1_score_lr <- 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)
f1_score_lr

################################################################################
# Logistic Regression - Lasso
################################################################################

train_control_lasso_lr <- trainControl(method="cv", number = 5)

tune_grid_lasso_lr <- expand.grid(
  alpha = 1,  # 0 = Ridge, 1 = Lasso, 0.5 = Elastic Net
  lambda = seq(0.001, 0.1, by = 0.001)
)

# Train the model using glmnet for Lasso (L1) or Ridge (L2)
log_reg_lasso <- train(
  y_train ~ ., 
  data = cbind(x_train_norm, y_train),
  method = "glmnet",
  trControl = train_control_lasso_lr,
  tuneGrid = tune_grid_lasso_lr,
  family = "binomial"
)

# Predict and ROC Curve
predict_lr_lasso = predict(log_reg_lasso, newdata=x_test_norm, type = "prob")[,2]
predict_lr_lasso_ROC <- roc(y_test, as.numeric(predict_lr_lasso))
predict_lr_lasso_ROC
quartz(title="ROC Curve Logistic Regression")
plot(predict_lr_lasso_ROC)
auc_value_lr_lasso <- auc(predict_lr_lasso_ROC)
text(x = 0.6, y = 0.2, labels = paste("AUC =", round(auc_value_lr_lasso, 3)), col = "red")

# predict_lr <- ifelse(predict_lr >0.5,1,0)
predict_lr_lasso <- factor(ifelse(predict_lr_lasso >0.5,1,0), levels = c("0", "1"))
predict_lr_lasso
y_test

# Confusion Matrix
conf_matrix_lr_lasso <- confusionMatrix(y_test, predict_lr_lasso)

cm_lr_lasso <- as.table(conf_matrix_lr_lasso$table)

cm <- as.data.frame(cm_lr_lasso)
colnames(cm) <- c("Reference", "Prediction", "Freq")

quartz("Confusion Matrix")
ggplot(cm, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Classification Result", x = "Prediction", y = "True Values") +
  theme_minimal()

# Calculate the correct and wrong prediction
correct_predictions_lasso_lr <- sum(diag(cm_lr_lasso))
total_predictions <- sum(cm_lr_lasso)
incorrect_predictions_lasso_lr <- total_predictions - correct_predictions_lasso_lr
cat("Correct Predictions:", correct_predictions_lasso_lr, "\n")
cat("Incorrect Predictions:", incorrect_predictions_lasso_lr, "\n")

# Compute the F1 Score
TP_lasso_lr <- cm_lr_lasso[2, 2]
FP_lasso_lr <- cm_lr_lasso[1, 2]
FN_lasso_lr <- cm_lr_lasso[2, 1]
TN_lasso_lr <- cm_lr_lasso[1, 1]

precision_lasso_lr <- TP_lasso_lr / (TP_lasso_lr + FP_lasso_lr)
recall_lasso_lr <- TP_lasso_lr / (TP_lasso_lr + FN_lasso_lr)
f1_score_lasso_lr <- 2 * (precision_lasso_lr * recall_lasso_lr) / (precision_lasso_lr + recall_lasso_lr)
f1_score_lasso_lr

################################################################################
# Logistic Regression - Ridge
################################################################################

train_control_ridge_lr <- trainControl(method="cv", number = 5)

tune_grid_ridge_lr <- expand.grid(
  alpha = 0,  # 0 = Ridge, 1 = Lasso, 0.5 = Elastic Net
  lambda = seq(0.001, 0.1, by = 0.001)
)

# Train the model using glmnet for Lasso (L1) or Ridge (L2)
log_reg_ridge <- train(
  y_train ~ ., 
  data = cbind(x_train_norm, y_train),
  method = "glmnet",
  trControl = train_control_ridge_lr,
  tuneGrid = tune_grid_ridge_lr,
  family = "binomial"
)

# Predict and ROC Curve
predict_lr_ridge = predict(log_reg_ridge, newdata=x_test_norm, type = "prob")[,2]
predict_lr_ridge_ROC <- roc(y_test, as.numeric(predict_lr_ridge))
predict_lr_ridge_ROC
quartz(title="ROC Curve Logistic Regression")
plot(predict_lr_ridge_ROC)
auc_value_lr_ridge <- auc(predict_lr_ridge_ROC)
text(x = 0.6, y = 0.2, labels = paste("AUC =", round(auc_value_lr_ridge, 3)), col = "red")

# predict_lr <- ifelse(predict_lr >0.5,1,0)
predict_lr_ridge <- factor(ifelse(predict_lr_ridge >0.5,1,0), levels = c("0", "1"))
predict_lr_ridge
y_test

# Confusion Matrix
conf_matrix_lr_ridge <- confusionMatrix(y_test, predict_lr_ridge)

cm_lr_ridge <- as.table(conf_matrix_lr_ridge$table)

cm <- as.data.frame(cm_lr_ridge)
colnames(cm) <- c("Reference", "Prediction", "Freq")

quartz("Confusion Matrix")
ggplot(cm, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Classification Result", x = "Prediction", y = "True Values") +
  theme_minimal()

# Calculate the correct and wrong prediction
correct_predictions_ridge_lr <- sum(diag(cm_lr_ridge))
total_predictions <- sum(cm_lr_ridge)
incorrect_predictions_ridge_lr <- total_predictions - correct_predictions_ridge_lr
cat("Correct Predictions:", correct_predictions_ridge_lr, "\n")
cat("Incorrect Predictions:", incorrect_predictions_ridge_lr, "\n")

# Compute the F1 Score
TP_ridge_lr <- cm_lr_ridge[2, 2]
FP_ridge_lr <- cm_lr_ridge[1, 2]
FN_ridge_lr <- cm_lr_ridge[2, 1]
TN_ridge_lr <- cm_lr_ridge[1, 1]

precision_ridge_lr <- TP_ridge_lr / (TP_ridge_lr + FP_ridge_lr)
recall_ridge_lr <- TP_ridge_lr / (TP_ridge_lr + FN_ridge_lr)
f1_score_ridge_lr <- 2 * (precision_ridge_lr * recall_ridge_lr) / (precision_ridge_lr + recall_ridge_lr)
f1_score_ridge_lr


################################################################################
# Logistic Regression - ElasticNet
################################################################################

train_control_elasticnet_lr <- trainControl(method="cv", number = 5)

tune_grid_elasticnet_lr <- expand.grid(
  alpha = 0.5,  # 0 = elasticnet, 1 = Lasso, 0.5 = Elastic Net
  lambda = seq(0.001, 0.1, by = 0.001)
)

# Train the model using glmnet for Lasso (L1) or elasticnet (L2)
log_reg_elasticnet <- train(
  y_train ~ ., 
  data = cbind(x_train_norm, y_train),
  method = "glmnet",
  trControl = train_control_elasticnet_lr,
  tuneGrid = tune_grid_elasticnet_lr,
  family = "binomial"
)

# Predict and ROC Curve
predict_lr_elasticnet = predict(log_reg_elasticnet, newdata=x_test_norm, type = "prob")[,2]
predict_lr_elasticnet_ROC <- roc(y_test, as.numeric(predict_lr_elasticnet))
predict_lr_elasticnet_ROC
quartz(title="ROC Curve Logistic Regression")
plot(predict_lr_elasticnet_ROC)
auc_value_lr_elasticnet <- auc(predict_lr_elasticnet_ROC)
text(x = 0.6, y = 0.2, labels = paste("AUC =", round(auc_value_lr_elasticnet, 3)), col = "red")

# predict_lr <- ifelse(predict_lr >0.5,1,0)
predict_lr_elasticnet <- factor(ifelse(predict_lr_elasticnet >0.5,1,0), levels = c("0", "1"))
predict_lr_elasticnet
y_test

# Confusion Matrix
conf_matrix_lr_elasticnet <- confusionMatrix(y_test, predict_lr_elasticnet)

cm_lr_elasticnet <- as.table(conf_matrix_lr_elasticnet$table)

cm <- as.data.frame(cm_lr_elasticnet)
colnames(cm) <- c("Reference", "Prediction", "Freq")

quartz("Confusion Matrix")
ggplot(cm, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Classification Result", x = "Prediction", y = "True Values") +
  theme_minimal()

# Calculate the correct and wrong prediction
correct_predictions_elasticnet_lr <- sum(diag(cm_lr_elasticnet))
total_predictions <- sum(cm_lr_elasticnet)
incorrect_predictions_elasticnet_lr <- total_predictions - correct_predictions_elasticnet_lr
cat("Correct Predictions:", correct_predictions_elasticnet_lr, "\n")
cat("Incorrect Predictions:", incorrect_predictions_elasticnet_lr, "\n")

# Compute the F1 Score
TP_elasticnet_lr <- cm_lr_elasticnet[2, 2]
FP_elasticnet_lr <- cm_lr_elasticnet[1, 2]
FN_elasticnet_lr <- cm_lr_elasticnet[2, 1]
TN_elasticnet_lr <- cm_lr_elasticnet[1, 1]

precision_elasticnet_lr <- TP_elasticnet_lr / (TP_elasticnet_lr + FP_elasticnet_lr)
recall_elasticnet_lr <- TP_elasticnet_lr / (TP_elasticnet_lr + FN_elasticnet_lr)
f1_score_elasticnet_lr <- 2 * (precision_elasticnet_lr * recall_elasticnet_lr) / (precision_elasticnet_lr + recall_elasticnet_lr)
f1_score_elasticnet_lr


################################################################################
#####                       K-Nearest Neighbors                            #####
################################################################################

train_control_knn <- trainControl(method="cv", number = 10)
tune_grid_knn <- expand.grid(k = 10)

knn_model <- train(
  y_train ~ ., 
  data = cbind(x_train_norm, y_train),
  method = "knn",
  trControl = train_control_knn,
  tuneGrid = tune_grid_knn
)

# Predict and ROC Curve
predict_knn = predict(knn_model, newdata=x_test_norm, type = "prob")[, 2]
predict_knn_ROC <- roc(y_test,as.numeric(predict_knn))
predict_knn_ROC
quartz(title="ROC Curve KNN")
plot(predict_knn_ROC)
auc_value_knn <- auc(predict_knn_ROC)
text(x = 0.6, y = 0.2, labels = paste("AUC =", round(auc_value_knn, 3)), col = "red")

predict_knn <- factor(ifelse(predict_knn > 0.5, 1, 0), levels = c("0", "1"))

# Confusion Matrix
conf_matrix_knn <- confusionMatrix(y_test, predict_knn)

cm_knn <- as.table(conf_matrix_knn$table)

cm_knn_df <- as.data.frame(cm_knn)
colnames(cm_knn_df) <- c("Reference", "Prediction", "Freq")

quartz("Confusion Matrix")
ggplot(cm_knn_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Classification Result", x = "Prediction", y = "True Values") +
  theme_minimal()

# Calcolare le previsioni corrette e errate
correct_predictions <- sum(diag(cm_knn))
total_predictions <- sum(cm_knn)
incorrect_predictions <- total_predictions - correct_predictions

# Visualizzare i risultati
cat("Correct Predictions:", correct_predictions, "\n")
cat("Incorrect Predictions:", incorrect_predictions, "\n")

# calcolare F1_score
TP <- cm_knn[2, 2]
FP <- cm_knn[1, 2]
FN <- cm_knn[2, 1]
TN <- cm_knn[1, 1]

precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score_knn <- 2 * (precision * recall) / (precision + recall)
f1_score_knn
