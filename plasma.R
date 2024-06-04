
# Load necessary libraries
library(tidyverse)
library(caret)

# Load the data
file_path <- "C:/Users/kfrei/OneDrive - Ostbayerische Technische Hochschule Regensburg/Desktop/HDA_project/data/plasma.csv"
plasma_data <- read.csv(file_path)

################################################################################
# Basic exploration of the dataset
################################################################################

# Basic information
print(dim(plasma_data))
print(head(plasma_data))
print(str(plasma_data))
print(summary(plasma_data))

num_samples <- nrow(plasma_data)
num_components <- ncol(plasma_data)

# Check for missing values
print(colSums(is.na(plasma_data)))

################################################################################
# Preprocessing
################################################################################

# Remove the index column 'X'
plasma_data <- plasma_data %>% 
  select(-X)

# Convert categorical variables to factors
plasma_data$sex <- as.factor(plasma_data$sex)
plasma_data$smokstat <- as.factor(plasma_data$smokstat)

# Convert factors to dummy variables
plasma_data <- plasma_data %>%
  mutate(across(c(sex, smokstat), as.factor)) %>%
  mutate(dummy_sex              = if_else(sex      == "female",         1, 0),
         dummy_smokstat_never   = if_else(smokstat == "Never",          1, 0),
         dummy_smokstat_former  = if_else(smokstat == "Former",         1, 0),
         dummy_smokstat_current = if_else(smokstat == "Current Smoker", 1, 0))

# Remove original categorical columns
plasma_data <- plasma_data %>%
  select(-c(sex, smokstat))  

# Setting up X and Y variable
response_var <- "retplasma"
predictors <- names(plasma_data)[!names(plasma_data) %in% response_var]

# Filter out non-numeric predictors before normalization
numeric_predictors <- predictors[sapply(plasma_data[predictors], is.numeric)]

# Normalize numeric predictors to the interval [-1, +1]
plasma_data[numeric_predictors] <- map_df(plasma_data[numeric_predictors], ~ {
  2 * (. - min(.)) / (max(.) - min(.)) - 1
})

################################################################################
# Visualize the relationship between dietary factors and plasma levels
################################################################################

library(ggplot2)

ggplot(plasma_data, aes(x = betadiet, y = betaplasma)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Relationship between Dietary Beta-Carotene and Plasma Beta-Carotene", 
       x = "Dietary Beta-Carotene", 
       y = "Plasma Beta-Carotene")

ggplot(plasma_data, aes(x = retdiet, y = retplasma)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(title = "Relationship between Dietary Retinol and Plasma Retinol", 
       x = "Dietary Retinol", 
       y = "Plasma Retinol")

################################################################################
# Model training and testing
################################################################################

# Define the training control using cross-validation
train_control <- trainControl(
  method = "cv",            # k-fold cross-validation
  number = 5,               # number of folds
  savePredictions = "final",
  verboseIter = TRUE
)

# Define the training control using bootstrap
train_control <- trainControl(
  method = "boot",          # Using bootstrap
  number = 10,              # Number of bootstrap resamples
  savePredictions = "final",
  verboseIter = TRUE
)

# Train the model using linear regression
model <- train(
  reformulate(predictors, response=response_var), 
  data = plasma_data,
  method = "glm",
  trControl = train_control
)

# Extracting and preparing results
results <- model$resample
results$Resample <- rownames(results)

# Plotting MSE for each fold
mse_plot <- ggplot(results, aes(x = Resample, y = RMSE)) +
  geom_line(aes(group=1), color = "blue") +
  geom_point() +
  labs(title = "MSE Across Folds", x = "Fold", y = "MSE")

print(mse_plot)
