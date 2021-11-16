### Assignment 2 ###
# Initializing packages
library(pls)
library(ggplot2)
library(glmnet)
library(e1071)
library(kernlab)
library(hydroGOF)

## Importing the original data set
original_df <- read.csv("HW2.csv", header = TRUE)

# Defined to calculate the R-Squared values for PCR and PLS
train_y_original <- original_df[1:80, c("Property")]
test_y_original <- original_df[81:nrow(original_df), c("Property")]

# Creating a blank data frame to store R-Squared values of all techniques used
r_squared_values <- data.frame(matrix(ncol=3,nrow=0, dimnames=list(NULL, c("Technique_Used", "training_R_squared", "test_R_squared"))))

################################################ Linear Regression  ############################################################
# Splitting the data into train and test data
mlr_train <- original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
mlr_test <- original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]

# Training MLR Model with training data
mlr_model <- lm(train_y_original~., data = mlr_train)

# Using model to make prediction on the test set
test_mlr_pred <- predict(mlr_model, mlr_test, ncomp=2)

# Calculating R-Squared Value for test data
# Test data
test_sumofsquares_total_mlr <- sum((test_y_original - mean(test_y_original))^2)
test_sumofsquares_residuals_mlr <- sum((test_y_original - test_mlr_pred)^2)
test_mlr_r2 <- 1 - test_sumofsquares_residuals_mlr / test_sumofsquares_total_mlr

# Appending the values of MLR R-Squared to Data Frame
r_squared_values[nrow(r_squared_values) + 1,] = c('Multiple Linear Regression', summary(mlr_model)$r.squared, test_mlr_r2)

############################################## Principal Component Regression #################################################################################### 
# Generating random seeds for better prediction
set.seed(50)

# Splitting the data into train and test data
pcr_train <- original_df[1:80, c("Property", "Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
pcr_x_train <- original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
pcr_y_test <- original_df[81:nrow(original_df), c("Property")]
pcr_x_test <- original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]

# Creating the model
pcr_model <- pcr(Property~Feature1+Feature2+Feature3+Feature4+Feature5+Feature6+Feature7+Feature8, data=pcr_train, scale=TRUE, validation="CV")

# Using model to make prediction on the train and test set
train_pcr_pred <- predict(pcr_model, pcr_x_train, ncomp=2)
test_pcr_pred <- predict(pcr_model, pcr_x_test, ncomp=2)

# Calculating R-Squared Value for training and test data
# Train data
train_sumofsquares_total_pcr <- sum((train_y_original - mean(train_y_original))^2)
train_sumofsquares_residuals_pcr <- sum((train_y_original - train_pcr_pred)^2)
train_pcr_r2 <- 1 - train_sumofsquares_residuals_pcr / train_sumofsquares_total_pcr
# Test data
test_sumofsquares_total_pcr <- sum((test_y_original - mean(test_y_original))^2)
test_sumofsquares_residuals_pcr <- sum((test_y_original - test_pcr_pred)^2)
test_pcr_r2 <- 1 - test_sumofsquares_residuals_pcr / test_sumofsquares_total_pcr

# Appending the values of PCR R-Squared to Data Frame
r_squared_values[nrow(r_squared_values) + 1,] = c('Principal Component Regression', train_pcr_r2, test_pcr_r2)

############################################### Partial Least Square  ##############################################################
# Generating random seeds for better prediction
set.seed(50)

# Splitting the data into train and test data
plsr_train <- original_df[1:80, c("Property", "Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
plsr_x_train <- original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
plsr_y_test <- original_df[81:nrow(original_df), c("Property")]
plsr_x_test <- original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]

# Creating the model
plsr_model <- plsr(Property~Feature1+Feature2+Feature3+Feature4+Feature5+Feature6+Feature7+Feature8, data=original_df, scale=TRUE, validation="CV")

# Using model to make prediction on the train and test set
train_plsr_pred <- predict(plsr_model, plsr_x_train, ncomp=2)
test_plsr_pred <- predict(plsr_model, plsr_x_test, ncomp=2)

# Calculating R-Squared Value for training and test data
# Train data
train_sumofsquares_total_plsr <- sum((train_y_original - mean(train_y_original))^2)
train_sumofsquares_residuals_plsr <- sum((train_y_original - train_plsr_pred)^2)
train_plsr_r2 <- 1 - train_sumofsquares_residuals_plsr / train_sumofsquares_total_plsr
# Test data
test_sumofsquares_total_plsr <- sum((test_y_original - mean(test_y_original))^2)
test_sumofsquares_residuals_plsr <- sum((test_y_original - test_plsr_pred)^2)
test_plsr_r2 <- 1 - test_sumofsquares_residuals_plsr / test_sumofsquares_total_plsr

# Appending the values of PLS R-Squared to Data Frame
r_squared_values[nrow(r_squared_values) + 1,] = c('Partial Least Square', train_plsr_r2, test_plsr_r2)

############################################### Lasso Regression ##############################################################
# The 'glmnet' package being used here requires the predictor variable to be in a matrix
# Defining matrix of predictor variables
ridge_x_train <- data.matrix(original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")])
ridge_x_test <- data.matrix(original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")])

# Fit Ridge Regression Model
ridge_model <- glmnet(ridge_x_train, train_y_original, alpha = 0)

# Perform k-fold cross-validation to find optimal lambda value
cv_ridge_model <- cv.glmnet(ridge_x_train, train_y_original, alpha = 0)

best_lambda_ridge <- cv_ridge_model$lambda.min

# Using the best lambda value to train the model again
best_ridge_model <- glmnet(ridge_x_train, train_y_original, alpha = 0, lambda = best_lambda_ridge)

# Using model to make prediction on the train and test set
train_y_predicted_ridge <- predict(ridge_model, s = best_lambda_ridge, newx = ridge_x_train)
test_y_predicted_ridge <- predict(ridge_model, s = best_lambda_ridge, newx = ridge_x_test)

# Calculating R-Squared Value for training and test data
#Train
train_ridge_sst <- sum((train_y_original - mean(train_y_original))^2)
train_ridge_sse <- sum((train_y_predicted_ridge - train_y_original)^2)
#Test
test_ridge_sst <- sum((test_y_original - mean(test_y_original))^2)
test_ridge_sse <- sum((test_y_predicted_ridge - test_y_original)^2)

train_ridge_rsq <- 1 - train_ridge_sse/train_ridge_sst
test_ridge_rsq  <- 1 - test_ridge_sse/test_ridge_sst

# Appending the values of PCR R-Squared to Data Frame
r_squared_values[nrow(r_squared_values) + 1,] = c('Ridge Regression', train_ridge_rsq, test_ridge_rsq)

############################################### Lasso Regression ##############################################################
# The 'glmnet' package being used here requires the predictor variable to be in a matrix
# Defining matrix of predictor variables
lasso_x_train <- data.matrix(original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")])
lasso_x_test <- data.matrix(original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")])

# Fit Lasso Regression Model
lasso_model <- glmnet(lasso_x_train, train_y_original, alpha = 1)

# Perform k-fold cross-validation to find optimal lambda value
cv_lasso_model <- cv.glmnet(lasso_x_train, train_y_original, alpha = 1)

best_lambda_lasso <- cv_lasso_model$lambda.min

# Using the best lambda value to train the model again
best_lasso_model <- glmnet(lasso_x_train, train_y_original, alpha = 1, lambda = best_lambda_lasso)

# Using model to make prediction on the train and test set
train_y_predicted_lasso <- predict(lasso_model, s = best_lambda_lasso, newx = lasso_x_train)
test_y_predicted_lasso <- predict(lasso_model, s = best_lambda_lasso, newx = lasso_x_test)

# Calculating R-Squared Value for training and test data
#Train
train_lasso_sst <- sum((train_y_original - mean(train_y_original))^2)
train_lasso_sse <- sum((train_y_predicted_lasso - train_y_original)^2)
#Test
test_lasso_sst <- sum((test_y_original - mean(test_y_original))^2)
test_lasso_sse <- sum((test_y_predicted_lasso - test_y_original)^2)

train_lasso_rsq <- 1 - train_lasso_sse/train_lasso_sst
test_lasso_rsq  <- 1 - test_lasso_sse/test_lasso_sst

# Appending the values of PCR R-Squared to Data Frame
r_squared_values[nrow(r_squared_values) + 1,] = c('Lasso Regression', train_lasso_rsq, test_lasso_rsq)

############################################### Support Vector Regression ##############################################################
svr_x_train <- original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
svr_x_test <- original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]

# Fit SVR Model
svr_model = svm(train_y_original~., data = svr_x_train)

# Using model to make prediction on the train and test set 
train_y_predicted_svr = predict(svr_model, svr_x_train)
test_y_predicted_svr = predict(svr_model, svr_x_test)

# Calculating R-Squared Value for training and test data
#Train
train_sumofsquares_total_svr <- sum((train_y_original - mean(train_y_original))^2)
train_sumofsquares_residuals_svr <- sum((train_y_original - train_y_predicted_svr)^2)
train_svr_r2 <- 1 - train_sumofsquares_residuals_svr / train_sumofsquares_total_svr
#Test
test_sumofsquares_total_svr <- sum((test_y_original - mean(test_y_original))^2)
test_sumofsquares_residuals_svr <- sum((test_y_original - test_y_predicted_svr)^2)
test_svr_r2 <- 1 - test_sumofsquares_residuals_svr / test_sumofsquares_total_svr

# Appending the values of PLS R-Squared to Data Frame
r_squared_values[nrow(r_squared_values) + 1,] = c('Support Vector Regression', train_svr_r2, test_svr_r2)

############################################### Gaussian Process Regression ##############################################################
gpr_x_train <- original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
gpr_x_test <- original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]

#Develop the model on the training data
gpr_model<-gausspr(gpr_x_train,train_y_original)

#Training and testing outputs from the models
train_y_predicted_gpr<-predict(gpr_model,gpr_x_train)
test_y_predicted_gpr<-predict(gpr_model,gpr_x_test)

# Calculating R-Squared Value for training and test data
#Train
train_gpr_sst <- sum((train_y_original - mean(train_y_original))^2)
train_gpr_sse <- sum((train_y_predicted_gpr - train_y_original)^2)
#Test
test_gpr_sst <- sum((test_y_original - mean(test_y_original))^2)
test_gpr_sse <- sum((test_y_predicted_gpr - test_y_original)^2)

train_gpr_rsq <- 1 - train_gpr_sse/train_gpr_sst
test_gpr_rsq  <- 1 - test_gpr_sse/test_gpr_sst

# Appending the values of PCR R-Squared to Data Frame
r_squared_values[nrow(r_squared_values) + 1,] = c('Gaussian Process Regression', train_gpr_rsq, test_gpr_rsq)

############################################### Random Forest Regression ##############################################################
rf_train <- original_df[1:80, c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]
rf_test <- original_df[81:nrow(original_df), c("Feature1", "Feature2", "Feature3", "Feature4", "Feature5", "Feature6", "Feature7", "Feature8")]

train_y_original <- original_df[1:80, c("Property")]
test_y_original <- original_df[81:nrow(original_df), c("Property")]

model<-randomForest(train_y_original~.,data=rf_train, mtry=3,importance=TRUE, na.action=na.omit)
rf_pred_train<-predict(model,rf_train)
rf_pred_test<-predict(model,rf_test)

# Calculating R-Squared Value for training and test data
#Train
train_rf_sst <- sum((train_y_original - mean(train_y_original))^2)
train_rf_sse <- sum((rf_pred_train - train_y_original)^2)
#Test
test_rf_sst <- sum((test_y_original - mean(test_y_original))^2)
test_rf_sse <- sum((rf_pred_test - test_y_original)^2)

train_rf_rsq <- 1 - train_rf_sse/train_rf_sst
test_rf_rsq  <- 1 - test_rf_sse/test_rf_sst

r_squared_values[nrow(r_squared_values) + 1,] = c('Random Forest Regression', train_rf_rsq, test_rf_rsq)


#############################################################################################################################
# Exporting CSV file for further visualizations
write.csv(r_squared_values,"r_squared_values.csv", row.names = FALSE)