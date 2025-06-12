#linear regression models 
set.seed(1) 

features <- subset(wine_data, select = c(1:12))
train=sample(nrow(wine_data),size = nrow(wine_data)/2) 
test = wine_data[-train,]
training = wine_data[train,]
xtrain = features[train,]
matrix.xtrain = model.matrix(~., data = xtrain)[, -1]
ytrain = wine_data$quality[train]
xtest=features[-train,]  #test data
matrix.xtest = model.matrix(~., data = xtest)[, -1]
ytest=wine_data$quality[-train]

#does 10-fold CV
k_fold_cv <- function(wine_data, K = 10) {
  n <- nrow(wine_data)
  folds <- sample(rep(1:K, length.out = n)) 
  
  mse_values <- numeric(K) 
  
  for (k in 1:K) {
    test_data <- wine_data[folds == k, ]
    train_data <- wine_data[folds != k, ]
    
    # testing regression model with all covariates
    model <- lm(quality~., data = train_data)
    
    predictions <- predict(model, newdata = test_data)
    
    mse_values[k] <- mean((predictions - test_data$quality)^2)
  }
  return(mean(mse_values))
}

cv_mse <- k_fold_cv(wine_data, K = 10)
cat("10-Fold CV Error for Linear Regression With All Covariates", cv_mse,"\n")

k_fold_cv <- function(wine_data, K = 10) {
  n <- nrow(wine_data)
  folds <- sample(rep(1:K, length.out = n))  
  
  mse_values <- numeric(K)  
  for (k in 1:K) {
    test_data <- wine_data[folds == k, ]
    train_data <- wine_data[folds != k, ]
    
    # testing regression model with statistically insignificant covariates removed 
    model <- lm(quality~. -chlorides - citric.acid, data = train_data)
    
    predictions <- predict(model, newdata = test_data)
    
    mse_values[k] <- mean((predictions - test_data$quality)^2)
  }
  return(mean(mse_values))
}

cv_mse <- k_fold_cv(wine_data, K = 10)

cat("10-Fold CV Error for Sparser Linear Regression:", cv_mse,"\n")



#best subsets
set.seed(1)
library(leaps)
n <- nrow(wine_data)

k_fold_cv <- function(wine_data, K = 10) {
  
  mse_values <- matrix(NA, K, 12)  
  folds <- sample(rep(1:K, length.out = n))
  
  for (k in 1:K) {
    test_data <- wine_data[folds == k, ]
    train_data <- wine_data[folds != k, ]
    
    # testing best subsets regression
    model <- regsubsets(quality~.,data=train_data,nvmax=12)
    
    test_mat <- model.matrix(quality ~ ., data = test_data)
    
    for (j in 1:12) {
      coef_j <- coef(model, id = j)
      vars <- names(coef_j)
      
      pred <- test_mat[, vars, drop = FALSE] %*% coef_j
      mse_values[k, j] <- mean((test_data$quality - pred)^2)
    }
  }
  
  avg_mse <- colMeans(mse_values, na.rm = TRUE)
  mse_df <- data.frame(
    Num_Predictors = 1:12,
    Mean_CV_MSE = avg_mse
  )
  return(mse_df)
}

cv_mse <- k_fold_cv(wine_data, K = 10)
print(cv_mse)



#to make the grid, because large n and small p, grid can be wider to not penalize as much
grid = 10^seq(10, -4, length = 100)
set.seed(1)
library(glmnet)

x = model.matrix(quality ~ ., wine_data)[, -1]
y = wine_data$quality
K <- 10
folds <- sample(rep(1:K, length.out = nrow(wine_data)))

#ridge
test_mse_ridge = c()

for (k in 1:10) {
  test_idx = which(folds == k)
  train_idx = which(folds != k)
  
  x_train = x[train_idx, ]
  y_train = y[train_idx]
  x_test = x[test_idx, ]
  y_test = y[test_idx]
  
  #10-fold CV to find optimal parameter lambda
  cv_ridge = cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)
  y_pred = predict(cv_ridge, s = "lambda.min", newx = x_test)
  
  mse = mean((y_pred - y_test)^2)
  test_mse_ridge[k] = mse
}

cat("10-Fold CV Ridge Error",mean(test_mse_ridge), "\n")


#lasso
test_mse_lasso = c()

for (k in 1:10) {
  test_idx = which(folds == k)
  train_idx = which(folds != k)
  
  x_train = x[train_idx, ]
  y_train = y[train_idx]
  x_test = x[test_idx, ]
  y_test = y[test_idx]
  
  #10-fold CV to find optimal parameter lambda
  cv_lasso = cv.glmnet(x_train, y_train, alpha = 1, nfolds = 10)
  
  y_pred = predict(cv_lasso, s = "lambda.min", newx = x_test)
  
  # Compute MSE
  mse = mean((y_pred - y_test)^2)
  
  # Store
  test_mse_lasso[k] = mse
}

cat("10-Fold CV Lasso Error",mean(test_mse_lasso), "\n")



#PCR
library(pls)
set.seed(1)
K <- 10
folds <- sample(rep(1:K, length.out = nrow(wine_data)))

fold_mse <- numeric(K)

for (k in 1:K) {
  pcr.fit=pcr(quality~., data=wine_data[folds != k, ],scale=TRUE,validation="CV")
  
  rmse_vals <- RMSEP(pcr.fit)$val[1, , -1]
  best_min <- which.min(rmse_vals)
  
  y_pred <- predict(pcr.fit, newdata = wine_data[folds == k, ], ncomp = best_min)
  y_true <- wine_data[folds == k, ]$quality
  
  fold_mse[k] <- mean((y_pred - y_true)^2)
}

cv_mse_pcr <- mean(fold_mse)

cat("10-Fold CV PCR Error:", cv_mse_pcr, "\n")



#boosting for regression
library(xgboost)
library(dplyr)

set.seed(1)

wine_data_params <- as.matrix(wine_data |> select(-quality))
wine_data_quality <- as.matrix(wine_data |> select(quality))

dtrain <- xgb.DMatrix(data = wine_data_params, label = wine_data_quality)

model <- xgb.cv(data =  dtrain,
                objective = "reg:squarederror", 
                nrounds = 100, 
                nfold = 10,
                eval_metric = "rmse",
                verbose = 0) #### With 10 folds

mse_boost <- min(summary(model$evaluation_log$test_rmse_mean))^2

cat("10-Fold CV Boosting Regression MSE:", mse_boost, "\n")