#boosting for classification
quality_factor <- as.factor(wine_data$quality)
wine_data_labels <- as.numeric(quality_factor) - 1

dtrain <- xgb.DMatrix(data = wine_data_params, label = wine_data_labels)

model_soft_problem <- xgb.cv(
  data = dtrain,
  objective = "multi:softprob",
  num_class = 7,
  nrounds = 100,
  nfold = 10,
  early_stopping_rounds = 10,
  eval_metric = "merror",
  verbose = 0,
  print_every_n = 1
)

class_error_boost <- min(summary(model_soft_problem$evaluation_log$test_merror_mean))

cat("10-Fold CV Boosting Classification Error:", class_error_boost, "\n")



# decision tree
data_wine <- wine_data
data_wine$quality <- factor(data_wine$quality) #changing quality to be a factor 
library(tree)
set.seed(1)
K <- 10
errors <- numeric(K)
folds <- sample(rep(1:K, length.out = nrow(wine_data)))
for (k in 1:K) {
  model <- tree(
    quality ~ .,
    data = data_wine[which(folds != k), ]
  )
  preds <- predict(model, data_wine[which(folds == k), ], type = "class")
  errors[k] <- mean(preds != data_wine$quality[which(folds == k)])
}
tree_class_error <- mean(errors)
cat("10-Fold CV Decision Tree Classification Error :", tree_class_error, "\n")
#plotting the last tree from the 10th fold
plot(model)
text(model, pretty=0)
#pruning attempt on last tree from the 10th fold
cv.tree <- cv.tree(model, FUN = prune.misclass)
plot(cv.tree$size, cv.tree$dev, type = "b", xlab = "Tree Size", ylab = "CV Error")
prune_tree <- prune.misclass(model, best = cv.tree$size[which.min(cv.tree$dev)])
plot(prune_tree)
text(prune_tree, pretty=0)
summary(prune_tree)



#bagging
library(ranger)
set.seed(1)
K <- 10
folds <- sample(rep(1:K, length.out = nrow(wine_data)))
errors <- numeric(K)
for (k in 1:K) {
  model <- ranger(
    quality ~ .,
    data = data_wine[which(folds != k), ],
    mtry = 12,
    seed = 1,#for reproducibility
    classification=TRUE#bagging is random forest where m = p
  )
  preds <- predict(model, data_wine[which(folds == k), ])$predictions
  errors[k] <- mean(preds != data_wine$quality[which(folds == k)])
}
bagging_class_error <- mean(errors)
cat("Bagging Classification Error:", bagging_class_error, "\n")



#random forest
set.seed(1)
K <- 10
errors <- numeric(K)
folds <- sample(rep(1:K, length.out = nrow(wine_data)))
library(ranger)
for (k in 1:K) {
  model <- ranger(
    quality ~ .,
    data = data_wine[which(folds != k), ],
    mtry = floor(sqrt(ncol(wine_data) - 1)), # RF Suggests Sq Root (we did floor)
    seed = 1, #for reproducibility
    classification=TRUE
  )
  preds <- predict(model, data_wine[which(folds == k), ])$predictions
  errors[k] <- mean(preds != data_wine$quality[which(folds == k)])
}
RF_class_error <- mean(errors)
cat("10-Fold CV Random Forest Classification Error :", RF_class_error, "\n")