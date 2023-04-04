install.packages("smotefamily")
install.packages("ISLR")
install.packages("ROSE")
install.packages("tidyverse")
install.packages("bruceR")
install.packages("devtools")
install.packages("mlr3")
install.packages("mlr3learners")
install.packages("mlr3viz")
install.packages("mlr3verse")
install.packages("ggplot2")
install.pakcages("nnet")
install.packages("kknn")
install.packages("e1071")
install.packages("ranger")
install.packages("xgboost")
install.packages("precrec")
install.packages("aqp")
library(readr)
library(dplyr)
library(nnet)
library(lubridate)
library(smotefamily)
library(caret)
library(ISLR)
library(pROC)
library(randomForest)
library(rpart)
library(rpart.plot)
library(data.table)
library(gridExtra)
library(DMwR2)
library(ROSE)
library(tidyverse)
library(data.table)
library(mlr3)
library(mlr3learners)
library(mlr3viz)
library(mlr3verse)
library(mlbench)
library(kknn)
library(e1071)
library(ranger)
library(xgboost)
library(precrec)
library(ggplot2)
library(tidyr)
library(aqp)

# Group 21

transform_data <- function(file_name){
  # read data
  dataset <- read.csv(file_name)
  dataset <- as.data.frame(dataset)
  summary(dataset)
  
  
  # Data Cleaning
  # if there is empty value or missing value
  any(is.na(dataset))
  any(dataset=='')
  colSums(is.na(dataset))
  colSums(dataset=='')
  
  
  # Data Transformation
  # change date of birth(dob) to age
  date_sr <- ymd_hms(dataset$trans_date_trans_time)
  date_ed <- ymd(dataset$dob)
  date_all <- interval(date_ed,date_sr)
  dataset$dob <- round(time_length(date_all,'year'),0)
  dataset <- rename(dataset, c("age" = "dob"))
  
  # change transaction date and time to hour week and month
  dataset$hour <- hour(ymd_hms(dataset$trans_date_trans_time))
  dataset$week <- wday(ymd_hms(dataset$trans_date_trans_time))
  dataset$month <- month(ymd_hms(dataset$trans_date_trans_time))
  
  # change category and state to numeric variable
  normset <- dataset[c('amt','category','state','lat','long','merch_lat','merch_long','age','hour','week','month')]
  normset <- normset %>% mutate_if(is.character,as.factor)
  normset <- normset %>% mutate_if(is.factor,as.numeric)
  if (file_name == "archive/fraudTest.csv"){
    normset$state <- ifelse(normset$state>=9, normset$state+1, normset$state)
  }

  
  # the normalization of the numeric data
  normset$hour <- as.numeric(normset$hour)
  normset <- scale(normset,center=F,scale=T)
  
  transformed_dataset <- cbind(normset,dataset$is_fraud)
  transformed_dataset <- as.data.frame(transformed_dataset)
  
  transformed_dataset <- rename(transformed_dataset, c("is_fraud" = "V12"))
  return(transformed_dataset)
}

# transform Training Data
TrainSet <- transform_data("archive/fraudTrain.csv")

# Sampling
table(TrainSet$is_fraud)

#balance TrainSet and ValidationSet
# ROSE
TrainSet_Rose <- ROSE(is_fraud~.,data=TrainSet,seed=1)$data
#write.csv(TrainSet_Rose,"archive/TrainSet_Rose.csv")

# SMOTE
TrainSet_Smote <- SMOTE(TrainSet[,-12],TrainSet[,12],dup_size=0,K=5)$data
TrainSet_Smote$class <- as.numeric(TrainSet_Smote$class)
TrainSet_Smote <- rename(TrainSet_Smote,c("is_fraud" = "class"))
#write.csv(TrainSet_Smote,"archive/TrainSet_Smote.csv")

# ADASYN
TrainSet_Adasyn <- ADAS(TrainSet[,-12],TrainSet[,12],K=5)$data
TrainSet_Adasyn$class <- as.numeric(TrainSet_Adasyn$class)
TrainSet_Adasyn <- rename(TrainSet_Adasyn,c("is_fraud" = "class"))
#write.csv(TrainSet_Adasyn,"archive/TrainSet_Adasyn.csv")


# Fraud Data Analytics

# choose models
# 1. logistic regression model
# 2. Neural Network model
# 3. KNN model
# 4. SVM model
# 5. Random Forest model 
# 6. XGBoost model

# transform 1 to "Fraud", 0 to "Not Fraud"
#TrainSet_Rose2 <- TrainSet_Rose[c("amt","lat","long","merch_lat","merch_long","age","hour","week","month","is_fraud")]
TrainSet_Rose$is_fraud <- ifelse(TrainSet_Rose$is_fraud == 1, "Fraud", "Not_Fraud")
TrainSet_Rose$is_fraud <- as.factor(TrainSet_Rose$is_fraud)
TrainSet_Smote$is_fraud <- ifelse(TrainSet_Smote$is_fraud == 1, "Fraud", "Not_Fraud")
TrainSet_Smote$is_fraud <- as.factor(TrainSet_Smote$is_fraud)
TrainSet_Adasyn$is_fraud <- ifelse(TrainSet_Adasyn$is_fraud == 1, "Fraud", "Not_Fraud")
TrainSet_Adasyn$is_fraud <- as.factor(TrainSet_Adasyn$is_fraud)

# transform Training Data
TestSet <- transform_data("archive/fraudTest.csv")
TestSet <- na.omit(TestSet)
any(is.na(TestSet))
# balance TestSet
table(TestSet$is_fraud)
TestSet_Adasyn <- ADAS(TestSet[,-12],TestSet[,12],K=5)$data
TestSet_Adasyn$class <- as.numeric(TestSet_Adasyn$class)
TestSet_Adasyn <- rename(TestSet_Adasyn,c("is_fraud" = "class"))
TestSet_Adasyn$is_fraud <- ifelse(TestSet_Adasyn$is_fraud == 1, "Fraud", "Not_Fraud")
TestSet_Adasyn$is_fraud <- as.factor(TestSet_Adasyn$is_fraud)
TestSet$is_fraud <- ifelse(TestSet$is_fraud == 1, "Fraud", "Not_Fraud")
TestSet$is_fraud <- as.factor(TestSet$is_fraud)


# logistic regression model
lr_model <- function(TrainingSet,TestingSet){
  task_lr <- TaskClassif$new("detectfraud", TrainingSet, target = "is_fraud")
  learner_lr <- lrn("classif.log_reg",predict_type = "prob")
  learner_lr$param_set
  search_space_lr <- ps(
    epsilon = p_dbl(1e-09, 1e-07),
    maxit = p_int(5,50)
  )
  terminator_lr <- trm("evals", n_evals = 6)
  tuner_lr <- tnr("grid_search")
  resampling_lr <- rsmp("holdout",ratio = 0.7)
  measure_lr <- msr("classif.recall")
  autoTuner_lr <- AutoTuner$new(
    learner = learner_lr,
    resampling = resampling_lr,
    search_space = search_space_lr,
    measure = measure_lr,
    tuner = tuner_lr,
    terminator = terminator_lr
  )
  options(warn = 1)
  autoTuner_lr$train(task_lr)
  autoTuner_lr$model
  task_test_lr = TaskClassif$new("testfraud", TestingSet, target = "is_fraud")
  predictions_lr <- autoTuner_lr$predict(task_test_lr)
  return(predictions_lr)
}


# Neural Network model
nn_model <- function(TrainingSet,TestingSet){
  task_nn <- TaskClassif$new("detectfraud", TrainingSet, target = "is_fraud")
  learner_nn <- lrn("classif.nnet",predict_type = "prob")
  learner_nn$param_set
  search_space_nn <- ps(
    MaxNWts = p_int(100,2000),
    maxit = p_int(50,500),
    size = p_int(1,5)
  )
  terminator_nn <- trm("evals", n_evals = 6)
  tuner_nn <- tnr("grid_search")
  resampling_nn <- rsmp("holdout",ratio = 0.7)
  measure_nn <- msr("classif.recall")
  autoTuner_nn <- AutoTuner$new(
    learner = learner_nn,
    resampling = resampling_nn,
    search_space = search_space_nn,
    measure = measure_nn,
    tuner = tuner_nn,
    terminator = terminator_nn
  )
  options(warn = 1)
  autoTuner_nn$train(task_nn)
  autoTuner_nn$model
  task_test_nn = TaskClassif$new("testfraud", TestingSet, target = "is_fraud")
  predictions_nn <- autoTuner_nn$predict(task_test_nn)
  return(predictions_nn)
}


# KNN model
knn_model <- function(TrainingSet,TestingSet){
  task_knn <- TaskClassif$new("detectfraud", TrainingSet, target = "is_fraud")
  learner_knn <- lrn("classif.kknn",predict_type = "prob",k=9)
  learner_knn$param_set
  search_space_knn <- ps(
    k = p_int(1,20)
  )
  terminator_knn <- trm("evals", n_evals = 3)
  tuner_knn <- tnr("grid_search")
  resampling_knn <- rsmp("holdout",ratio = 0.7)
  measure_knn <- msr("classif.recall")
  autoTuner_knn <- AutoTuner$new(
    learner = learner_knn,
    resampling = resampling_knn,
    search_space = search_space_knn,
    measure = measure_knn,
    tuner = tuner_knn,
    terminator = terminator_knn
  )
  options(warn = 1)
  autoTuner_knn$train(task_knn)
  autoTuner_knn$model
  task_test_knn = TaskClassif$new("testfraud", TestingSet, target = "is_fraud")
  predictions_knn <- autoTuner_knn$predict(task_test_knn)
  return(predictions_knn)
}


# SVM model
#svm_model <- function(TrainingSet,TestingSet){
#  task_svm <- TaskClassif$new("detectfraud", TrainingSet, target = "is_fraud")
#  learner_svm <- lrn("classif.svm",predict_type = "prob")
#  learner_svm$param_set
#  search_space_svm <- ps(
#    cross = p_int(-1,2),
#    epsilon = p_dbl(0.01,1),
#    tolerance = p_dbl(0.0001,0.01)
#    
#  )
#  terminator_svm <- trm("evals", n_evals = 3)
#  tuner_svm <- tnr("grid_search")
#  resampling_svm <- rsmp("holdout",ratio = 0.7)
#  measure_svm <- msr("classif.recall")
#  autoTuner_svm <- AutoTuner$new(
#    learner = learner_svm,
#    resampling = resampling_svm,
#    search_space = search_space_svm,
#    measure = measure_svm,
#    tuner = tuner_svm,
#    terminator = terminator_svm
#  )
#  options(warn = 1)
#  autoTuner_svm$train(task_svm)
#  autoTuner_svm$model
#  task_test_svm = TaskClassif$new("testfraud", TestingSet, target = "is_fraud")
#  predictions_svm <- autoTuner_svm$predict(task_test_svm)
#  return(predictions_svm)
#}


# random forest model
rf_model <- function(TrainingSet,TestingSet){
  task_rf <- TaskClassif$new("detectfraud", TrainingSet, target = "is_fraud")
  learner_rf <- lrn("classif.ranger",predict_type = "prob")
  learner_rf$param_set
  search_space_rf <- ps(
    num.trees = p_int(300,500)
  )
  terminator_rf <- trm("evals", n_evals = 3)
  tuner_rf <- tnr("grid_search")
  resampling_rf <- rsmp("holdout",ratio = 0.7)
  measure_rf <- msr("classif.recall")
  autoTuner_rf <- AutoTuner$new(
    learner = learner_rf,
    resampling = resampling_rf,
    search_space = search_space_rf,
    measure = measure_rf,
    tuner = tuner_rf,
    terminator = terminator_rf
  )
  options(warn = 1)
  autoTuner_rf$train(task_rf)
  autoTuner_rf$model
  task_test_rf = TaskClassif$new("testfraud", TestingSet, target = "is_fraud")
  predictions_rf <- autoTuner_rf$predict(task_test_rf)
  return(predictions_rf)
}


# XGBoost model
xgb_model <- function(TrainingSet,TestingSet){
  task_xgb <- TaskClassif$new("detectfraud", TrainingSet, target = "is_fraud")
  learner_xgb <- lrn("classif.xgboost",predict_type = "prob")
  learner_xgb$param_set
  search_space_xgb <- ps(
    max_depth = p_int(3,9)
  )
  terminator_xgb <- trm("evals", n_evals = 6)
  tuner_xgb <- tnr("grid_search")
  resampling_xgb <- rsmp("holdout",ratio = 0.7)
  measure_xgb <- msr("classif.recall")
  autoTuner_xgb <- AutoTuner$new(
    learner = learner_xgb,
    resampling = resampling_xgb,
    search_space = search_space_xgb,
    measure = measure_xgb,
    tuner = tuner_xgb,
    terminator = terminator_xgb
  )
  options(warn = 1)
  autoTuner_xgb$train(task_xgb)
  autoTuner_xgb$model
  task_test_xgb = TaskClassif$new("testfraud", TestingSet, target = "is_fraud")
  predictions_xgb <- autoTuner_xgb$predict(task_test_xgb)
  return(predictions_xgb)
}


print_result <- function(predictions,prefix_name){
  file_name <- paste0(prefix_name,".txt")
  # Confusion Matrix
  cat("Confusion Matrix",file=file_name,sep='\n',append=T)
  cat(predictions$confusion,file=file_name,sep='\n',append=T)
  # Accuracy
  cat("Accuracy:",file=file_name,sep='\n',append=T)
  cat(predictions$score(msr("classif.acc")),sep='\n',file=file_name,append=T)
  # Recall
  cat("Recall:",file=file_name,sep='\n',append=T)
  cat(predictions$score(msr("classif.recall")),sep='\n',file=file_name,append=T)
  # Precision
  cat("Precision:",file=file_name,sep='\n',append=T)
  cat(predictions$score(msr("classif.precision")),sep='\n',file=file_name,append=T)
  # F1 score
  cat("F1 score:",file=file_name,sep='\n',append=T)
  cat(predictions$score(msr("classif.fbeta")),sep='\n',file=file_name,append=T)
  # ROC-AUC
  cat("POC-AUC:",file=file_name,sep='\n',append=T)
  cat(predictions$score(msr("classif.auc")),sep='\n',file=file_name,append=T)
  # plot ROC
  png(filename = paste0(prefix_name,".jpg"))
  print(autoplot(predictions, type = "roc"))
  dev.off()
}

# rose
print("logistic regression model")
predictions_lr_rose <- lr_model(TrainSet_Rose,TestSet)
print_result(predictions_lr_rose,"lr_rose")


print("Neural Network model")
predictions_nn_rose <- nn_model(TrainSet_Rose,TestSet)
print_result(predictions_nn_rose,"nn_rose")


print("KNN model")
predictions_knn_rose <- knn_model(TrainSet_Rose,TestSet)
print_result(predictions_knn_rose,"knn_rose")


#print("SVM model")
#predictions_svm_rose <- svm_model(TrainSet_Rose,TestSet)
#print_result(predictions_svm_rose,"svm_rose")


print("Random Forest model")
predictions_rf_rose <- rf_model(TrainSet_Rose,TestSet)
print_result(predictions_rf_rose,"rf_rose")


print("XGBoost model")
predictions_xgb_rose <- xgb_model(TrainSet_Rose,TestSet)
print_result(predictions_xgb_rose,"xgb_rose")


# smote
print("logistic regression model")
predictions_lr_smote <- lr_model(TrainSet_Smote,TestSet)
print_result(predictions_lr_smote,"lr_smote")


print("Neural Network model")
predictions_nn_smote <- nn_model(TrainSet_Smote,TestSet)
print_result(predictions_nn_smote,"nn_smote")


print("KNN model")
predictions_knn_smote <- knn_model(TrainSet_Smote,TestSet)
print_result(predictions_knn_smote,"knn_smote")


#print("SVM model")
#predictions_svm_smote <- svm_model(TrainSet_Smote,TestSet)
#print_result(predictions_svm_smote,"svm_smote")


print("Random Forest model")
predictions_rf_smote <- rf_model(TrainSet_Smote,TestSet)
print_result(predictions_rf_smote,"rf_smote")


print("XGBoost model")
predictions_xgb_smote <- xgb_model(TrainSet_Smote,TestSet)
print_result(predictions_xgb_smote,"xgb_smote")


# adasyn
print("logistic regression model")
predictions_lr_adasyn <- lr_model(TrainSet_Adasyn,TestSet)
print_result(predictions_lr_adasyn,"lr_adasyn")


print("Neural Network model")
predictions_nn_adasyn <- nn_model(TrainSet_Adasyn,TestSet)
print_result(predictions_nn_adasyn,"nn_adasyn")


print("KNN model")
predictions_knn_adasyn <- knn_model(TrainSet_Adasyn,TestSet)
print_result(predictions_knn_adasyn,"knn_adasyn")


#print("SVM model")
#predictions_svm_adasyn <- svm_model(TrainSet_Adasyn,TestSet)
#print_result(predictions_svm_adasyn,"svm_adasyn")


print("Random Forest model")
predictions_rf_adasyn <- rf_model(TrainSet_Adasyn,TestSet)
print_result(predictions_rf_adasyn,"rf_adasyn")


print("XGBoost model")
predictions_xgb_adasyn <- xgb_model(TrainSet_Adasyn,TestSet)
print_result(predictions_xgb_adasyn,"xgb_adasyn")

# compare xgboost model with different testsets
print("XGBoost model")
predictions_xgb_adasyn <- xgb_model(TrainSet_Adasyn,TestSet_Adasyn)
print_result(predictions_xgb_adasyn,"xgb_adasyn")