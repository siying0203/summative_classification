library("skimr")
library("data.table")
library("mlr3verse")
library("tidyverse")
library("ggplot2")
library("tidymodels")
library(caret)
library("rsample")
library("recipes")
library("keras")

set.seed(457)
bank_data <- read.csv("https://www.louisaslett.com/Courses/MISCADA/bank_personal_loan.csv")
skim(bank_data)
DataExplorer::plot_bar(bank_data, )
DataExplorer::plot_histogram(bank_data, )

#Data Clean----omit the negative value
#mean_exp <- mean(bank_data$Experience[bank_data$Experience >= 0])
bank_data["Personal.Loan"]<-lapply(bank_data["Personal.Loan"],factor)
bank_data <- bank_data|>select(-ZIP.Code)
bank_data$Experience[bank_data$Experience < 0] <- NA
bank_data <- na.omit(bank_data)


###fit the model
#define a task and set seed for reproducibility
bank_data_task1 <- TaskClassif$new(id = "task_1",
                                   backend = bank_data,
                                   target = "Personal.Loan",
                                   positive = "1")

#use rsmp function constructs a resampling strategy
set.seed(321)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_data_task1)

#use rsmp constructs a boostrap strategy
set.seed(321)
bootstrap <- rsmp("bootstrap",repeats = 10)
bootstrap$instantiate(bank_data_task1)

#set learners
lrn_rpart <- lrn("classif.rpart", predict_type = "prob")
lrn_log_reg <- lrn("classif.log_reg",predict_type = "prob")
lrn_qda <-lrn("classif.qda",predict_type="prob")
lrn_glmnet <-lrn("classif.glmnet",predict_type="prob")
lrn_kknn <-lrn("classif.kknn",predict_type="prob")


#fit these learners with benchmark
set.seed(321)
res_cv5 <- benchmark(data.table(
  task       = list(bank_data_task1),
  learner    = list(lrn_rpart,lrn_log_reg,
                    lrn_qda,lrn_glmnet,lrn_kknn),
  resampling = list(cv5)
), store_models = TRUE)

res_boots <- benchmark(data.table(
  task       = list(bank_data_task1),
  learner    = list(lrn_rpart,lrn_log_reg,
                    lrn_qda,lrn_glmnet,lrn_kknn),
  resampling = list(bootstrap)
), store_models = TRUE)

# Look at accuracy
l <- list(msr("classif.ce"),
          msr("classif.acc"),
          msr("classif.auc"),
          msr("classif.fpr"),
          msr("classif.fnr"))
res <- rbind(res_cv5$aggregate(l),res_boots$aggregate(l))
res
autoplot(res_cv5,type='roc')+ theme_bw()
autoplot(res_boots,type='roc')+ theme_bw()



###compare with the facotered data
bank_data_f <- bank_data|>
  mutate(Education = as.factor(Education))

#define a task and set seed for reproducibility
set.seed(321)
bank_data_task2 <- TaskClassif$new(id = "task_2",
                                   backend = bank_data_f,
                                   target = "Personal.Loan",
                                   positive = "1")

#use rsmp constructs a cv strategy
set.seed(321)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_data_task1)

lrn_rpart_f <-lrn("classif.rpart",predict_type="prob",cp=0)

set.seed(321)
res_boots_f <- benchmark(data.table(
  task       = list(bank_data_task2),
  learner    = list(lrn_rpart_f),
  resampling = list(cv5)
), store_models = TRUE)
print(res_boots_f$aggregate(l))


###Enable cross validation
set.seed(321)
lrn_rpart_cv <- lrn("classif.rpart", predict_type = "prob", xval=10)
res_rpart_cv <- resample(bank_data_task1, lrn_rpart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_rpart_cv$learners[[5]]$model)



###test/train/validate
#training =0.6,test=0.2,validation=0.2
set.seed(532)
bank_data_split1 <-initial_split(bank_data,prop=0.6)
bank_data_train <-training(bank_data_split1)

bank_data_split2 <- initial_split(testing(bank_data_split1),prop=0.5)
bank_data_test <-training(bank_data_split2)
bank_data_valid<-testing(bank_data_split2)

bank_data_train_task <-TaskClassif$new(id = "task_3",
                                       backend = bank_data_train,
                                       target = "Personal.Loan",
                                       positive = "1")

bank_data_test_task <-TaskClassif$new(id = "task_4",
                                      backend = bank_data_test,
                                      target = "Personal.Loan",
                                      positive = "1")

set.seed(321)
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(bank_data_train_task)

set.seed(321)
lrn_rpart$train(bank_data_train_task)
val_rpart <- lrn_rpart$predict_newdata (bank_data_valid)
pred_rpart <- lrn_rpart$predict(bank_data_test_task)

cat("the accuracy on the validation set is:")
val_rpart$score(msr("classif.acc"))
cat("the accuracy on the test set is:")
pred_rpart$score(msr("classif.acc"))
pred_rpart$confusion
autoplot(pred_rpart)



###Get the resample results from cv 
trees <- res_cv5$resample_result(1)
#look at the tree from 4th CV iteration
tree1 <- trees$learners[[4]]
tree1_rpart <- tree1$model
# Plot the tree
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.8)


###use grid search to search penalty
set.seed(849)
paramGrid <- expand.grid(.cp = seq(0, 0.02, 0.001))
control <- trainControl(method="cv", number=5)
set.seed(321)
GS <- train(Personal.Loan ~ .,data=bank_data,method="rpart", 
            trControl=control,metric="Accuracy",tuneGrid=paramGrid)

print(GS$bestTune)
cp<-GS$bestTune[,1]



###choose a cost penalty
lrn_rpart_cp <-lrn("classif.rpart",predict_type="prob",cp=cp)
set.seed(321)
res_boots_cp <- benchmark(data.table(
  task       = list(bank_data_task1),
  learner    = list(lrn_rpart_cp ),
  resampling = list(cv5)
), store_models = TRUE)
print(res_boots_cp$aggregate(l))

