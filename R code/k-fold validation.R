#####################
##### load data #####
#####################
setwd("C:/Users/Administrator/Desktop/myproject/")

telcom <- read.csv("R code/data/churn.csv", header = TRUE)
head(telcom)

### drop unnecessary variables ###
telcom <- telcom[ , -c(1, 3)]
head(telcom)

### tabulate the outcome variable ###
table(telcom$Churn)

### declare factor variables ###
telcom$International.plan <- as.factor(telcom$International.plan)
telcom$Voice.mail.plan <- as.factor(telcom$Voice.mail.plan)
telcom$Churn <- as.factor(telcom$Churn)


### logistic regression ###
logitr <- glm(Churn ~ International.plan + Number.vmail.messages, family="binomial", data=telcom)
summary(logitr)

logitr <- glm(Churn ~ International.plan + Number.vmail.messages + Total.intl.calls + Customer.service.calls, family="binomial", data=telcom)
summary(logitr)

logitr <- glm(Churn ~ ., family="binomial", data=telcom)
summary(logitr)


### predicted probabilities ###
telcom$pprob <- predict(logitr, type="response")
table(telcom$pprob>0.5)
table(telcom$Churn)

### confusion matrix ###
confm <- table(telcom$pprob>0.5, telcom$Churn)
confm

### misclassification rate ###
(confm[1,2] + confm[2,1]) / sum(confm)

### accuracy ###
(confm[1,1] + confm[2,2]) / sum(confm)

### sensitivity ###
confm[2,2] / (confm[1,2] + confm[2,2])

### specificity ###
confm[1,1] / (confm[1,1] + confm[2,1])


### ROC ###
library(pROC)
logitr_roc <- roc(telcom$Churn, telcom$pprob)
plot.roc(logitr_roc, legacy.axes=TRUE, print.auc=TRUE, print.thres="best")
coords(logitr_roc, "best", ret="threshold", transpose = FALSE)

### AUC ###
auc(logitr_roc)


### confusion matrix at the best threshold ###
cutoff <- as.numeric(coords(logitr_roc, "best", ret="threshold", transpose = FALSE))
confm <- table(telcom$pprob > cutoff, telcom$Churn)
confm

### misclassification rate ###
(confm[1,2] + confm[2,1]) / sum(confm)

### accuracy ###
(confm[1,1] + confm[2,2]) / sum(confm)

### sensitivity ###
confm[2,2] / (confm[1,2] + confm[2,2])

### specificity ###
confm[1,1] / (confm[1,1] + confm[2,1])




###################################
##### k-fold cross validation #####
###################################
library(MASS)

# randomly shuffle the data
set.seed(1)
telcomcv <- telcom[sample(nrow(telcom), replace=FALSE),]

# set the number of folds
k <- 4

# create k equally-sized folds
folds <- cut(seq(1,nrow(telcomcv)), breaks=k, labels=FALSE)

# create an empty vector of auc
auc.logit <- rep(NA, k)

for(i in 1:k){
  train <- telcomcv[folds!=i,]  # set the training set
  test <- telcomcv[folds==i,]   # set the test set
  
  # train a model on the training set
  #newglm <- glm(Churn ~ International.plan + Number.vmail.messages, family="binomial", data=train)
  #newglm <- glm(Churn ~ International.plan + Number.vmail.messages + Total.intl.calls + Customer.service.calls, family="binomial", data=train)
  newglm <- glm(Churn ~ ., family="binomial", data=train)
  
  # get out of sample predicitons
  test$pprob <- predict(newglm, type="response", newdata=test)
  
  logitr_roc <- roc(test$Churn, test$pprob)
  
  auc.logit[i] <- as.numeric(auc(logitr_roc)) 
}
auc.mean <- mean(auc.logit)
auc.mean

# re-estimate the final model #
newglm <- glm(Churn ~ ., family="binomial", data=telcomcv)
summary(newglm)

telcomcv$pprob <- predict(newglm, type="response", data=telcomcv)
logitr_roc <- roc(telcomcv$Churn, telcomcv$pprob)
coords(logitr_roc, "best", ret="threshold", transpose = FALSE)































