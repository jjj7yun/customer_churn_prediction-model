#install.packages("psych")
library(psych)
#install.packages("pastecs")
library(pastecs)
#install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
library(randomForest)


#####################################
##### Random forest: loan offer #####
#####################################

bank.df <- read.csv("universalbank.csv", header=TRUE)

# describes the structure of the data
str(bank.df)

# drop id and zip code
# bank.df <- subset(bank.df, select=-c(ID, ZIP.Code))
bank.df <- bank.df[, -c(1, 5)]
str(bank.df)

# declare variable type
bank.df$Education <- as.factor(bank.df$Education)
bank.df$Personal.Loan <- as.factor(bank.df$Personal.Loan)
bank.df$Securities.Account <- as.factor(bank.df$Securities.Account)
bank.df$CD.Account <- as.factor(bank.df$CD.Account)
bank.df$Online <- as.factor(bank.df$Online)
bank.df$CreditCard <- as.factor(bank.df$CreditCard)
str(bank.df)

# tabulate outcome values
table(bank.df$Personal.Loan)
prop.table(table(bank.df$Personal.Loan))


# build a forest with 500 trees
set.seed(999)
rf.init <- randomForest(formula = Personal.Loan ~ ., data = bank.df, proximity=TRUE)
rf.init

# oob error rate over 500 trees
rf.init$err.rate[,1]

# plot oob error rates
plot(rf.init$err.rate[,1], cex=.3, xlab="Number of trees", ylab="OOB error rate")

# identify the forest size with the lowest OOB error
which.min( rf.init$err.rate[,1] )

# OOB error rate at the optimal forest size
rf.init$err.rate[,1][  which.min(rf.init$err.rate[,1]) ]

# re-train the forest at a tuned forest size
set.seed(999)
rf.optsize <- randomForest(formula = Personal.Loan ~ ., data = bank.df, ntree = which.min( rf.init$err.rate[,1] ), proximity = TRUE)
rf.optsize


# tune no. of variables at split (forest size=optimal)
oob.values <- rep(NA, 4)
for(i in 1:4){
  set.seed(999)
  rf.temp <- randomForest(formula = Personal.Loan ~ ., data = bank.df, mtry = i, ntree = which.min( rf.init$err.rate[,1] ), proximity = TRUE)
  oob.values[i] <- rf.temp$err.rate[,1][  which.min(rf.temp$err.rate[,1]) ]
}
oob.values
which.min(oob.values)

# re-train the forest at a tuned no. of variables at split
set.seed(999)
rf.tuned <- randomForest(formula = Personal.Loan ~ ., data = bank.df, mtry = which.min(oob.values), ntree = which.min( rf.init$err.rate[,1] ), proximity=TRUE)
rf.tuned








###############################################
##### Customer churn prediction: logistic #####
###############################################
rm(list=ls())
telcom.b <- read.csv("churn_baseline.csv", header = TRUE)

# tabulate the outcome variable
table(telcom.b$Churn)

# declare factor variables
telcom.b$Churn <- as.factor(telcom.b$Churn)
telcom.b$Churn <- as.numeric(telcom.b$Churn)-1

# estimate model
final.glm <- glm(Churn ~ ., family="binomial", data=telcom.b)
summary(final.glm)

describe(telcom.b)

final.glm <- glm(Churn ~ Account.length + Number.vmail.messages + Total.day.minutes + Total.day.calls
                 + Total.day.charge + Total.eve.minutes + Total.eve.calls + Total.eve.charge + Total.night.minutes
                 + Total.night.calls + Total.night.charge + Total.intl.minutes + log(Total.intl.calls+.01) 
                 + Total.intl.charge + log(Customer.service.calls+.01), family="binomial", data=telcom.b)
summary(final.glm)

# read future data
telcom.f <- read.csv("churn_future.csv", header = TRUE)

# make classification and calculate accuracy
pprob <- predict(final.glm, type="response", newdata=telcom.f)
pclass <- ifelse(pprob>0.5, 1, 0)
mean(pclass == telcom.f$Churn)






##########################################
##### Customer churn prediction: kNN #####
##########################################
#install.packages('DEoptimR')
#install.packages("caret", dependencies = TRUE)
library(caret)
#install.packages('FNN')
library(FNN)
#install.packages('binaryLogic')
library(binaryLogic)

rm(list=ls())
telcom.b <- read.csv("churn_baseline.csv", header = TRUE)

# tabulate the outcome variable
table(telcom.b$Churn)

# declare factor variables
telcom.b$Churn <- as.factor(telcom.b$Churn)
telcom.b$Churn <- as.numeric(telcom.b$Churn)-1

# normalize data (features only)
preproc <- preProcess(telcom.b[, c(1:15)], method=c("range"))
telcom.b[, c(1:15)] <- predict(preproc, newdata=telcom.b[, c(1:15)])

# randomly shuffle the data
set.seed(1)
telcom.s <- telcom.b[sample(nrow(telcom.b), replace=FALSE),]

# set the maximum k
q <- 30

# set the number of folds
n <- 4

# create n equally-sized folds
folds <- cut(seq(1,nrow(telcom.s)), breaks=n, labels=FALSE)

# create an empty data frame for accuracy at varying degrees of k
acc.kfold <- rep(NA, n)
acc.list <- data.frame(k=seq(1,q,1), accuracy=rep(NA,q))

for(h in 1:q){
  for(i in 1:n){
    telcom.tr <- telcom.s[folds!=i,]  # set the training set
    telcom.ts <- telcom.s[folds==i,]   # set the test set
    
    # train kNN
    nn <- knn(train=telcom.tr[,1:15], test=telcom.ts[,1:15], cl=telcom.tr[,16], k=h, prob=TRUE)
    
    # accuracy
    acc.kfold[i] <- mean(nn == telcom.ts$Churn)
  }
  acc.list[h, 2] <- mean(acc.kfold)
}
acc.list

plot(acc.list[,2], type="b", xlab="k", ylab="accuracy")

which.max(acc.list[,2])
acc.list[,2][which.max(acc.list[,2])]

# read future data
telcom.f <- read.csv("churn_future.csv", header = TRUE)

# declare factor variables
telcom.f$Churn <- as.factor(telcom.f$Churn)
telcom.f$Churn <- as.numeric(telcom.f$Churn)-1

# normalize data (features only)
preproc <- preProcess(telcom.f[, c(1:15)], method=c("range"))
telcom.f[, c(1:15)] <- predict(preproc, newdata=telcom.f[, c(1:15)])

nn <- knn(train=telcom.b[,1:15], test=telcom.f[,1:15], cl=telcom.b[,16], k=which.max(acc.list[,2]), prob=TRUE)
mean(nn == telcom.f$Churn)






#########################################
##### Customer churn prediction: RF #####
#########################################
rm(list=ls())
telcom.b <- read.csv("churn_baseline.csv", header = TRUE)

# tabulate the outcome variable
table(telcom.b$Churn)

# declare factor variables
telcom.b$Churn <- as.factor(telcom.b$Churn)
#telcom.b$Churn <- as.numeric(telcom.b$Churn)-1

# build a forest with 500 trees
set.seed(999)
rf.init <- randomForest(formula = Churn ~ ., data = telcom.b, proximity=TRUE)
rf.init

# oob error rate over 500 trees
rf.init$err.rate[,1]

# plot oob error rates
plot(rf.init$err.rate[,1], cex=.3, xlab="Number of trees", ylab="OOB error rate")


# identify the forest size with the lowest OOB error
which.min( rf.init$err.rate[,1] )

# OOB error rate at the optimal forest size
rf.init$err.rate[,1][  which.min(rf.init$err.rate[,1]) ]

# re-train the forest at a tuned forest size
set.seed(999)
rf.optsize <- randomForest(formula = Churn ~ ., data = telcom.b, ntree = which.min( rf.init$err.rate[,1] ), proximity = TRUE)
rf.optsize


# tune no. of variables at split (forest size=optimal)
oob.values <- rep(NA, 4)
for(i in 1:4){
  set.seed(999)
  rf.temp <- randomForest(formula = Churn ~ ., data = telcom.b, mtry = i, ntree = which.min( rf.init$err.rate[,1] ), proximity = TRUE)
  oob.values[i] <- rf.temp$err.rate[,1][  which.min(rf.temp$err.rate[,1]) ]
}
oob.values
which.min(oob.values)

# re-train the forest at a tuned no. of variables at split
set.seed(999)
rf.tuned <- randomForest(formula = Churn ~ ., data = telcom.b, mtry = which.min(oob.values), ntree = which.min( rf.init$err.rate[,1] ), proximity=TRUE)
rf.tuned

# read future data
telcom.f <- read.csv("churn_future.csv", header = TRUE)

# generate predicted values
pred <- predict(rf.tuned, telcom.f)
tab.pred <- table(pred, telcom.f$Churn)
tab.pred

acc.test <- sum(diag(tab.pred)) / sum(tab.pred)
print(paste('Accuracy for test:', acc.test))

















