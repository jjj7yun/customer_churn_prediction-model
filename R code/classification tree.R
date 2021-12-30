#install.packages("psych")
library(psych)
#install.packages("pastecs")
library(pastecs)
#install.packages("rpart.plot")
library(rpart)
library(rpart.plot)



##############################################
##### Classification tree: riding mowers #####
##############################################

mower <- read.csv("RidingMowers.csv", header=TRUE)
head(mower)

mower$Income <- ifelse(mower$Income>69, 1, 0)
mower$Lot_Size <- ifelse(mower$Lot_Size>19, 1, 0)

### randomly shuffle the data ###
set.seed(1)
mower.s <- mower[sample(nrow(mower), replace=FALSE),]

### build a full-sized tree ###
model.full <- rpart(Ownership ~ ., data = mower, method = 'class', cp = 0, minsplit = 1)

### plot tree ###
prp(model.full, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)

### print and plot cp values ###
printcp(model.full)
plotcp(model.full)

### build a pruned tree ###
model.prune <- prune(model.full, cp = model.full$cptable[which.min(model.full$cptable[,"xerror"]), "CP"])

### plot tree ###
prp(model.prune, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)

### write down rules ###
rpart.rules(model.prune)

### new data ###
mower.unseen <- data.frame(Income=1, Lot_Size=1)
pclass <- predict(model.prune, mower.unseen, type = 'class')
pclass



###########################################
##### Classification tree: loan offer #####
###########################################

bank <- read.csv("universalbank.csv", header=TRUE)
head(bank)

### drop unnecessary variables ###
bank <- bank[, -c(1, 5)]

### declare factor variables ###
bank$Education <- as.factor(bank$Education)

### randomly shuffle the data ###
set.seed(1)
bank.s <- bank[sample(nrow(bank), replace=FALSE),]

### build a full-sized tree ###
model.full <- rpart(Personal.Loan ~ ., data = bank.s, method = 'class', cp = 0, minsplit = 5)

### plot tree ###
prp(model.full, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)

### print and plot cp values ###
printcp(model.full)
plotcp(model.full)

### build a pruned tree ###
model.prune <- prune(model.full, cp = model.full$cptable[which.min(model.full$cptable[,"xerror"]), "CP"])

### plot tree ###
prp(model.prune, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)

### write down rules ###
rpart.rules(model.prune)

### new data ###
bank.unseen <- data.frame(Age=45, Experience=15, Income=119, Family=2, CCAvg=2.2, Education=3, 
                           Mortgage=200, Securities.Account=0, CD.Account=0, Online=1, CreditCard=1)
bank.unseen$Education <- as.factor(bank.unseen$Education)

pclass <- predict(model.prune, bank.unseen, type = 'class')
pclass



#######################################
##### Regression tree: loan offer #####
#######################################
rm(list = ls())

bank <- read.csv("universalbank.csv", header=TRUE)

# keep Age, Experience, Income, Family, CCAvg, Education; drop all others
bank <- subset(bank, select=c(Age, Experience, Income, Family, CCAvg, Education))
str(bank)

# declare factor variables
bank$Education <- as.factor(bank$Education)

### randomly shuffle the data ###
set.seed(1)
bank.s <- bank[sample(nrow(bank), replace=FALSE),]

### build a full-sized tree ###
model.full <- rpart(CCAvg ~ ., data = bank.s, method = 'anova', cp = 0, minsplit=20)

### plot tree ###
prp(model.full, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)

### print and plot cp values ###
printcp(model.full)
plotcp(model.full)

### build a pruned tree ###
model.prune <- prune(model.full, cp = model.full$cptable[which.min(model.full$cptable[,"xerror"]), "CP"])

### plot tree ###
prp(model.prune, type=1, extra=1, under=TRUE, split.font=1, varlen=-10)

### write down rules ###
rpart.rules(model.prune)

### new data ###
bank.unseen <- data.frame(Age=45, Experience=15, Income=119, Family=2, Education=3)
bank.unseen$Education <- as.factor(bank.unseen$Education)

pclass <- predict(model.prune, bank.unseen)
pclass

























