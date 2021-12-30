install.packages('DEoptimR')
install.packages("caret", dependencies = TRUE)
library(caret)
install.packages('FNN')
library(FNN)
install.packages('binaryLogic')
library(binaryLogic)


##### kNN classification: MNIST example #####
setwd("C:/Users/Administrator/Desktop/myproject/R code/data")
mnist <- read.csv("mnist.csv", header=TRUE)
mnist <- mnist[1:12000,]

# normalize data (features only)
mnist[,2:ncol(mnist)] <- (mnist[,2:ncol(mnist)] / 255)

# randomly shuffle the data
set.seed(1)
mnist.s <- mnist[sample(nrow(mnist), replace=FALSE),]

# set the maximum k
q <- 20

# set the number of folds
n <- 4

# create n equally-sized folds
folds <- cut(seq(1,nrow(mnist.s)), breaks=n, labels=FALSE)

# create an empty data frame for accuracy at varying degrees of k
acc.kfold <- rep(NA, n)
acc.list <- data.frame(k=seq(1,q,1), accuracy=rep(NA,q))

for(h in 1:q){
  for(i in 1:n){
    mnist.tr <- mnist.s[folds!=i,]  # set the training set
    mnist.ts <- mnist.s[folds==i,]   # set the test set
    
    # train kNN
    nn <- knn(train=mnist.tr[,2:785], test=mnist.ts[,2:785], cl=mnist.tr[,1], k=h, prob=TRUE)
    
    # accuracy
    acc.kfold[i] <- mean(nn == mnist.ts$X5)
  }
  acc.list[h, 2] <- mean(acc.kfold)
}
acc.list

plot(acc.list[,2], type="b", xlab="k", ylab="accuracy")

which.max(acc.list[,2])
acc.list[,2][which.max(acc.list[,2])]

# new data
mnist <- read.csv("mnist.csv", header=TRUE)
mnist.unseen <- mnist[12001:13000,]

mnist.unseen[,2:ncol(mnist.unseen)] <- (mnist.unseen[,2:ncol(mnist.unseen)] / 255)

nn <- knn(train=mnist.s[,2:785], test=mnist.unseen[,2:785], cl=mnist.s[,1], k=which.max(acc.list[,2]), prob=TRUE)
mean(nn == mnist.unseen$X5)







### kNN gives better results with a larger sample ###
setwd("C:/Users/Tae-Young Pak/Desktop/Teaching/Spring 2021/SIC 5022/notes/week11_knn/")

mnist <- read.csv("mnist.csv", header=TRUE)
mnist.few <- mnist[1:50,]
#mnist.few <- mnist[1:100,]
#mnist.few <- mnist[1:500,]
#mnist.few <- mnist[1:1000,]
#mnist.few <- mnist[1:2000,]
#mnist.few <- mnist[1:5000,]

# normalize data (features only)
mnist.few[,2:ncol(mnist.few)] <- (mnist.few[,2:ncol(mnist.few)] / 255)

# randomly shuffle the data
set.seed(1)
mnist.s <- mnist.few[sample(nrow(mnist.few), replace=FALSE),]

# set aside one fifth of the sample as test set
split <- cut(seq(1,nrow(mnist.s)), breaks=5, labels=FALSE)
mnist.tr <- mnist.s[split!=5,]
mnist.ts <- mnist.s[split==5,]

# kNN at k=1
nn <- knn(train=mnist.tr[,2:785], test=mnist.ts[,2:785], cl=mnist.tr[,1], k=1, prob=TRUE)
accuracy <- mean(nn == mnist.ts$X5)
print(paste('Accuracy =', accuracy)) 







##### kNN vs. linear models #####

### kNN ###
setwd("C:/Users/Tae-Young Pak/Desktop/Teaching/Spring 2021/SIC 5022/notes/week11_knn/")

mnist <- read.csv("mnist.csv", header=TRUE)
mnist$X5[(mnist$X5 <= 4)|(mnist$X5 >= 7)] <- NA
mnist$X5 <- mnist$X5-5
mnist <- na.omit(mnist)

# normalize data (features only)
mnist[,2:ncol(mnist)] <- (mnist[,2:ncol(mnist)] / 255)

# randomly shuffle the data
set.seed(1)
mnist.s <- mnist[sample(nrow(mnist), replace=FALSE),]

# set aside one fifth of the sample as test set
split <- cut(seq(1,nrow(mnist.s)), breaks=5, labels=FALSE)
mnist.tr <- mnist.s[split!=5,]
mnist.ts <- mnist.s[split==5,]

# kNN at k=1
nn <- knn(train=mnist.tr[,2:785], test=mnist.ts[,2:785], cl=mnist.tr[,1], k=1, prob=TRUE)
accuracy <- mean(nn == mnist.ts$X5)
print(paste('Accuracy =', accuracy)) 


### logistic regression ###
glm.r <- glm(X5 ~ . , family="binomial", data=mnist.tr)
summary(glm.r)

mnist.ts$pprob <- predict(glm.r, type="response", newdata=mnist.ts)

mnist.ts$pclass <- ifelse(mnist.ts$pprob>0.5, 1, 0)

accuracy <- mean(mnist.ts$pclass == mnist.ts$X5)
accuracy


### lpm ###
lm.r <- lm(X5 ~ . , data=mnist.tr)
summary(lm.r)

mnist.ts$pprob <- predict(lm.r, newdata=mnist.ts)

mnist.ts$pclass <- ifelse(mnist.ts$pprob>0.5, 1, 0)

accuracy <- mean(mnist.ts$pclass == mnist.ts$X5)
accuracy






