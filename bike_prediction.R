##########################################################
##### Predicting the demand for Seoul Bike Ddareungi #####
##########################################################

seoul <- read.csv("SeoulBikeData.csv", header = TRUE)
head(seoul)

### create a separate dataframe for descriptive analysis ###
seoul.sm <- seoul[ , -c(1, 12:14)]

### descriptive statistics ###
install.packages("psych")
library(psych)
describe(seoul.sm)

### correlation matrix ###
round(cor(seoul.sm), 2)

### correlation plot ###
plot(seoul.sm, cex=.1)

### scatter plot ###
plot(seoul$hour, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="hour")
plot(seoul$temperature, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="temperature")
plot(seoul$humidity, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="humidity")
plot(seoul$wind.speed, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="wind.speed")
plot(seoul$visibility, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="visibility")
plot(seoul$dew.point.temp, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="dew.point.temp")
plot(seoul$solar.rad, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="solar.rad")
plot(seoul$rainfall, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="rainfall")
plot(seoul$snowfall, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="snowfall")

### histogram ###
hist(seoul$rented.bike.count, main=NULL, xlab="bike count")
hist(seoul$temperature, main=NULL, xlab="temperature")
hist(seoul$humidity, main=NULL, xlab="humidity")
hist(seoul$wind.speed, main=NULL, xlab="wind.speed")
hist(seoul$visibility, main=NULL, xlab="visibility")
hist(seoul$dew.point.temp, main=NULL, xlab="dew.point.temp")
hist(seoul$solar.rad, main=NULL, xlab="solar.rad")
hist(seoul$rainfall, main=NULL, xlab="rainfall")
hist(seoul$snowfall, main=NULL, xlab="snowfall")


### drop observations with missing or miscoded values ###
seoul$humidity[seoul$humidity==0] <- NA
seoul <- na.omit(seoul)

### declare factor variables ###
seoul$hour <- as.factor(seoul$hour)
seoul$seasons <- as.factor(seoul$seasons)
seoul$holiday <- as.factor(seoul$holiday)
seoul$functioning.day <- as.factor(seoul$functioning.day)


### identify model specification ###
summary(lm(rented.bike.count ~ temperature, data = seoul))
summary(lm(rented.bike.count ~ poly(temperature, 2, raw=T), data = seoul))
summary(lm(rented.bike.count ~ poly(temperature, 3, raw=T), data = seoul))

summary(lm(rented.bike.count ~ wind.speed, data = seoul))
summary(lm(rented.bike.count ~ log(wind.speed+.01), data = seoul))

summary(lm(rented.bike.count ~ rainfall, data = seoul))
summary(lm(rented.bike.count ~ log(rainfall+.01), data = seoul))

summary(lm(rented.bike.count ~ snowfall, data = seoul))
summary(lm(rented.bike.count ~ log(snowfall+.01), data = seoul))


### How does log-transformation of x increase model fit? ###
### let's look at the case of rainfall... ###
lmfit <- lm(rented.bike.count ~ rainfall, data = seoul)
summary(lmfit)
plot(seoul$rainfall, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="rainfall")
abline(lmfit)

lmfit <- lm(rented.bike.count ~ log(rainfall+.01), data = seoul)
summary(lmfit)
seoul$log_rainfall <- log(seoul$rainfall+.01)
plot(seoul$log_rainfall, seoul$rented.bike.count, cex=.1, ylab="rented.bike.count", xlab="log_rainfall")
abline(lmfit)


### model estimation ###
# initial model (only a few explanatory variables; not incorporating log transformation and poly structure)
lmfit <- lm(rented.bike.count ~ temperature, data = seoul)
summary(lmfit)

# initial model (only a few explanatory variables; not incorporating log transformation and poly structure)
lmfit <- lm(rented.bike.count ~ temperature + humidity + wind.speed + visibility, data = seoul)
summary(lmfit)

# initial model (not incorporating log transformation and poly structure)
lmfit <- lm(rented.bike.count ~ hour + temperature + humidity + wind.speed + visibility + dew.point.temp + solar.rad + rainfall + snowfall + seasons + holiday + functioning.day, data = seoul)
summary(lmfit)

# revised model (incorporating log-transformed variables and poly structure)
# feature selection at alpha_crit = 0.3
lmfit <- lm(rented.bike.count ~ hour + poly(temperature, 3, raw=T) + humidity + log(wind.speed+.01) + visibility + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = seoul)
summary(lmfit)

lmfit <- lm(rented.bike.count ~ hour + poly(temperature, 3, raw=T) + humidity + log(wind.speed+.01) + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = seoul)
summary(lmfit)

lmfit <- lm(rented.bike.count ~ hour + poly(temperature, 3, raw=T) + humidity + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = seoul)
summary(lmfit)


### box-cox transformation ###
lmfit <- lm((rented.bike.count+.01) ~ hour + poly(temperature, 3, raw=T) + humidity + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = seoul)
summary(lmfit)

library(MASS)
boxcox(lmfit,plotit=T)

lmfit <- lm(log(rented.bike.count+.01) ~ hour + poly(temperature, 3, raw=T) + humidity + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = seoul)
summary(lmfit)









###################################
##### k-fold cross validation #####
###################################
library(MASS)

# randomly shuffle the data
set.seed(1)
seoulf <- seoul[sample(nrow(seoul), replace=FALSE),]

# set the number of folds
k <- 5

# create k equally-sized folds
folds <- cut(seq(1,nrow(seoulf)), breaks=k, labels=FALSE)

# create an empty vector of mspe
mspe <- rep(NA, k)

for(i in 1:k){
  train <- seoulf[folds!=i,]  # set the training set
  test <- seoulf[folds==i,]   # set the test set
  
  # train a model on the training set
  newlm <- lm(rented.bike.count ~ temperature, data = train)
  #newlm <- lm(rented.bike.count ~ temperature + humidity + wind.speed + visibility, data = train)
  #newlm <- lm(rented.bike.count ~ hour + temperature + humidity + wind.speed + visibility + dew.point.temp + solar.rad + rainfall + snowfall + seasons + holiday + functioning.day, data = train)
  #newlm <- lm(rented.bike.count ~ hour + poly(temperature, 3, raw=T) + humidity + log(wind.speed+.01) + visibility + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = train)
  #newlm <- lm(log(rented.bike.count+.01) ~ hour + poly(temperature, 3, raw=T) + humidity + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = train)

  # get out of sample predicitons
  test$newpred <- predict(newlm, newdata=test)
  test$newpred <- exp(test$newpred)
  
  mspe[i] <- sum((test$rented.bike.count - test$newpred)^2) / nrow(test)
}

cve <- mean(mspe)
cve






#################
##### LOOCV #####
#################
library(MASS)

# create an empty vector of mspe
mspe <- rep(NA, nrow(seoul))

for(i in 1:nrow(seoul)){
  train <- seoul[-i,]  # set the training set
  test <- seoul[i,]    # set the test set

  # train a model on the training set  
  #newlm <- lm(rented.bike.count ~ temperature, data = train)
  newlm <- lm(rented.bike.count ~ temperature + humidity + wind.speed + visibility, data = train)
  #newlm <- lm(rented.bike.count ~ hour + temperature + humidity + wind.speed + visibility + dew.point.temp + solar.rad + rainfall + snowfall + seasons + holiday + functioning.day, data = train)
  #newlm <- lm(rented.bike.count ~ hour + poly(temperature, 3, raw=T) + humidity + log(wind.speed+.01) + visibility + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = train)
  #newlm <- lm(log(rented.bike.count+.01) ~ hour + poly(temperature, 3, raw=T) + humidity + dew.point.temp + solar.rad + log(rainfall+.01) + log(snowfall+.01) + seasons + holiday + functioning.day, data = train)
  
  # get out of sample predicitons
  test$newpred <- predict(newlm, newdata=test)
  #test$newpred <- exp(test$newpred)
  
  mspe[i] <- (test$rented.bike.count - test$newpred)^2
}
cve <- mean(mspe)
cve








##################################
##### overfitting experiment #####
##################################

# subset data into the first 60 observations
seoul <- seoul[1:60,]

# set the number of folds
k <- 5
# set complexity parameter
complx <- 20

# randomly shuffle the data
set.seed(1)
seoulf <- seoul[sample(nrow(seoul), replace=FALSE),]

# create k equally-sized folds
folds <- cut(seq(1,nrow(seoulf)), breaks=k, labels=FALSE)

# create an empty vector of mspe and cve
mspe <- rep(NA, k)
cve <- rep(NA, complx)

for(i in 1:complx){
  for(j in 1:k){
    train <- seoulf[folds!=j,]  # set the training set
    test <- seoulf[folds==j,]   # set the test set
    
    # train a model on the training set
    newlm <- lm(rented.bike.count ~ poly(temperature, i, raw=T), data = train)
  
    # get out of sample predicitons
    test$newpred <- predict(newlm, newdata=test)
    
    mspe[j] <- sum(abs(test$rented.bike.count - test$newpred)) / nrow(test)
  }
  cve[i] <- mean(mspe)
}
cve
min(cve)
which(cve==min(cve))

plot(cve, xlab="degree of polynomial", cex=.7)



























