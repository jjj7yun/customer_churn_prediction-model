install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("neuralnet")

library(tm)
library(SnowballC)
library(wordcloud)
library(neuralnet)
library(caret)


#######################
##### Text mining #####
#######################

# read the file, ask the R to treat text as text (stringAsFactors=FALSE)...
emails <- read.csv('emails.csv', stringsAsFactors = FALSE)

str(emails)

table(emails$spam)

# corpus: a collection of text doucments
# VectorSource(x): takes each element of the vector x as document
# VCorpus: treat the text data as "volatile" corpus 
# -> once the R object is destroyed, the whole corpus is gone
corpus <- VCorpus(VectorSource(emails$text))

# inspect the first three corpus
inspect(corpus[1:3])
# if you want to see the first entry in the corpus,
corpus[[1]]$content

# convert the text to lower case
corpus <- tm_map(corpus, content_transformer(tolower))
# remove all numbers; they may not be a good signal of spam...
corpus <- tm_map(corpus, removeNumbers)
# remove all punctuations
corpus <- tm_map(corpus, removePunctuation)
# multiple whitespace characters are collapsed to a single blank
corpus <- tm_map(corpus, stripWhitespace)
# remove all stopwords (e.g., the, a, an, is, I, my.. -> preposition, article, postposition, etc)
corpus <- tm_map(corpus, removeWords, c(stopwords("english"))) 

# print the list of stopwords in English
stopwords(kind = "english")

# text stemming
# e.g., for waiting, waited, waits, the stem is "wait"
corpus <- tm_map(corpus, stemDocument)
# take everything in corpus as plain text document
corpus <- tm_map(corpus, PlainTextDocument)

# convert corpus to document term matrix
dtm <- DocumentTermMatrix(corpus)
dtm

# sparsity: the proportion of zeros for each term
# remove sparse terms (that don't appear very often across documents)
# sparsity threshold: any term with sparsity of greater than 0.95 is removed
spdtm <- removeSparseTerms(dtm, 0.95)
spdtm
as.matrix(spdtm)

# store the term document matrix as data frame
emailsSparse <- as.data.frame(as.matrix(spdtm))
colnames(emailsSparse) <- make.names(colnames(emailsSparse))

# eyeball how many times each term shows up
sort(colSums(emailsSparse))

# add outcome label to this dataset
emailsSparse$spam <- emails$spam

# tabluate outcome
table(emailsSparse$spam)


# barplot for spams and hams
emails.spam <- emailsSparse[emailsSparse$spam==1,]
emails.spam$spam <- NULL
emails.ham <- emailsSparse[emailsSparse$spam==0,]

a <- sort(colSums(emails.spam), decreasing = TRUE)
a <- head(a,20)
barplot(a, las = 2, col ="lightblue", main ="Most frequent words (spams)",
        ylab = "Word frequencies")

b <- sort(colSums(emails.ham), decreasing = TRUE)
b <- head(b,20)
barplot(b, las = 2, col ="lightblue", main ="Most frequent words (hams)",
        ylab = "Word frequencies")


# wordcloud for spams
emails.spam <- emails[emails$spam==1,]
corpus.spam <- VCorpus(VectorSource(emails.spam$text))

corpus.spam <- tm_map(corpus.spam, content_transformer(tolower))
corpus.spam <- tm_map(corpus.spam, removeNumbers)
corpus.spam <- tm_map(corpus.spam, removePunctuation)
corpus.spam <- tm_map(corpus.spam, stripWhitespace)
corpus.spam <- tm_map(corpus.spam, removeWords, c(stopwords("english"))) 
corpus.spam <- tm_map(corpus.spam, stemDocument)
corpus.spam <- tm_map(corpus.spam, PlainTextDocument)

dev.new(width = 1000, height = 1000, unit = "px")
wordcloud(corpus.spam, min.freq=1, max.words=200, scale=c(4,.5), random.order=FALSE, rot.per=0.35, colors=brewer.pal(8,"Dark2"))


# wordcloud for hams
emails.ham <- emails[emails$spam==0,]
corpus.ham <- VCorpus(VectorSource(emails.ham$text))

corpus.ham <- tm_map(corpus.ham, content_transformer(tolower))
corpus.ham <- tm_map(corpus.ham, removeNumbers)
corpus.ham <- tm_map(corpus.ham, removePunctuation)
corpus.ham <- tm_map(corpus.ham, stripWhitespace)
corpus.ham <- tm_map(corpus.ham, removeWords, c(stopwords("english"))) 
corpus.ham <- tm_map(corpus.ham, stemDocument)
corpus.ham <- tm_map(corpus.ham, PlainTextDocument)

dev.new(width = 1000, height = 1000, unit = "px")
wordcloud(corpus.ham, min.freq=1, max.words=200, scale=c(4,.5), random.order=FALSE, rot.per=0.35, colors=brewer.pal(8,"Dark2"))


# randomly shuffle the data
set.seed(1)
email.s <- emailsSparse[sample(nrow(emailsSparse), replace=FALSE),]

# normalize data (features only)
preproc <- preProcess(email.s[,c(1,204)], method=c("range"))
email.s[,c(1,204)] <- predict(preproc, newdata=email.s[,c(1,204)])

train <- email.s[1:4000,]
test <- email.s[4001:5726,]


### nn with one hidden layer ###
set.seed(99999)
nn <- neuralnet(spam ~ ., data = train,
                err.fct ="sse", act.fct = "logistic", hidden = c(8), 
                algorithm = "rprop+", threshold = 0.001, learningrate = 0.05, linear.output = F)
plot(nn, rep="best")

pred <- predict(nn, test)

test$pclass <- ifelse(pred>0.5, 1, 0)
head(test$pclass)

conf <- table(test$pclass, test$spam)
conf

accu <- (conf[1,1] + conf[2,2]) / sum(conf)
accu


### nn with two hidden layers ###
nn <- neuralnet(spam ~ ., data = train,
                err.fct ="sse", act.fct = "logistic", hidden = c(14,8), 
                algorithm = "rprop+", threshold = 0.001, learningrate = 0.05, linear.output = F)
plot(nn, rep="best")

pred <- predict(nn, test)

test$pclass <- ifelse(pred>0.5, 1, 0)
head(test$pclass)

conf <- table(test$pclass, test$spam)
conf

accu <- (conf[1,1] + conf[2,2]) / sum(conf)
accu


















