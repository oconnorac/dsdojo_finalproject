# set working directory as titanic folder
setwd("~/Desktop/titanic")

# load in libraries
install.packages("randomForest")
install.packages("mice")
library(randomForest)
library(mice)

# read in train and test data
titanic.train <- read.csv('train.csv', stringsAsFactors = FALSE, header = TRUE)
titanic.test <- read.csv('test.csv', stringsAsFactors = FALSE, header = TRUE)

# Add IsTrainSet to merge the dataframes
titanic.train$IsTrainSet <- TRUE
titanic.test$IsTrainSet <- FALSE

# Add a Survived column to the test dataframe
titanic.test$Survived <- NA

# merge the two dataframes
titanic.full <- rbind(titanic.train, titanic.test)

# clean age, fare,and embarked
age.median <- median(titanic.full$Age, na.rm = TRUE)
titanic.full[is.na(titanic.full$Age), "Age"]<- age.median
#titanic.full[titanic.full$Embarked =='', "Embarked"] <- "S"
#fare.median <- median(titanic.full$Fare, na.rm =TRUE)
#titanic.full[is.na(titanic.full$Fare), "Fare"] <- fare.median

# determine upper whisker of box plot
upper.whisker <- boxplot.stats(titanic.full$Fare)$stats[5]
outlier.filter <- titanic.full$Fare < upper.whisker
titanic.full[outlier,filter,]

fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(
  formula = fare.equation,
  data = titanic.full[outlier.filter,]
)

fare.row <- titanic.full[
  is.na(titanic.full$Fare),
  c("Pclass","Sex","Age","SibSp","Parch","Embarked")
]

fare.predictions <- predict(fare.model, newdata = fare.row)
titanic.full[is.na(titanic.full$Fare),"Fare"] <- fare.predictions

# impute missing age values using MICE
# No iteration. But I want to get Predictor-Matrix
init = mice(titanic.full, maxit=0)
predM = init$predictorMatrix
# Do not use following columns to impute values in 'Age'. Use the rest.
predM[, c("PassengerId", "Name","Ticket","Cabin")]=0
imp<-mice(titanic.full, m=5, predictorMatrix = predM)
# Get the final data-frame with imputed values filled in 'Age'
titanic.full <- complete(imp)
View(titanic.full)


titanic.full$PassengerId <- as.factor(titanic.full$PassengerId)

# categorical casting
#make this ordinal
titanic.full$Pclass <-  as.ordered(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)

# split dataset back out into train and test
titanic.train <- titanic.full[titanic.full$IsTrainSet == TRUE,]
titanic.test <- titanic.full[titanic.full$IsTrainSet == FALSE,]
titanic.train$Survived <- as.factor(titanic.train$Survived)
survive.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + fancy_score + fancier_score"
survived.formula <- as.formula(survive.equation)

titanic.model <- randomForest(formula = survived.formula, data = titanic.train, ntree = 500, mtry = 3, nodesize = 0.01* nrow(titanic.test))

features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + fancy_score + fancier_score"

Survived <- predict(titanic.model, newdata = titanic.test)

PassengerId <- titanic.test$PassengerId

output.df <-as.data.frame(PassengerId)

output.df$Survived <- Survived

write.csv(output.df, file="aidan_entry.csv", row.names = FALSE)



## MODEL EVALUATION
## Predict test set outcomes, reporting class labels
titanic.model.predictions <- predict(titanic.model, titanic.test, type="response")
## calculate the confusion matrix
titanic.model.confusion <- table(titanic.model.predictions, titanic.test$Survived)
print(titanic.model.confusion)
## accuracy
titanic.model.accuracy <- sum(diag(titanic.model.confusion)) / sum(titanic.model.confusion)
print(titanic.model.accuracy)
## precision
titanic.model.precision <- titanic.model.confusion[2,2] / sum(titanic.model.confusion[2,])
print(titanic.model.precision)
## recall
titanic.model.recall <- titanic.model.confusion[2,2] / sum(titanic.model.confusion[,2])
print(titanic.model.recall)
## F1 score
titanic.model.F1 <- 2 * titanic.model.precision * titanic.model.recall / (titanic.model.precision + titanic.model.recall)
print(titanic.model.F1)
# We can also report probabilities
titanic.model.predictions.prob <- predict(titanic.model, titanic.test, type="prob")
print(head(titanic.model.predictions.prob))
print(head(titanic.test))

## show variable importance
importance(titanic.model)
varImpPlot(titanic.model)

write.csv(titanic.train, file="train.csv", row.names = FALSE)
write.csv(titanic.test, file="test.csv", row.names = FALSE)