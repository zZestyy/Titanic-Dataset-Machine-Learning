# Title: Project 1: R Component - Titanic Data Logistic Regression
# Author: "Val Wong - vmw170030"
# Date: 3/7/2021 

# setwd("C:\\Users\\Val Wong\\Documents\\RStudio-Workspace\\Proj01")

# Load data and Load the csv file into RStudio. 
Titanic <- read.csv("titanic_project.csv") # read csv file
Titanic$pclass <- factor(Titanic$pclass)            # Levels 1,2,3
Titanic$survived <- factor(Titanic$survived)        # Levels 0 1
Titanic$sex <- factor(Titanic$sex)                  # Levels 0 1

# Divide the data into train and test set
# Divide into train and test sets with 900 in train and the rest for test.
i <- 1:900
train <- Titanic[i, ] # 900 in train
test <- Titanic[-i, ] # 146 in test

# Train a naive Bayes model on the train data, survived~pclass+sex+age
library(e1071)
start <- Sys.time() # Stopwatch start

nb1 <- naiveBayes(survived ~ pclass+sex+age, data=train) # naive bayes model

end <- Sys.time() # Stopwatch end
end - start # calculate run time

nb1 # print the model


# Test on the test data
p1 <- predict(nb1, newdata=test, type="class") # predict with naive bayes
p1raw <- predict(nb1, newdata=test, type="raw") # predict with naive bayes
p1raw[1:5]

library(caret)
table <- confusionMatrix(p1, test$survived) # confusion matrix
table 

acc <- mean(p1==test$survived) # accuracy
sens <- table[["byClass"]][["Sensitivity"]]
spec <- table[["byClass"]][["Specificity"]]

print(paste("Accuracy =", acc*100,"%"))
print(paste("Sensitivity = ", sens))
print(paste("Specificity =", spec))


