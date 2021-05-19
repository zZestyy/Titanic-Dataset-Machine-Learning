# Title: Project 1: R Component - Titanic Data Logistic Regression
# Author: "Val Wong - vmw170030"
# Date: 3/7/2021

# setwd("C:\\Users\\Val Wong\\Documents\\RStudio-Workspace\\Proj01")

# Load data and Load the csv file into RStudio
Titanic <- read.csv("titanic_project.csv") # read csv file
Titanic$survived <- factor(Titanic$survived)        # Levels 0 1

# Divide into train and test sets with 900 in train and the rest for test.
i <- 1:900
train <- Titanic[i, ] # 900 in train
test <- Titanic[-i, ] # 146 in test

start <- Sys.time() # Stopwatch start

# Train a logistic regression model on all data, survived ~ pclass.
glm1 <- glm(survived~pclass, data=train, family="binomial")

end <- Sys.time() # Stopwatch end

timer <- end - start # calculate run time
print(paste("Logistic Regression Algorithm Run Time: ", timer))

summary(glm1) # summary of glm1

# Print the coefficients of the model
glm1$coefficients # coefficients of train logistic model


# Test on the data
prob <- predict(glm1, newdata=test, type="response") # predict on test data
pred <- ifelse(prob>0.5, 1, 0) # make probs binary

library(caret) # library for confusion matrix
table <- confusionMatrix(factor(pred), factor(test$survived))
table

acc <- mean(pred==test$survived) # accuracy
sens <- table[["byClass"]][["Sensitivity"]]
spec <- table[["byClass"]][["Specificity"]]

print(paste("Accuracy =", acc*100,"%"))
print(paste("Sensitivity = ", sens))
print(paste("Specificity =", spec))
