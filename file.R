library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(caret)

## Importing the original data set
original_df <- read.csv("HW2.csv", header = TRUE)

# From data set, define Property as categorical : ie. Yes or No; we select 7.825 as it is in middle.
Property1=ifelse(original_df$Property<=7.825,"No","Yes")
modded_df=data.frame(original_df,Property1)

# Removing y column from predictor columns
modded_df = modded_df[,-1]

model_accuracy <- data.frame(matrix(ncol=2,nrow=0, dimnames=list(NULL, c("Technique_Used", "acuracy"))))


# Building the tree, predicting Property1
fit<-rpart(Property1~., modded_df)
rpart.plot(fit)
summary(fit)
plot(fit)
text(fit,pretty=1)

# Now we want to divide in training and test data.
set.seed(456)
train = sample(1:nrow(modded_df),80)
fit.modded_df<-rpart(Property1~., modded_df, subset=train)
plot(fit.modded_df)
text(fit.modded_df,pretty=0)

#Let's name the data differently to keep it easier to remember
data_test = modded_df[-train,]
data_train = modded_df[train,]

# Get a table for the test accuracy
tree.pred=predict(fit.modded_df,data_test,type="class")
table <- with(data_test,table(tree.pred, Property1))
accuracy_decisiontree <- (table[1] + table[4]) / (table[1] + table[2] + table[3] + table[4])
model_accuracy[nrow(model_accuracy) + 1,] = c('Decision Tree', accuracy_decisiontree*100)

accuracy_tune<-function(fit){
  predict_unseen<-predict(fit,data_test,type='class')
  table_mat<-table(data_test$Property1,predict_unseen)
  accuracy_Test<-sum(diag(table_mat))/sum(table_mat)
  accuracy_Test
}

# Changing the parameters (minsplit,minbucket,maxdepth,cp) and see how the model changes
control<-rpart.control(minsplit=4,minbucket=2,maxdepth=3,cp=0)
tune_fit<-rpart(Property1~.,data=data_train,method='class')
accuracy_tune(tune_fit)

#Plotting the accuracy
plot(fit, uniform=TRUE,main="Classification Tree for Property 1")
text(fit, use.n=TRUE, all=TRUE, cex=1)

#Pruning the tree as a function of cp (complexity parameter)
pfit<-prune(fit,cp=.1)
plot(pfit, uniform=TRUE,main="Pruned Tree")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)

#Bagging
cvcontrol<-trainControl(method="repeatedcv",number=10,allowParallel=TRUE)
train.bagg<-train(as.factor(Property1)~.,data=data_train,method="treebag",trControl=cvcontrol,importance=TRUE, nbagg = 40) #Can add term: nbagg=x to change the value of B in the bagging.  Default is 25.
train.bagg
plot(varImp(train.bagg))
model_accuracy[nrow(model_accuracy) + 1,] = c('Decision Tree w/ Bagging', train.bagg$results$Accuracy*100)

#Random Forest
train.rf <- train(as.factor(Property1) ~ ., data=data_train,method="rf",trControl=cvcontrol,importance=TRUE)
train.rf
model_accuracy[nrow(model_accuracy) + 1,] = c('Random Forest', train.rf$results$Accuracy[1]*100)

#Random Forest Boosting
train.gbm <- train(as.factor(Property1) ~ ., data=data_train,method="gbm",verbose=F,trControl=cvcontrol)
train.gbm
model_accuracy[nrow(model_accuracy) + 1,] = c('Random Forest Boosting', max(train.gbm$results$Accuracy)*100)

#Random Forest Regression, using sales values, instead of just category
#Would want to split into training and test data first
Property<-original_df$Property
descriptors<-original_df[,2:9]
model<-randomForest(Property~.,data=descriptors, mtry=3,importance=TRUE, na.action=na.omit)
property_pred<-predict(model,descriptors)

rf_sst <- sum((Property - mean(Property))^2)
rf_sse <- sum((property_pred - Property)^2)
rf_rsq <- 1 - rf_sse/rf_sst

plot(Property,property_pred)
print(model)
plot(model)
