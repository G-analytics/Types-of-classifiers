#set working Directory


#Read Data from csv file
data <- read.csv("bankdata.csv", header = TRUE)
#View(data)
str(data)

#Find the missing values
sum(is.na(data))
colSums(is.na(data))
x<-manyNAs(data)
data<-data[-x,]

#Fill the missing values
library(DMwR)
data <- centralImputation(data)
sum(is.na(data))
e=2.71828
data$Attr29<-e^data$Attr29
data$target <- ifelse(data$target=="No", 0, 1)
data$target<-as.factor(data$target)

data1 <- subset(data,select=-c(65))


#Remove correlated values
install.packages("corrplot")
library("corrplot")
y<-cor(data1)
corrplot(y)

#set cutoff and remove those columns
library(caret)
hc = findCorrelation(cor(data1), cutoff=0.9, verbose = FALSE) # put any value as a "cutoff" 
hc = sort(hc)
data2 = data1[,-c(hc)]
summary(data2)

data2 <- scale(data2)
data2 <- data.frame(data2)

data3=cbind(data2,data$target)
colnames(data3)[38] <- "target"
str(data3)

#library(vegan)
#library(car)

###Divide data into train and test data
set.seed(123)
rows<-seq(1,nrow(data3),1)
trainrows<-sample(rows,(0.7*nrow(data3)))
train<-data3[trainrows,]
test<-data3[-trainrows,]
dim(train)
dim(test)

###Build different models on the data #######

##### 1. Random Forest #####
library(caret)
library(randomForest)
data_rf <- randomForest(target ~ ., data = train,keep.forest = TRUE,ntrees=100)

#View results and understand important attributes
print(data_rf)
data_rf$predicted
data_rf$importance

#View results and understand important attributes
#(directly prints the important attributes)
varImpPlot(data_rf)

#Predict the values using model on test data sets.
pred = predict(data_rf, test)
pred
#Calculate precision, recall and accuracy
result<- table(pred, test$target);result # 0(-ve) and 1(+ve)

confusionMatrix(train$target, data_rf$predicted)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction    No   Yes
# No  28460  1040
# Yes    82   379
# 
# Accuracy : 0.9626          
# 95% CI : (0.9603, 0.9647)
# No Information Rate : 0.9526          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.389           
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.9620          
#             Specificity : 0.6689          
#          Pos Pred Value : 0.9647          
#          Neg Pred Value : 0.8221          
#              Prevalence : 0.9526          
#          Detection Rate : 0.9499          
#    Detection Prevalence : 0.9846          
#       Balanced Accuracy : 0.6321          
#                                           
#        'Positive' Class : 0   



confusionMatrix(train$target, data_rf$predicted)
#Confusion Matrix and Statistics

# Reference
# Prediction  
#no    #yes
# No  28460  1040
# Yes    82   379
# 
# Accuracy : 0.9626          
# 95% CI : (0.9603, 0.9647)
# No Information Rate : 0.9526          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.389           
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Sensitivity : 0.9620          
# Specificity : 0.6689          
# Pos Pred Value : 0.9647          
# Neg Pred Value : 0.8221          
# Prevalence : 0.9526          
# Detection Rate : 0.9499          
# Detection Prevalence : 0.9846          
# Balanced Accuracy : 0.6321          

####################Scaling ########################
# Will be firm be "bankrupt" or not.

#1. STUDY DATA AND PRE-PROCESSING

#a. Check the input data structure
str(data3)

#b. Making the target variable as factor
# APPLYING SEVERAL MACHINE LEARNING CLASSFICATION TECHNIQUES 

# (1) CART
library(rpart)

# (1) Build rpart model on the training dataset

cart_obj <- rpart(target ~ ., train, method = "class")
summary(cart_obj)

# predicting on train dataset
cart_pred <- predict(cart_obj, newdata = train, type="vector") 
table(cart_pred)
# if we choose type=vector, then we will have to use the following ifelse
cart_pred <- ifelse(test = cart_pred == 1, 0, 1) #if 1 replace with 0, 
# else repl with 1
table(cart_pred)
check1 <- table(train$target, cart_pred)
confusionMatrix(check1)


# Confusion Matrix and Statistics
# 
# cart_pred
# 0     1
# 0 28517    25
# 1  1148   271
# 
# Accuracy : 0.9608         
# 95% CI : (0.9586, 0.963)
# No Information Rate : 0.9901         
# P-Value [Acc > NIR] : 1              
# 
# Kappa : 0.3047         
# Mcnemar's Test P-Value : <2e-16         
# 
# Sensitivity : 0.9613         
# Specificity : 0.9155         
# Pos Pred Value : 0.9991         
# Neg Pred Value : 0.1910         
# Prevalence : 0.9901         
# Detection Rate : 0.9518         
# Detection Prevalence : 0.9526         
# Balanced Accuracy : 0.9384         
# 
# 'Positive' Class : 0 
# prediction on test dataset
cart_test <- predict(cart_obj, newdata = test, type="vector")
cart_test <- ifelse(test = cart_test == 1, 0, 1)
table(cart_test)
check1 <- table(test$target, cart_test)
confusionMatrix(check1)
# Confusion Matrix and Statistics
# 
# cart_test
# 0     1
# 0 12179    15
# 1   535   112
# 
# Accuracy : 0.9572          
# 95% CI : (0.9535, 0.9606)
# No Information Rate : 0.9901          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2775          
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.9579          
#             Specificity : 0.8819          
#          Pos Pred Value : 0.9988          
#          Neg Pred Value : 0.1731          
#              Prevalence : 0.9901          
#          Detection Rate : 0.9484          
#    Detection Prevalence : 0.9496          
#       Balanced Accuracy : 0.9199          
#                                           
#        'Positive' Class : 0               
#                                           

###### (2) Build C5.0 model on the training dataset
library(C50)
dtC50_obj <- C5.0(target ~., train, rules = T)
summary(dtC50_obj)

# predicting with the train dataset
dtC50_pred <- predict(dtC50_obj, train, type = "class")
dtC50_pred <- as.vector(dtC50_pred)
table(dtC50_pred)
confusionMatrix(train$target,dtC50_pred)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 28518    24
# 1   983   436
# 
# Accuracy : 0.9664          
# 95% CI : (0.9643, 0.9684)
# No Information Rate : 0.9846          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.4514          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.9667          
# Specificity : 0.9478          
# Pos Pred Value : 0.9992          
# Neg Pred Value : 0.3073          
# Prevalence : 0.9846          
# Detection Rate : 0.9518          
# Detection Prevalence : 0.9526          
# Balanced Accuracy : 0.9573          
# 
# 'Positive' Class : 0


# prediction on test dataset 
dtC50_test <- predict(dtC50_obj, newdata = test, type = "class")
dtC50_test <- as.vector(dtC50_test)
table(dtC50_test)
check2 <- table(test$target, dtC50_test)
confusionMatrix(check2)
# Confusion Matrix and Statistics
# 
# dtC50_test
# 0     1
# 0 12170    24
# 1   473   174
# 
# Accuracy : 0.9613          
# 95% CI : (0.9578, 0.9646)
# No Information Rate : 0.9846          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3976          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.9626          
# Specificity : 0.8788          
# Pos Pred Value : 0.9980          
# Neg Pred Value : 0.2689          
# Prevalence : 0.9846          
# Detection Rate : 0.9477          
# Detection Prevalence : 0.9496          
# Balanced Accuracy : 0.9207          
# 
# 'Positive' Class : 0               


# # (3) Build KNN on the training dataset
# library(class)
# #Building the model on train data 
# #Converting Education 
# 
# train_knnwithoutclass = subset(train,select=-c(target))
# test_knnwithoutclass = subset(test,select=-c(target))
# 
# knn_pred=knn(train_knnwithoutclass, train_knnwithoutclass ,train$target, k = 5)
# knn_test=knn(train_knnwithoutclass, test_knnwithoutclass ,train$target, k = 5)
# 
# check3<- table(test$target, knn_test)
# confusionMatrix(check3)
# 
# accu_knn<-sum(diag(check3))/sum(check3)

#Perform Logistic Regression
data1_log<-glm(formula = target~.,data = train,family = "binomial")
summary(data1_log)

#Predicting on train data
pred<-predict(data1_log,type = "response")

#predicitng on test data
pred1<-predict(data1_log,newdata=test,type = "response")
#Manually choose the threshold; Here, we take it as 0.5
pred_class<-ifelse(pred>0.50,1,0)
tab<-table(train$target,pred_class)
confusionMatrix(tab)
# Confusion Matrix and Statistics
# 
# pred_class
# 0     1
# 0 28531    11
# 1  1415     4
# 
# Accuracy : 0.9524          
# 95% CI : (0.9499, 0.9548)
# No Information Rate : 0.9995          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.0046          
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.952748        
#             Specificity : 0.266667        
#          Pos Pred Value : 0.999615        
#          Neg Pred Value : 0.002819        
#              Prevalence : 0.999499        
#          Detection Rate : 0.952271        
#    Detection Prevalence : 0.952638        
#       Balanced Accuracy : 0.609707        
#                                           
#        'Positive' Class : 0    

preds_test<-ifelse(pred1>0.50,1,0)
tab1<-table(test$target,preds_test)
confusionMatrix(tab1)
# Confusion Matrix and Statistics
# 
# pred_class
# 0     1
# 0 28531    11
# 1  1415     4
# 
# Accuracy : 0.9524          
# 95% CI : (0.9499, 0.9548)
# No Information Rate : 0.9995          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.0046          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.952748        
# Specificity : 0.266667        
# Pos Pred Value : 0.999615        
# Neg Pred Value : 0.002819        
# Prevalence : 0.999499        
# Detection Rate : 0.952271        
# Detection Prevalence : 0.952638        
# Balanced Accuracy : 0.609707        
# 
# 'Positive' Class : 0               

confusionMatrix(tab1)
# Confusion Matrix and Statistics
# 
# preds_test
# 0     1
# 0 12185     9
# 1   642     5
# 
# Accuracy : 0.9493         
# 95% CI : (0.945,0.953)
# No Information Rate : 0.9989         
# P-Value [Acc > NIR] : 1              
# 
# Kappa : 0.013          
# Mcnemar's Test P-Value : <2e-16         
#                                          
#             Sensitivity : 0.949949       
#             Specificity : 0.357143       
#          Pos Pred Value : 0.999262       
#          Neg Pred Value : 0.007728       
#              Prevalence : 0.998910       
#          Detection Rate : 0.948914       
#    Detection Prevalence : 0.949615       
#       Balanced Accuracy : 0.653546       
#                                          
#        'Positive' Class : 0     

#######################Stacking############################
#Combining training predictions of CART, C5.0 & KNN together
train_pred_all_models <- data.frame(cart_pred, dtC50_pred, pred_class)
train_pred_all_models<-data.frame(apply(train_pred_all_models,2,as.numeric))
str(train_pred_all_models)
train_pred_all_models <- data.frame(apply(train_pred_all_models, 2, function(x) {as.factor(x)}))
str(train_pred_all_models)
summary(train_pred_all_models)

# (5) Viewing the predictions of each model
table(train_pred_all_models$cart_pred) #CART 
table(train_pred_all_models$dtC50_pred) #C5.0
#table(train_pred_all_models$knn_pred) #KNN
table(train_pred_all_models$log_pred) #Logreg
table(train$target) #Original Dataset DV

# (6) Adding the original DV to the dataframe
train_pred_all_models <- data.frame(train_pred_all_models, train$target)
names(train_pred_all_models)[3] = "Actual"

# (7) Ensemble Model using mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

train_pred_all_models$target<-apply(train_pred_all_models[1:3],1,Mode)
Tab <- table(train_pred_all_models$target,train_pred_all_models$Actual)
confusionMatrix(Tab)

# Confusion Matrix and Statistics
# 
# 
# 0     1
# 0 29671    14
# 1   275     1
# 
# Accuracy : 0.9904          
# 95% CI : (0.9892, 0.9914)
# No Information Rate : 0.9995          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.0059          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.990817        
# Specificity : 0.066667        
# Pos Pred Value : 0.999528        
# Neg Pred Value : 0.003623        
# Prevalence : 0.999499        
# Detection Rate : 0.990321        
# Detection Prevalence : 0.990788        
# Balanced Accuracy : 0.528742        
# 
# 'Positive' Class : 0  

#Combining test predictions of CART, C5.0 & log_reg together 
test_pred_all_models <- data.frame(cart_test, dtC50_test,preds_test)
test_pred_all_models<-data.frame(apply(test_pred_all_models,2,as.numeric))
test_pred_all_models <- data.frame(apply(test_pred_all_models, 2, function(x) {as.factor(x)}))

str(test_pred_all_models)
head(test_pred_all_models)
# (11) Change column names 
colnames(test_pred_all_models)[1:3] <- c("cart_pred", "dtC50_pred","log_pred")
test_pred_all_models <-as.data.frame(test_pred_all_models)

test_pred_all_models$target<-apply(test_pred_all_models,1,Mode)
Tab1 <- table(test$target,test_pred_all_models$target)
confusionMatrix(Tab1)
# Confusion Matrix and Statistics
# 
# 
# 0     1
# 0 12188     6
# 1   535   112
# 
# Accuracy : 0.9579          
# 95% CI : (0.9543, 0.9613)
# No Information Rate : 0.9908          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2816          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.9580          
# Specificity : 0.9492          
# Pos Pred Value : 0.9995          
# Neg Pred Value : 0.1731          
# Prevalence : 0.9908          
# Detection Rate : 0.9491          
# Detection Prevalence : 0.9496          
# Balanced Accuracy : 0.9536          
# 
# 'Positive' Class : 0              
#                                   

######################SMOTE#######################################
table(train$target)
table(test$target)

# #Training a model on the data ----
# library(e1071)
# library(DMwR)
# train1 <- SMOTE(target ~ ., train, perc.over = 100, perc.under = 200)
# table(train1$target)
# 
# #Training a model on the data ----
# library(e1071)
# # classify the data
# fit <- naiveBayes(train1[,-38], train1$target)
# pred <- predict(fit, test)
# conf<-table(test$target, pred)
# confusionMatrix(conf)

##################XGBOOST#################################
install.packages("mlbench")
install.packages("xgboost")
library(mlbench)
library(xgboost)
library(caret)

set.seed(10)
indices <- createDataPartition(data3$target, p = .70, list = F)
trainingData <- data3[indices,]
testData <- data3[-indices,]

transformation <- preProcess(trainingData,method = c("range"))
trainingData <- predict(transformation, trainingData)
testData <- predict(transformation, testData)


xgb.ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 3,
                         search='random',
                         allowParallel=T)

set.seed(20)
xgb.tune <-train(target~.,
                 data = trainingData,
                 method="xgbTree",
                 trControl=xgb.ctrl,
                 # tuneGrid=xgb.grid,
                 tuneLength=20,
                 verbose=T,
                 metric="Accuracy",
                 nthread=3)

xgb.tune
View(xgb.tune$results)



############
a <- xgb.tune$results[order(xgb.tune$results$Accuracy, decreasing = TRUE),]
View(a)
par(mfrow=c(2,1))
hist(a$nrounds[1:10])
hist(a$nrounds[40:50])
############

plot(xgb.tune)

preds <- predict(xgb.tune, trainingData)
confusionMatrix(trainingData$target, preds)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 28498    18
# 1   544   903
# 
# Accuracy : 0.9812          
# 95% CI : (0.9796, 0.9827)
# No Information Rate : 0.9693          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.7534          
# Mcnemar's Test P-Value : < 2.2e-16       
# 
# Sensitivity : 0.9813          
# Specificity : 0.9805          
# Pos Pred Value : 0.9994          
# Neg Pred Value : 0.6240          
# Prevalence : 0.9693          
# Detection Rate : 0.9511          
# Detection Prevalence : 0.9517          
# Balanced Accuracy : 0.9809          
# 
# 'Positive' Class : 0    


preds <- predict(xgb.tune, testData)
confusionMatrix(testData$target, preds)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 12148    72
# 1   397   222
# 
# Accuracy : 0.9635          
# 95% CI : (0.9601, 0.9666)
# No Information Rate : 0.9771          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.4698          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.9684          
# Specificity : 0.7551          
# Pos Pred Value : 0.9941          
# Neg Pred Value : 0.3586          
# Prevalence : 0.9771          
# Detection Rate : 0.9462          
# Detection Prevalence : 0.9518          
# Balanced Accuracy : 0.8617          
# 
# 'Positive' Class : 0  

##################ADABOOST##########################
# build the classification model using Adaboost
library(ada)
x = subset(train, select = -target) 
y = as.factor(train$target) 
a = subset(test, select = -target) 
b = as.factor(test$target) 

model = ada(x, y, iter=20, loss="logistic") # 20 Iterations 
model

# predict the values using model on test data sets. 
pred1 = predict(model, x)
confusionMatrix(y,pred1)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 28524    18
# 1  1046   373
# 
# Accuracy : 0.9645          
# 95% CI : (0.9623, 0.9666)
# No Information Rate : 0.9869          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3999          
# Mcnemar's Test P-Value : <2e-16          
# 
# Sensitivity : 0.9646          
# Specificity : 0.9540          
# Pos Pred Value : 0.9994          
# Neg Pred Value : 0.2629          
# Prevalence : 0.9869          
# Detection Rate : 0.9520          
# Detection Prevalence : 0.9526          
# Balanced Accuracy : 0.9593          
# 
# 'Positive' Class : 0    


# predict the values using model on test data sets. 
pred = predict(model, a)
confusionMatrix(b,pred)
# Confusion Matrix and Statistics
# 
# Reference
# Prediction     0     1
# 0 12181    13
# 1   499   148
# 
# Accuracy : 0.9601          
# 95% CI : (0.9566, 0.9634)
# No Information Rate : 0.9875          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.3534          
# Mcnemar's Test P-Value : <2e-16          
#                                           
#             Sensitivity : 0.9606          
#             Specificity : 0.9193          
#          Pos Pred Value : 0.9989          
#          Neg Pred Value : 0.2287          
#              Prevalence : 0.9875          
#          Detection Rate : 0.9486          
#    Detection Prevalence : 0.9496          
#       Balanced Accuracy : 0.9400          
#                                           
#        'Positive' Class : 0 

# calculate accuracy
