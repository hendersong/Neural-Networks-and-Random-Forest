
#data set to use
positive=read.csv('positive_train2.csv')
negative=read.csv('negative_train2.csv')

#load packages
library(caret)
library(nnet)
library(randomForest)
library(e1071)
library(parallel)

#set up data set so that data mining can be done on it
ppm=positive[,-1]
train=t(cbind(positive[,-1],negative[,-1]))
train=as.data.frame(train)
names(train)<- tolower(names(train))
train$label=0
train$label[1:94]=1

#randomize, so that you dont just get all postive or negative samples
random.order=sample(1:nrow(train),nrow(train))
train=train[random.order,]

#training set
x_train=train[1:800,1:1095]
y_train=train[1:800,1096]
#validation set
x_valid=train[801:1034,1:1095]
y_valid=train[801:1034,1096]

#create random forest from training data
rf <- randomForest(x=x_train,y=y_train,importance=TRUE,ntree=201,proximity=TRUE)
#predict using randomForest
rf_predict <- predict(rf,x_valid)
rf_predict=round(rf_predict)
#accuracy measurements
confusion_rf=confusionMatrix(rf_predict,y_valid)
save(rf, file="RF.rda")


#Find the most important variables
rf_importance=importance(rf,scale=FALSE)
#%IncMSE  IncNodePurity
a=rf_importance[order(-rf_importance[,"%IncMSE"]),]

#most important variables
impor_var=as.data.frame(t(a[1:150]))

#get only columns that match these important variables
names(x_train)<- tolower(names(x_train))
names(impor_var)<- tolower(names(impor_var))
nn_x_train=x_train[,match(names(impor_var), names(x_train))]

#neural network, using nnet and caret and RF to lower number of variables used
model_rf <- train(nn_x_train, y_train, method='nnet', linout=TRUE,MaxNWts=10000,maxit=100, trace = FALSE,
               #Grid of tuning parameters to try:
               tuneGrid=expand.grid(.size=c(1,5,10),.decay=c(.01,0.1,1)))
save(model_rf, file="nn with RF.rda")
#predict validation set with neural net
ps_rf <- predict(model_rf, x_valid)
ps_rf=round(ps_rf)
ps_rf[ps_rf>1]<-1

#if this doesn't work, it's because model isn't predicting one variable at all
table_nn_rf=table(ps_rf, y_valid)
con=confusionMatrix(ps_rf, y_valid)

#without RF
model <- train(x_train, y_train, method='nnet', MaxNWts=6000,linout=TRUE,maxit=100, trace = FALSE,
               tuneGrid=expand.grid(.size=c(1,5,10),.decay=c(0.1,0.25)))
save(model, file="nn without RF.rda")
#predict validation set with neural net
ps <- predict(model, x_valid)
ps=round(ps)
ps[ps>1]<-1
table_nn=table(ps,y_valid)
con_nn=confusionMatrix(ps, y_valid)