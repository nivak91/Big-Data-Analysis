R code:
  
# remove unknown values and duration
data <- subset(data, poutcome!="unknown")
data <- subset(data, age!="unknown")
data <- subset(data, job!="unknown")
data <- subset(data, marital!="unknown")
data <- subset(data, education!="unknown")
data <- subset(data, default!="unknown")
data <- subset(data, housing!="unknown")
data <- subset(data, loan!="unknown")
data <- subset(data, contact!="unknown")
data <- subset(data, month!="unknown")
data <- subset(data, day_of_week!="unknown")
data <- subset(data, campaign!="unknown")
data <- subset(data, pdays!="unknown")
data <- subset(data, previous!="unknown")
data <- subset(data, poutcome!="unknown")
data <- subset(data, emp.var.rate!="unknown")
data <- subset(data, cons.price.idx!="unknown")
data <- subset(data, euribor3m!="unknown")
data <- subset(data, nr.employed!="unknown")
data <- data[,-11]

#make numeric the catigorical values
library(dummies)
dataX <- data[,-20]
oneHotData <- dummy.data.frame(dataX, names=c("job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"))
oneHotData <- data.frame(oneHotData, data$y)

#trainData & validationData with oneHot
set.seed(60)
indoH <- sample(2,nrow(oneHotData), replace=TRUE, prob=c(0.75, 0.25))
trainDataoH <- oneHotData[indoH==1, ]
validationDataoH <- oneHotData[indoH==2, ]

##random forest classification
library(caret)
library(ROSE)
library(FSelector)
library(dummies)
set.seed(1234)
dataX <- data[,-20]
oneHotData <- dummy.data.frame(dataX, names=c("job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"))
oneHotData <- data.frame(oneHotData, data$y)
ind <- createDataPartition(oneHotData$data.y, p = 0.7, list = FALSE)
trainData <- oneHotData[ind, ]
testData <- oneHotData[-ind, ]
subset<-cfs(data.y ~., oneHotData)
sf <- as.simple.formula(subset, "y")
table(trainData$data.y)
data.rose <- ROSE(data.y~ ., data = trainData, seed=1)$data ## ROSE to make dataset balanced
table(data.rose$data.y)

trControlRf <- trainControl(method = "cv",
                           number = 10,
                           search = "grid")

#best mtry
set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(1: 20))
rf_mtry <- train(y ~ pdays, poutcomesuccess, emp.var.rate,
                 data = data.rose,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControlRf,
                 importance = TRUE,
                 ntree = 300)
print(rf_mtry)

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  set.seed(1234)
  rf_maxnode <- train(y ~pdays + poutcomesuccess +emp.var.rate,
                      data = data.rose,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

#random forest best ntrees
store_maxtrees <- list()
for (ntree in c(250, 300, 350, 400, 450, 500, 550, 600, 800, 1000, 2000)) {
  set.seed(5678)
  rf_maxtrees <- train(data.y~ pdays + poutcomesuccess +emp.var.rate.,
                       data = data.rose,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 14,
                       maxnodes = 15,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

rf_rose <- train(data.y ~ pdays + poutcomesuccess + emp.var.rate , data=data.rose, method = "rf", metric ="Accuracy", tuneGrid = tuneGrid, trControl = trControlRf, importance = TRUE, maxnodes = 15, ntree = 250)
pred_rf_rose <- predict(rf_rose, validationData)
cm_rf_rose <- confusionMatrix(validationData$y, pred_rf_rose)
cm_rf_rose<-as.table(cm_rf_rose)
accuracy_rf_rose <- Accuracy(pred_rf_rose, validationData$y)
precision_rf_rose <- precision(pred_rf_rose, validationData$y, "yes")
recall_rf_rose <- recall(pred_rf_rose, validationData$y, "yes")
F_Measure_rf_rose <- 2*cm_rf_rose[2, 2]/(2*cm_rf_rose[2, 2] + cm_rf_rose[1, 2] + cm_rf_rose[2, 1])

#sampling with smote
library(DMwR)
trainDatasmote <- SMOTE(data.y ~., data=trainDataoH)
                        
#decition tree
library(rpart)
library(rpart.plot)
library(caret)
model1 <- rpart(data.y ~ ., data = trainDataoH, method = "class", minsplit = 10, cp = 0.01)
rpart.plot(model1, extra = 104, nn=T)
predicted1 <-predict(model1, validationDataoH, type = "class")
cm1 <- confusionMatrix(predicted1, validationDataoH$data.y)
cm1
precision1 <- Precision(validationDataoH$data.y, predicted1)
recall1 <- Recall(validationDataoH$data.y, predicted1)
F_Measure1 <- F1_Score(validationDataoH$data.y, predicted1)
accuracy1 <- Accuracy(predicted1, validationDataoH$data.y)
accuracy1
precision1
recall1
F_Measure1

#CFS & Naive Bayes
library(FSelector)
library(caret)
subset <- cfs(data.y ~ ., trainDataoH)
f <- as.simple.formula(subset, "data.y")
print(f)
library(e1071)
model <- naiveBayes(data.y ~ ., data=trainDataoH, laplace = 1)
#simpler_model <- naiveBayes(f, data=trainDataoH, laplace = 1)
pred <- predict(model, validationDataoH)
#simpler_pred <- predict(simpler_model, validationDataoH)
library(MLmetrics)
#paste("Accuracy in training all attributes", Accuracy(train_pred, trainDataoH$y), sep=" - ")
validationDataoH$data.y <- factor(validationDataoH$data.y, levels = c("yes", "no"))
#simpler_pred <- factor(simpler_pred , levels=c("yes", "no"))
cm_cfs <- ConfusionMatrix (pred, validationDataoH$data.y)
cm_cfs
cm_cfs <- as.table(cm_cfs)
precision <- Precision(validationData$y, pred)
recall <- Recall(validationData$y, pred)
F_Measure <- F1_Score(validationData$y, pred)
accuracy <- Accuracy(pred, validationData$y)
accuracy
precision
recall
F_Measure

##hierarchical clustering
library(cluster)
library(scatterplot3d)
library(dummies)
library(FSelector)
dataX <- data[,-20]
oneHotData <- dummy.data.frame(dataX, names=c("job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"))
oneHotData <- data.frame(oneHotData, data$y)
set.seed(123)
subset<-cfs(data.y ~., oneHotData)
sf <- as.simple.formula(subset, "y")
oneHotDataCfs <- oneHotData[,c("euribor3m" , "pdays" , "poutcomesuccess" , "emp.var.rate")]
scaledData <- as.matrix(scale(oneHotDataCfs))
d <- dist(scaledData)
hc <- hclust(d, method = "complete")
slc = c()
for(i in 2:20){
  clusters <- cutree(hc, k=i)
  slc[i-1] <- mean(silhouette(clusters, d)[,3])
}
plot(2:20, slc, type = "b", xlab="Number of clusters", ylab = "Silhouette")
clusters <- cutree(2, method="complete")
m = table(clusters, data[,20])
precision <- m[2, 2] / (m[2,2] + m[2, 1])
recall <- m[2, 2] / (m[2, 2]+m[1, 2])
f = 2 * (precision * recall) / (precision + recall)
accuracy = (m[1,1]+m[2,2])/(m[1,2]+m[1,1]+m[2,1]+m[2,2])

#clustering with k-means and best k
dataX <- data[,-20]
library(dummies)
oneHotData <- dummy.data.frame(dataX, names=c("job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"))
oneHotData <- data.frame(oneHotData, data$y)
library(FSelector)
subset<-cfs(data.y ~., oneHotData)
sf <- as.simple.formula(subset, "y")
library(cluster)
oneHotDataCfs <- oneHotData[,c("euribor3m" , "pdays" , "poutcomesuccess" , "emp.var.rate")]
scaledData <- as.matrix(scale(oneHotDataCfs))
set.seed(123)
k.max <- 15
wss <- sapply(1:k.max, function(k){kmeans(scaledData, k, nstart=50,iter.max = 15)$tot.withinss})
plot(1:k.max, wss,
     +      type="b", pch = 19, frame = FALSE, 
     +      xlab="Number of clusters K",
     +      ylab="Total within-clusters sum of squares")
kmeansModel <- kmeans(scaledData, 3, nstart = 50, iter.max = 15)
cohesionkmeans <- kmeansMοdel$tot.withinss
seperationkmeans <- kmeansModel$betweenss
plot(scaledData, col = kmeansModel$cluster)
points(kmeansModel$centers, col = 4, pch = "+", cex = 2)
model_silhouette = silhouette(kmeansModel$cluster, dist(scaledData))
plot(model_silhouette)
mean_silhouette =mean(model_silhouette[, 3])
m = table(kmeansModel$cluster, data[,20])
precision <- m[2, 2] / (m[2,2] + m[2, 1])
recall <- m[2, 2] / (m[2, 2]+m[1, 2])
f = 2 * (precision * recall) / (precision + recall)
accuracy = (m[1,1]+m[2,2])/(m[1,2]+m[1,1]+m[2,1]+m[2,2])
# Compute mean silhouette
mean_silhouette =mean(model_silhouette[, 3])

#DBSCAN
install.packages("dbscan")
library(dbscan)
dataX <- data[,-20]
library(dummies)
oneHotData <- dummy.data.frame(dataX, names=c("job","marital","education","default","housing","loan","contact","month","day_of_week","poutcome"))
oneHotData <- data.frame(oneHotData, data$y)
library(FSelector)
subset<-cfs(data.y ~., oneHotData)
sf <- as.simple.formula(subset, "y")
library(cluster)
oneHotDataCfs <- oneHotData[,c("euribor3m" , "pdays" , "poutcomesuccess" , "emp.var.rate")]
kNN_dist <- kNNdist(scaledData, k=5000)
kdist <- kNN_dist[,5000]
plot(sort(kdist), type='l', xlab = "Points sorted by distance", ylab = "1000-NN distance")
abline(h = 0.5, lty = 2)
dbscanModel <- dbscan(scaledData, eps = 1.5, minPts = 5000)
plot(scaledData, col= dbscanModel$cluster+1, pch=ifelse(dbscanModel$cluster,1,4))