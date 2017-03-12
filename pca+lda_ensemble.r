#Leaf Classification

# Loading libraries
library(e1071)
library(MASS)
library(randomForest)

# Importing dataset from "https://www.kaggle.com/c/leaf-classification/data"
train=read.csv("../input/train.csv")
test=read.csv("../input/test.csv")
dim(train);dim(test)
head(train)
comb_data=rbind(train[,-2],test)

# Dimensionality reduction by PCA
prin_comp=prcomp(train[,3:194])
std_dev=prin_comp$sdev
pr_var=std_dev^2
prop_varex=pr_var/sum(pr_var)
plot(cumsum(prop_varex), xlab = "Principal Component",ylab = "Cumulative Proportion of Variance Explained",type = "b")

# Rebuilding datasets as components for features
train.data=data.frame(id=train$id,species=train$species,predict(prin_comp,train[,3:194]))
test.data=data.frame(id=test$id,predict(prin_comp,test[,-1]))
head(test.data)

# lda model with pca
obj1=suppressWarnings(lda(species~.,data=train.data[,2:194],tol=1.0e-07))    
prediction1=suppressWarnings(predict(obj1,newdata=test.data[,2:193]))

# lda model w/o pca
obj2=suppressWarnings(lda(species~.,data=train[,2:194],tol=1.0e-07))    
prediction2=suppressWarnings(predict(obj2,newdata=test[,2:193]))

# randomForest model with pca
obj3=randomForest(species~.,data=train[,2:194],importance=T,ntree=1000,mtry=100)
prediction3=suppressWarnings(predict(obj3,newdata=test[,2:193],type='prob'))

# Ensemble of above 3 models
finalpred=(data.frame(prediction1$posterior)+data.frame(prediction2$posterior)+data.frame(prediction3))/3
submit=data.frame(id=test$id,data.frame(finalpred))

# Output manipulation
for (i in 1:594){
    submit[i,2:99]=(submit[i,2:99])^4
    submit[i,2:99]=submit[i,2:99]/sum(submit[i,2:99])
}
head(submit)
write.csv(submit,"submit2.csv",row.names=F)
