#' ---
#' title: "R을 활용한 의생명 데이터 분석 - Part 2. R을 활용한 의생명 기계학습 분석"
#' date: '`r format(Sys.time(), "%A %B %d %Y")`'
#' output:
#'   pdf_document:
#'     keep_tex: yes
#'     number_sections: yes
#'     toc: yes
#'     toc_depth: 3
#'   html_document:
#'     toc: yes
#' header-includes: \usepackage{kotex}
#' ---
#' 
## ---- echo = FALSE, warning = FALSE--------------------------------------
library(knitr)

#' 
#' # Simple Linear Regression with lm
#' 
## ------------------------------------------------------------------------
# install.packages("car")
library(car)
data(Quartet)
str(Quartet)
head(Quartet)

# Linear regression
plot(Quartet$x, Quartet$y1)
lmfit = lm(Quartet$y1~Quartet$x)
abline(lmfit, col="red")   
lmfit

# Least squares fit
plot(Quartet$x, Quartet$y1)
lmfit2 = lsfit(Quartet$x,Quartet$y1)
abline(lmfit2, col="red")  

#' 
#' ## Summarizing Linear Model Fits
#' 
## ------------------------------------------------------------------------
summary(lmfit)

#' ## See also
## ------------------------------------------------------------------------
coefficients(lmfit)
confint(lmfit, level=0.95) # Confidence interval
fitted(lmfit) # Extract model fitted values
residuals(lmfit) # Extract model residuals
anova(lmfit) # Compute analysis of variance tables for fitted model object
vcov(lmfit) # Calculate variance-covariance matrix for a fitted model object
influence(lmfit) # Diagnose quality of regression fits

#' 
#' ## Using Linear Regression to Predict Unknown Values
#' 
## ------------------------------------------------------------------------
plot(Quartet$x, Quartet$y2)
lmfit <- lm(y2 ~ x, Quartet)

newdata = data.frame(x = c(3,6,15))

newdata
predict(lmfit, newdata, interval="confidence", level=0.95)
predict(lmfit, newdata, interval="predict")



#' 
## ------------------------------------------------------------------------
lmfit <- lm(y2 ~ x, Quartet)
par(mfrow=c(2,2))
plot(lmfit)

#' 
#' # Fitting a polynomial regression model with lm
#' 
## ------------------------------------------------------------------------
lmfit = lm(Quartet$y2~ I(Quartet$x)+I(Quartet$x^2))
lmfit = lm(Quartet$y2~poly(Quartet$x,2))
plot(Quartet$x, Quartet$y2)
lines(sort(Quartet$x), lmfit$fit[order(Quartet$x)], col = "red")
lmfit = lm(Quartet$y2~ I(Quartet$x)+I(Quartet$x^2))

#' 
#' 
#' 
#' # Generalized Addictive Model (GAM)
#' Fit Generalized Addictive Model to Data
## ------------------------------------------------------------------------
#install.packages("mgcv")
library(mgcv)

#install.packages("MASS")
library(MASS)
attach(Boston)
str(Boston)

fit <- gam(dis ~ s(nox))
summary(fit)
#? bam


#' Visualize Generalized Addictive Model
## ------------------------------------------------------------------------
plot(nox, dis)
x <- seq(0, 1, length = 500)
y <- predict(fit, data.frame(nox = x))
lines(x, y, col = "red", lwd = 2)
plot(fit)
fit2=gam(medv~crim+zn+crim:zn, data=Boston)
vis.gam(fit2)

#' 
#' ##Diagnostic of Generalized Addictive Model 
#' 
## ------------------------------------------------------------------------
gam.check(fit)
#?gam.check
#?choose.k


#' ## TCGA_BRCA_CNV_processed.txt, TCGA_BRCA_Expr_processed.txt 파일을 읽어들이고 head함수를 이용해서 읽어들인 데이터의 첫 5행과 5열을 출력하세요.


#' 
#' ## ERBB2 유전자의 expression과 copy number를 각각 x축과 y축으로하는 scatter plot을 그리고 linear model을 만들어 추세선을 그리세요.


#' 
#' ## linear model의 summary를 출력하세요.


#' 
#' ## plot 함수를 이용하여 모델을 평가하는 plot 4개를 그리세요.


#' 
#' ## polynomial regression을 이용해서 model을 만들고 평가 plot을 그리세요


#' 
#' ## Generalized additive model을 만들고 summary를 출력하세요.


#' 
#' ## 아래와 같이 모델의 추세곡선을 그리세요.


#' 
#' ## 모델의 평가 plot을 그리세요.


#' 
#' # Classification
#' ## Preparing the training and testing datasets
## ------------------------------------------------------------------------
#install.packages("C50")
#library(C50)
load("../data/churn.RData")
#data(churn)
str(churnTrain)
churnTrain = churnTrain[,! names(churnTrain) %in% c("state", "area_code", "account_length") ]
set.seed(2)
ind <- sample(2, nrow(churnTrain), replace = TRUE, prob=c(0.7, 0.3))
trainset = churnTrain[ind == 1,]
testset = churnTrain[ind == 2,]
dim(trainset)
dim(testset)
split.data = function(data, p = 0.7, s = 666){
    set.seed(s)
    index = sample(1:dim(data)[1])
    train = data[index[1:floor(dim(data)[1] * p)], ]
    test = data[index[((ceiling(dim(data)[1] * p)) + 1):dim(data)[1]], ]
    return(list(train = train, test = test))
} 

#' 
#' ## Recursive partitioning trees
## ------------------------------------------------------------------------
library(rpart)
churn.rp <- rpart(churn ~ ., data=trainset)
churn.rp 
printcp(churn.rp)
plotcp(churn.rp)
#summary(churn.rp)

#' ```
#' ?rpart 
#' ?printcp
#' ?summary.rpart
## ------------------------------------------------------------------------
plot(churn.rp, margin= 0.1)
text(churn.rp, all=TRUE, use.n = TRUE)
plot(churn.rp, uniform=TRUE, branch=0.6, margin=0.1)
text(churn.rp, all=TRUE, use.n = TRUE)
predictions <- predict(churn.rp, testset, type="class")
table(testset$churn, predictions)

#' 
## ------------------------------------------------------------------------
#install.packages("caret")
library(caret)
confusionMatrix(table(predictions, testset$churn))
min(churn.rp$cptable[,"xerror"])
which.min(churn.rp$cptable[,"xerror"])
churn.cp = churn.rp$cptable[7,"CP"]
churn.cp

prune.tree <- prune(churn.rp, cp= churn.cp)
plot(prune.tree, margin= 0.1)
text(prune.tree, all=TRUE , use.n=TRUE)
predictions <- predict(prune.tree, testset, type="class")
table(testset$churn, predictions)
confusionMatrix(table(predictions, testset$churn))


#' 
#' 
#' 
#' 
#' 
#' ## Classifying data with logistic regression
## ------------------------------------------------------------------------
fit <- glm(churn ~ ., data = trainset, family=binomial)
summary(fit)
pred = predict(fit,testset, type="response")
Class = pred >.5
summary(Class)
tb = table(testset$churn,Class)
tb
churn.mod = ifelse(testset$churn == "yes", 1, 0)
pred_class = churn.mod
pred_class[pred<=.5] = 1- pred_class[pred<=.5]
ctb = table(churn.mod, pred_class)
ctb
confusionMatrix(ctb)

#' 
#' 
#' 
#' ## Training neural network with neuralnet
## ------------------------------------------------------------------------
 data(iris)
 ind <- sample(2, nrow(iris), replace = TRUE, prob=c(0.7, 0.3))
 trainset = iris[ind == 1,]
 testset = iris[ind == 2,]
head(trainset)

 #install.packages("neuralnet")
 library(neuralnet)
 trainset$setosa = trainset$Species == "setosa"
 trainset$virginica = trainset$Species == "virginica"
 trainset$versicolor = trainset$Species == "versicolor"
 head(trainset)
 network = neuralnet(versicolor + virginica + setosa~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, trainset, hidden=3)
 #network


#' 
#' ##Visualizing neural network trained by neuralnet
## ------------------------------------------------------------------------
 plot(network)
 par(mfrow=c(2,2))
 gwplot(network,selected.covariate="Petal.Width")
 gwplot(network,selected.covariate="Sepal.Width")
 gwplot(network,selected.covariate="Petal.Length")
 gwplot(network,selected.covariate="Petal.Width")
#?gwplot

#' 
#' ##Predicting labels based upon a model trained by neuralnet
## ------------------------------------------------------------------------
 predict = compute(network, testset[-5])$net.result
 prediction = c("versicolor", "virginica", "setosa")[apply(predict, 1, which.max)]
 predict.table = table(testset$Species, prediction)
 predict.table
 confusionMatrix(predict.table)
 #compute(network, testset[-5])

#' 
#' 
#' 
#' 
#' 
#' # Unsupervised learning
#' 
#' ## Clustering Data With Hierarchical Clustering
## ------------------------------------------------------------------------

customer= read.csv('../data/customer.csv', header=TRUE)
head(customer)
str(customer)
customer = scale(customer[,-1])
hc = hclust(dist(customer, method="euclidean"), method="ward.D2")
hc
plot(hc, hang = -0.01, cex = 0.7)
hc2 = hclust(dist(customer), method="single")
plot(hc2, hang = -0.01, cex = 0.7)
#? diste  
#? hclust
#install.packages("dendextend")
#install.packages("margrittr")
library(dendextend)
library(magrittr)
dend = customer %>% dist %>% hclust %>% as.dendrogram
 dend %>% plot(horiz=TRUE, main = "Horizontal Dendrogram")

#' 
#' ## Cutting Tree Into Clusters
## ---- cache = TRUE-------------------------------------------------------
fit = cutree(hc, k = 4)
fit
table(fit)
plot(hc)
rect.hclust(hc, k = 4 , border="red")
rect.hclust(hc, k = 4 , which =2, border="red")
dend %>% color_branches(k=4) %>% plot(horiz=TRUE, main = "Horizontal Dendrogram")
dend %>% rect.dendrogram(k=4,horiz=TRUE)
abline(v = heights_per_k.dendrogram(dend)["4"] + .1, lwd = 2, lty = 2, col = "blue")

#' 
#' ## Clustering Data With Kmeans Method
## ---- cache = TRUE-------------------------------------------------------
fit = kmeans(customer, 4)
fit
#barplot(t(fit$centers), beside = TRUE,xlab="cluster", ylab="value")
plot(customer, col = fit$cluster)
#help(kmeans)

#' 
#' ## Drawing Bivariate Cluster Plot
## ---- cache = TRUE-------------------------------------------------------
#install.packages("cluster")
library(cluster)
clusplot(customer, fit$cluster, color=TRUE, shade=TRUE)
par(mfrow= c(1,2))
clusplot(customer, fit$cluster, color=TRUE, shade=TRUE)
rect(-0.7,-1.7, 2.2,-1.2, border = "orange", lwd=2)
clusplot(customer, fit$cluster, color = TRUE, xlim = c(-0.7,2.2), ylim = c(-1.7,-1.2))
#help(cmdscale)
#help(princomp)
mds = cmdscale(dist(customer), k = 2)
plot(mds, col = fit$cluster)

#' 
#' 
#' ## PCA
## ---- cache = TRUE-------------------------------------------------------
brca.cnv <- read.delim("../data/TCGA_BRCA_CNV_processed.txt")
brca.expr <- read.delim("../data/TCGA_BRCA_Expr_processed.txt")

pr.res <- prcomp(brca.expr, scale = TRUE)
par(mfrow=c(1,2))
plot(pr.res$x[,c(1, 2)], pch = 19)
plot(pr.res$x[,c(1, 2)], pch = 19, col = ifelse(brca.cnv[,"ERBB2_CN"] > 3, rainbow(2)[1], rainbow(2)[2]))
par(mfrow=c(1,1))

order.ERBB2.CNV <- order(brca.cnv[,"ERBB2_CN"])
brca.cnv <- brca.cnv[order.ERBB2.CNV, ]
brca.expr <- brca.expr[rownames(brca.cnv), ]

pr.res <- prcomp(brca.expr, scale = TRUE)
plot(pr.res$x[,c(5, 6)], pch = 19, col = ifelse(brca.cnv[,"ERBB2_CN"] > 3, rainbow(2)[1], rainbow(2)[2]))
pairs(pr.res$x[,1:6], col = ifelse(brca.cnv[,"ERBB2_CN"] > 3, rainbow(2)[1], rainbow(2)[2]), pch = 19)


#' 
#' # Exercise `r exer.num <- exer.num + 1; exer.num`
#' ## Data loading
## ---- echo = FALSE, cache = TRUE-----------------------------------------
#install.packages("foreign")
library(foreign)
Leu.training <- read.arff("../data/leukemia_train_38x7129.arff")
Leu.test <- read.arff("../data/leukemia_test_34x7129.arff")

identical(colnames(Leu.training), colnames(Leu.test))

#' 
#' ## Training 데이터를 이용해서 top 10 predictor 도출하기 (t-test 이용)

#' 
#' ## Logistic regression 모델을 만들고 평가하기
#' t-test로 도출된 top 10 gene들만 predictor (feature)로 이용
#' 

#' 
#' ## Neural network 모델을 만들고 평가하기
#' t-test로 도출된 top 10 gene들만 predictor (feature)로 이용
#' 

#' 
#' ## heatmap 함수를 이용해서 hierarchical clustering이 된 형태의 heat map을 그리세요.
#' 
#' 
