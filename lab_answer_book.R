## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#몸무게가 65보다 작은 사람들의 몸무게 평균을 구하시오
mean(weight[weight < 65])


## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#몸무게가 65보다 큰 사람들의 키 평균을 구하시오
mean(height[weight > 65])

## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#키가 130보다 작은 사람들의 몸무게 평균을 구하시오.
mean(weight[height < 130])

## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#country.frame에서 Pop이 50이상인 나라들만으로 구성된 새로운 data.frame을 contry.frame.pop_more_than_50 이라는 이름으로 만드시오.
contry.frame.pop_more_than_50 <- country.frame[country.frame$Pop >= 50, ]
print (contry.frame.pop_more_than_50)

## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#country.frame에서 Pop이 50이상인 나라들의 평균 Area는?
print (mean(country.frame[country.frame$Pop >= 50, ]$Area))


## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#주어진 dataset에서 ERBB2의 CNV가 3보다 큰 tumor sample들의 ERBB2 expression의 평균값을 구하시오.
mean(brca.expr$ERBB2_Expr[brca.cnv$ERBB2_CN > 3])

## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#94개 이상의 tumor sample들에서 CNV가 2 이상인 유전자를 도출하시오.
colnames(brca.cnv)[colSums(brca.cnv >= 2) >= 94]



## TCGA_BRCA_CNV_processed.txt, TCGA_BRCA_Expr_processed.txt 파일을 읽어들이고 head함수를 이용해서 읽어들인 데이터의 첫 5행과 5열을 출력하세요.
brca.cnv <- read.delim("../data/TCGA_BRCA_CNV_processed.txt") # Copy number variation
brca.expr <- read.delim("../data/TCGA_BRCA_Expr_processed.txt") # Gene expression
print(brca.cnv[1:5, 1:5])
print(brca.expr[1:5, 1:5])

## ERBB2 유전자의 expression과 copy number를 각각 x축과 y축으로하는 scatter plot을 그리고 linear model을 만들어 추세선을 그리세요.
plot(brca.expr$ERBB2_Expr, brca.cnv$ERBB2_CN, xlab = "Expression", ylab = "CNV", main = "ERBB2")

lmfit <- lm(brca.cnv$ERBB2_CN ~ brca.expr$ERBB2_Expr)
abline(lmfit)

## linear model의 summary를 출력하세요.
summary(lmfit)

## plot 함수를 이용하여 모델을 평가하는 plot 4개를 그리세요.
par(mfrow=c(2,2))
plot(lmfit)

## polynomial regression을 이용해서 model을 만들고 평가 plot을 그리세요
lmfit.poly <- lm(brca.cnv$ERBB2_CN ~ poly(brca.expr$ERBB2_Expr, 3))
par(mfrow=c(2,2))
plot(lmfit.poly)

## Generalized additive model을 만들고 summary를 출력하세요.
library(mgcv)
data.erbb2 <- data.frame(cbind(CN = brca.cnv$ERBB2_CN, Expr = brca.expr$ERBB2_Expr))
gamfit <- gam(CN ~ s(Expr), data = data.erbb2)
summary(gamfit)

## 아래와 같이 모델의 추세곡선을 그리세요.
plot(gamfit, ylab = "CNV", main = "ERBB2")

## 모델의 평가 plot을 그리세요.
gam.check(gamfit)


## TCGA_BRCA_CNV_processed.txt, TCGA_BRCA_Expr_processed.txt 파일을 읽어들이고 head함수를 이용해서 읽어들인 데이터의 첫 5행과 5열을 출력하세요.
brca.cnv <- read.delim("../data/TCGA_BRCA_CNV_processed.txt") # Copy number variation
brca.expr <- read.delim("../data/TCGA_BRCA_Expr_processed.txt") # Gene expression
print(brca.cnv[1:5, 1:5])
print(brca.expr[1:5, 1:5])

## ERBB2 유전자의 expression과 copy number를 각각 x축과 y축으로하는 scatter plot을 그리고 linear model을 만들어 추세선을 그리세요.
plot(brca.expr$ERBB2_Expr, brca.cnv$ERBB2_CN, xlab = "Expression", ylab = "CNV", main = "ERBB2")

lmfit <- lm(brca.cnv$ERBB2_CN ~ brca.expr$ERBB2_Expr)
abline(lmfit)

## linear model의 summary를 출력하세요.
summary(lmfit)

## plot 함수를 이용하여 모델을 평가하는 plot 4개를 그리세요.
par(mfrow=c(2,2))
plot(lmfit)

## polynomial regression을 이용해서 model을 만들고 평가 plot을 그리세요
lmfit.poly <- lm(brca.cnv$ERBB2_CN ~ poly(brca.expr$ERBB2_Expr, 3))
par(mfrow=c(2,2))
plot(lmfit.poly)

## Generalized additive model을 만들고 summary를 출력하세요.
library(mgcv)
data.erbb2 <- data.frame(cbind(CN = brca.cnv$ERBB2_CN, Expr = brca.expr$ERBB2_Expr))
gamfit <- gam(CN ~ s(Expr), data = data.erbb2)
summary(gamfit)

## 아래와 같이 모델의 추세곡선을 그리세요.
plot(gamfit, ylab = "CNV", main = "ERBB2")

## 모델의 평가 plot을 그리세요.
gam.check(gamfit)
