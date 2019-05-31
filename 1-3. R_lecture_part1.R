#' ---
#' title: "R을 활용한 의생명 데이터 분석 - Part 1. R의 Basic and Advanced Analysis"
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
opts_chunk$set(tidy = TRUE, error = TRUE, warning = FALSE)

exer.num = 0
lab.num = 0

#'
#' # Assignment
## ------------------------------------------------------------------------
x=2
x
x <-3
X
ls()
rm(x)
x
ls()

#'
#' # Arithmetic operators
#' 사칙 연산의 우선순위
#' : ^ = ** > * = / > + = -
#'
## ------------------------------------------------------------------------
7*3
7+2*3
(7+2)*3
12/2+4
12/(2+4)
3^2
3**2
2*3^2

#'
#' # Vector
#' ## Combine
## ------------------------------------------------------------------------
x=c(1.5, 2, 2.5)
x
x^2
x=c(x, 3)
x
y=c("This", "is", "an", "example")
y
z=c(x,"x")
z


#' ## Sequence
## ------------------------------------------------------------------------
seq(1, 10, 1)
seq(123, 132, 1)
seq(0, 1, 0.1)
seq(1, 10)
seq(123, 132)
a=seq(10)
a
b= 1:10
b
0:10/10
seq(5, 1, -1)
5:1

#'
#' ## Use of brakcets
## ------------------------------------------------------------------------
x= seq(0, 20, 10)
x
x[1]
x[2]
x[3]
x[4]
x[1:2]
x[c(1, 3)]
x[-1]
x[-c(1:2)]
y=1:2
x[-y]

x= seq(0, 20, 5)
x
x[1]
x[2]
x[5]
x[7]
x[1:4]
x[c(1, 5, 3)]
x[-1]
x[-c(1:2)]
y=1:2
x[-y]


#'
#' ## Arithmetic operators
## ------------------------------------------------------------------------
a= 5*(0:3)
b= 1:4
a+b
a-b
a*b
a/b
a**b

#'
#' # Simple statistics
## ------------------------------------------------------------------------
aa = seq(4, 500, 7)
mean(aa)
sum(aa)/length(aa)
var(aa)
bb=sum((aa - mean(aa))^2)/(length(aa)-1)
sd(aa)
sqrt(bb)
median(aa)
summary(aa)
fivenum(aa)

aa = seq(4, 500, 7)
cumsum(aa)
rev(aa)
min(aa)
max(aa)
quantile(aa)
range(aa)
quantile(aa, c(0,1))
quantile(aa, seq(0,1,0.1))

#'
#' # Logical opeators
## ------------------------------------------------------------------------
3 == 4
3 < 4
3 != 4
x= -3:3
x
x< 2
sum(x < 2)
sum(c (TRUE, TRUE, FALSE, TRUE)) # TRUE => 1, FALSE => 0

y= 1:30
y %% 2
z = y %% 2
z==0
sum(z == 0)
sum(z == 0) == 0
sum(z[seq(1,length(z),2)] %% 2) == 0
sum(z[seq(2,length(z),2)] %% 2) == 0

A = c("A", "B", "A", "D","B")
A == "A"
A == "B"
A[A=="A"]
A[A=="B"]
A[A!="A"]
A[A!="B"]
A[A != "A" & A != "B"]
A[A != "A" | A != "B"]
which(A=="A")
A[which(A=="A")]
A[-which(A=="A")]

weight= 60:68
height= c(seq(120, 155, 5), 135)
weight
height
height< 140
height[height < 140]
weight[weight > 65]
height[height < 140 & height != 120]

#' ## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#' 몸무게가 65보다 작은 사람들의 몸무게 평균을 구하시오
## ---- echo = FALSE-------------------------------------------------------

#'
#'
#' ## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#' 몸무게가 65보다 큰 사람들의 키 평균을 구하시오
## ---- echo = FALSE-------------------------------------------------------

#'
#' ## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#' 키가 130보다 작은 사람들의 몸무게 평균을 구하시오.
## ---- echo = FALSE-------------------------------------------------------

#'
#' # Data structure
#' ## Factor
#' Factor는 R에서 category 변수를 효율적으로 표현하기 위해서 사용
#'
#' **Character type vector를 factor로 변환하기**
## ------------------------------------------------------------------------
gender <-c("Male", "Female", "Female", "Male")
str(gender)
factor_gender<-factor(gender) # factor_gender has two levels called Male and Female
factor_gender
str(factor_gender)
levels(factor_gender)

#' **factor를 원래 type으로 변환하기**
## ------------------------------------------------------------------------
levels(factor_gender)[factor_gender]

#'
#' ## Matrix
#' ### Matrix 만들기
## ------------------------------------------------------------------------
m <- matrix(1:6)
matrix(1:6, nrow=2)
matrix(1:6, nrow=2, byrow=T)
x= 3:8
matrix(x, 3, 2)
matrix(x, ncol=2)
matrix(x, ncol=3)
matrix(x, ncol=3, byrow=T)

#'
#' ### Matrix 연산
## ------------------------------------------------------------------------
A= matrix(0:5, 2, 3)
B= matrix(seq(0, 10, 2), 2, 3)
A+B
A-B
A*B
t(A)%*%B
A%*%t(B)

#'
#' ### Matrix 값 얻기/변경하기
## ------------------------------------------------------------------------
data=c(197, 8, 1.8, 1355, 58, 1.7, 2075, 81, 1.8)
country.data= matrix(data, nrow=3, byrow=T)
country.data
dim(country.data)

country.data[1, 2]
country.data[2, 1:3]
country.data[2, ]
country.data[1,2] = 10
country.data
country.data[1,2] = 8
country.data

#'
#' ### Dimension names
## ------------------------------------------------------------------------
dimnames(country.data)
countries= c("Austria", "France", "Germany")
variables= c("GDP", "Pop", "Inflation")
dimnames(country.data)=list(countries, variables)
country.data
dimnames(country.data)

country.data["Austria", "Pop"]
country.data["France", "Inflation"] <- 2.5
country.data["France", "Inflation"]
country.data

#'
#' ### Matrix 합치기
## ------------------------------------------------------------------------
Area = c(84, 544, 358)
country.data= cbind(country.data, Area)
country.data
Switzerland= c(256, 7, 1.8, 41)
country.data= rbind(country.data, Switzerland)
country.data

#'
#' ## Data frame
#'
## ------------------------------------------------------------------------
name <- c("joe", "jhon", "Nancy")
sex <- c("M", "M", "F")
age <- c(27, 26, 26)
foo <- data.frame(name, sex, age)
foo
rownames(foo)
colnames(foo)

country.data1
EU = c("EU", "EU", "EU", "non-EU")
country.data1= cbind(country.data, EU)
country.data1
country.frame= data.frame(country.data, EU, stringsAsFactors = FALSE)
country.frame
str(country.frame)

country.frame["Austria", "Pop"]
country.frame[, "Pop"]
country.frame$Pop
summary(country.frame)

#'
#' ## Subsetting
## ------------------------------------------------------------------------
country.data[country.data[,"GDP"] > 1000,]
country.data[country.data[,"GDP"] > 1000, "Area"]
country.data[country.data[,"GDP"] > 1000, "Area", drop = FALSE]

country.frame[country.frame[,"GDP"] > 1000,]
country.frame[country.frame[,"GDP"] > 1000, "Area"]
country.frame[country.frame[,"GDP"] > 1000, "Area", drop = FALSE]

country.frame[country.frame$GDP > 1000,]
country.frame[country.frame$GDP > 1000, "Area"]
country.frame[country.frame$GDP > 1000, "Area", drop = FALSE]

#'
#' ## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#' country.frame에서 Pop이 50이상인 나라들만으로 구성된 새로운 data.frame을 contry.frame.pop_more_than_50 이라는 이름으로 만드시오.
## ---- echo = FALSE-------------------------------------------------------
contry.frame.pop_more_than_50 <- country.frame[country.frame$Pop >= 50, ]
print (contry.frame.pop_more_than_50)

#'
#' ## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#' country.frame에서 Pop이 50이상인 나라들의 평균 Area는?
## ---- echo = FALSE-------------------------------------------------------
print (mean(country.frame[country.frame$Pop >= 50, ]$Area))

#'
#' # For loop
## ------------------------------------------------------------------------
for( i in 1:4 ){
  print (i)
}

for( i in 1:4 ){
  max.col = max(country.data[,i])
  print(max.col)
}
for( i in 1:4) {
  sum.col = sum(country.data[,i])
  print(sum.col)
}

for( variable.name in colnames(country.data)) {
  print(variable.name)
}

for( variable.name in colnames(country.data)) {
  sum.col = sum(country.data[,variable.name])
  print(sum.col)
}


#'
#'
#'
#' # apply function
## ------------------------------------------------------------------------

apply(country.data, 2, max)

country.data.colMax <- apply(country.data, 2, max)
print(country.data.colMax)
names(country.data.colMax)
print(country.data.colMax[2])
print(country.data.colMax["Pop"])

apply(country.data, 2, summary)


#'
#' # Row-wise and column-wise functions
## ------------------------------------------------------------------------
rowSums(country.data)
colSums(country.data)
rowMeans(country.data)
colMeans(country.data)

#'
#' # Function (함수) 만들기
#' Variance를 계산하는 함수를 만들어보겠습니다. 아래는 variance를 계산하는 수식입니다.
#' \[Var(x) = \frac{1}{N-1}\sum_{1}^{N}(x_{i}-\bar{x})^2\]
#'
## ------------------------------------------------------------------------
New.var = function(x){
  mean.x = mean(x)
  SS = sum((x-mean.x)^2)
  new.var = 1/(length(x)-1) * SS
  return(new.var)
}
set.seed(2001003)
x = rnorm(100,1,10)
x
var(x)
New.var(x)

#'
#' 아래의 함수들과 apply 함수와 결합하여 연습해 보세요.
#'
#' mean, sd, var, median, etc.
#'
#'
#' #I/O
## ------------------------------------------------------------------------
# Breat cancer dataset
brca.cnv <- read.delim("../data/TCGA_BRCA_CNV_processed.txt") # Copy number variation
brca.expr <- read.delim("../data/TCGA_BRCA_Expr_processed.txt") # Gene expression

print(identical(rownames(brca.cnv), rownames(brca.expr)))

head(rownames(brca.cnv))
head(colnames(brca.cnv))
head(colnames(brca.expr))

brca.cnv[is.na(brca.cnv)] <- 0

mean(brca.expr$ERBB2_Expr)


#'
#' ## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#' 주어진 dataset에서 ERBB2의 CNV가 3보다 큰 tumor sample들의 ERBB2 expression의 평균값을 구하시오.
## ---- echo = FALSE-------------------------------------------------------

#'
#' ## 실습문제 `r lab.num <- lab.num + 1; lab.num`
#' 94개 이상의 tumor sample들에서 CNV가 2 이상인 유전자를 도출하시오.
## ---- echo = FALSE-------------------------------------------------------

#'
#' # Plot
## ------------------------------------------------------------------------
plot(country.frame$GDP, country.frame$Pop)
plot(country.frame$GDP, country.frame$Pop, xlab  = "GDP", ylab = "Populaton")
plot(country.frame$GDP, country.frame$Pop, xlab  = "GDP", ylab = "Populaton", main = "Population ~ GDP")

hist(country.frame$GDP)
boxplot(country.frame$GDP)


#'
#' # 데이터 살펴보기
#' **Univariate Descriptive Statistics in R**
## ------------------------------------------------------------------------
data(mtcars)
str(mtcars)
head(mtcars)

#' *mtcars*
#'
#' |Variable name| Description|
#' |:--------|--------------------------------------:|
#' |[, 1]	mpg	|Miles/(US) gallon|
#' |[, 2]	cyl	|Number of cylinders|
#' |[, 3]	disp	|Displacement (cu.in.)|
#' |[, 4]	hp	|Gross horsepower|
#' |[, 5]	drat	|Rear axle ratio|
#' |[, 6]	wt	|Weight (1000 lbs)|
#' |[, 7]	qsec	|1/4 mile time|
#' |[, 8]	vs	|V/S|
#' |[, 9]	am	|Transmission (0 = automatic, 1 = manual)|
#' |[,10]	gear	|Number of forward gears|
#' |[,11]	carb	|Number of carburetors|
## ------------------------------------------------------------------------
range(mtcars$mpg)
length(mtcars$mpg)
mean(mtcars$mpg)
median(mtcars$mpg)
sd(mtcars$mpg)
var(mtcars$mpg)
sd(mtcars$mpg) ^ 2
IQR(mtcars$mpg)
quantile(mtcars$mpg,0.67)
max(mtcars$mpg)
min(mtcars$mpg)
cummax(mtcars$mpg)
cummin(mtcars$mpg)
summary(mtcars)
table(mtcars$cyl)
stem(mtcars$mpg)
library(ggplot2)
qplot(mtcars$mpg, binwidth=2)
mode <- function(x) {
temp <- table(x)
names(temp)[temp == max(temp)]
}
x = c(1,2,3,3,3,4,4,5,5,5,6)
mode(x)

#' Correlations and Multivariate Analysis
## ------------------------------------------------------------------------
#install.packages("NMF")
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")

library(ComplexHeatmap)
cov(mtcars[1:3])
cor(mtcars[1:3])
Heatmap(cor(mtcars[1:3]))
library(reshape2)
qplot(x=Var1, y=Var2, data=melt(cor(mtcars[1:3])), fill=value, geom="tile")

#' Linear Regression and Multivariate Analysis
## ------------------------------------------------------------------------
lmfit = lm(mtcars$mpg ~ mtcars$cyl)
lmfit
summary(lmfit)
anova(lmfit)
lmfit = lm(mtcars$mpg ~ mtcars$cyl)
plot(mtcars$cyl, mtcars$mpg)
abline(lmfit)

#'
#' # 통계 테스트
#'
#'
#' ##Student's t-Test
## ------------------------------------------------------------------------
boxplot(mtcars$mpg, mtcars$mpg[mtcars$am==0], ylab = "mpg", names=c("overall","automobile"))
abline(h=mean(mtcars$mpg),lwd=2, col="red")
abline(h=mean(mtcars$mpg[mtcars$am==0]),lwd=2, col="blue")
mpg.mu = mean(mtcars$mpg)
mpg_am = mtcars$mpg[mtcars$am == 0]
t.test(mpg_am,mu = mpg.mu)

boxplot(mtcars$mpg~mtcars$am,ylab='mpg',names=c('automatic','manual'))
abline(h=mean(mtcars$mpg[mtcars$am==0]),lwd=2, col="blue")
abline(h=mean(mtcars$mpg[mtcars$am==1]),lwd=2, col="red")
t.test(mtcars$mpg~mtcars$am)

#'
#' ##Pearson's Chi-squared Test
## ------------------------------------------------------------------------
ftable = table(mtcars$am, mtcars$gear)
ftable
mosaicplot(ftable, main="Number of Forward Gears Within Automatic and Manual Cars", color = TRUE)
chisq.test(ftable)


#'
#'
#' #Creating a Graph
#' In R, graphs are typically created interactively.
#'
## ------------------------------------------------------------------------
# Creating a Graph
attach(mtcars)
plot(wt, mpg)
abline(lm(mpg~wt))
title("Regression of MPG on Weight")

#'
#'
#' The plot( ) function opens a graph window and plots weight vs. miles per gallon.
#' The next line of code adds a regression line to this graph. The final line adds a title.
#'
#'
#' Saving Graphs
#' You can save the graph in a variety of formats from the menu
#' File -> Save As.
#'
#' You can also save the graph via code using one of the following functions.
#'
#' Function	Output to
#' pdf("mygraph.pdf")	pdf file
#' win.metafile("mygraph.wmf")	windows metafile
#' png("mygraph.png")	png file
#' jpeg("mygraph.jpg")	jpeg file
#' bmp("mygraph.bmp")	bmp file
#' postscript("mygraph.ps")	postscript file
#' See input/output for details.
#'
#' Viewing Several Graphs
#' Creating a new graph by issuing a high level plotting command (plot, hist, boxplot, etc.) will typically overwrite a previous graph. To avoid this, open a new graph window before creating a new graph. To open a new graph window use one of the functions below.
#'
#' Function	Platform
#' windows()	Windows
#' X11()	Unix
#' quartz()	Mac
#' You can have multiple graph windows open at one time. See help(dev.cur) for more details.
#'
#' Alternatively, after opening the first graph window, choose History -> Recording from the graph window menu. Then you can use Previous and Next to step through the graphs you have created.
#'
#' Graphical Parameters
#' You can specify fonts, colors, line styles, axes, reference lines, etc. by specifying graphical parameters. This allows a wide degree of customization. Graphical parameters, are covered in the Advanced Graphs section. The Advanced Graphs section also includes a more detailed coverage of axis and text customization.
#'
#'
#'
#' #Histograms and Density Plots
#' ##Histograms
#' You can create histograms with the function hist(x) where x is a numeric vector of values to be plotted. The option freq=FALSE plots probability densities instead of frequencies. The option breaks= controls the number of bins.
#'
## ---- cache = TRUE-------------------------------------------------------
# Simple Histogram
hist(mtcars$mpg)

#'
#'
## ---- cache = TRUE-------------------------------------------------------
# Colored Histogram with Different Number of Bins
hist(mtcars$mpg, breaks=12, col="red")


#'
#'
## ---- cache = TRUE-------------------------------------------------------
# Add a Normal Curve (Thanks to Peter Dalgaard)
x <- mtcars$mpg
h<-hist(x, breaks=10, col="red", xlab="Miles Per Gallon",
  	main="Histogram with Normal Curve")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit <- yfit*diff(h$mids[1:2])*length(x)
lines(xfit, yfit, col="blue", lwd=2)


#'
#'
#'
#' Histograms can be a poor method for determining the shape of a distribution because it is so strongly affected by the number of bins used.
#'
#' To practice making a density plot with the hist() function, try this exercise.
#'
#' ##Kernel Density Plots
#' Kernal density plots are usually a much more effective way to view the distribution of a variable. Create the plot using plot(density(x)) where x is a numeric vector.
#'
#'
## ---- cache = TRUE-------------------------------------------------------
# Kernel Density Plot
d <- density(mtcars$mpg) # returns the density data
plot(d) # plots the results

#'
#'
#'
## ---- cache = TRUE-------------------------------------------------------
# Filled Density Plot
d <- density(mtcars$mpg)
plot(d, main="Kernel Density of Miles Per Gallon")
polygon(d, col="red", border="blue")

#'
#'
#'
#'
#'
#' #Bar Plots
#' Create barplots with the barplot(height) function, where height is a vector or matrix. If height is a vector, the values determine the heights of the bars in the plot. If height is a matrix and the option beside=FALSE then each bar of the plot corresponds to a column of height, with the values in the column giving the heights of stacked “sub-bars”. If height is a matrix and beside=TRUE, then the values in each column are juxtaposed rather than stacked. Include option names.arg=(character vector) to label the bars. The option horiz=TRUE to createa a horizontal barplot.
#'
#' ##Simple Bar Plot
## ---- cache = TRUE-------------------------------------------------------
# Simple Bar Plot
counts <- table(mtcars$gear)
barplot(counts, main="Car Distribution",
  	xlab="Number of Gears")

#'
#'
## ---- cache = TRUE-------------------------------------------------------
# Simple Horizontal Bar Plot with Added Labels
counts <- table(mtcars$gear)
barplot(counts, main="Car Distribution", horiz=TRUE,
names.arg=c("3 Gears", "4 Gears", "5 Gears"))


#'
#'
#' ##Stacked Bar Plot
#' # Stacked Bar Plot with Colors and Legend
## ------------------------------------------------------------------------
counts <- table(mtcars$vs, mtcars$gear)
barplot(counts, main="Car Distribution by Gears and VS",
  xlab="Number of Gears", col=c("darkblue","red"),
 	legend = rownames(counts))

#'
#'
#' ##Grouped Bar Plot
## ------------------------------------------------------------------------
# Grouped Bar Plot
counts <- table(mtcars$vs, mtcars$gear)
barplot(counts, main="Car Distribution by Gears and VS",
  xlab="Number of Gears", col=c("darkblue","red"),
 	legend = rownames(counts), beside=TRUE)

#'
#'
#' ##Notes
#' Bar plots need not be based on counts or frequencies. You can create bar plots that represent means, medians, standard deviations, etc. Use the aggregate( ) function and pass the results to the barplot( ) function.
#'
#' By default, the categorical axis line is suppressed. Include the option axis.lty=1 to draw it.
#'
#' With many bars, bar labels may start to overlap. You can decrease the font size using the cex.names = option. Values smaller than one will shrink the size of the label. Additionally, you can use graphical parameters such as the following to help text spacing:
#'
## ---- cache = TRUE-------------------------------------------------------
# Fitting Labels
par(las=2) # make label text perpendicular to axis
par(mar=c(5,8,4,2)) # increase y-axis margin.

counts <- table(mtcars$gear)
barplot(counts, main="Car Distribution", horiz=TRUE, names.arg=c("3 Gears", "4 Gears", "5   Gears"), cex.names=0.8)

#'
#'
#'
#'
#' #Pie Charts
#' Pie charts are not recommended in the R documentation, and their features are somewhat limited. The authors recommend bar or dot plots over pie charts because people are able to judge length more accurately than volume. Pie charts are created with the function pie(x, labels=) where x is a non-negative numeric vector indicating the area of each slice and labels= notes a character vector of names for the slices.
#'
#' ##Simple Pie Chart
## ---- cache = TRUE-------------------------------------------------------
# Simple Pie Chart
slices <- c(10, 12,4, 16, 8)
lbls <- c("US", "UK", "Australia", "Germany", "France")
pie(slices, labels = lbls, main="Pie Chart of Countries")


#'
#'
#' ##Pie Chart with Annotated Percentages
## ---- cache = TRUE-------------------------------------------------------
# Pie Chart with Percentages
slices <- c(10, 12, 4, 16, 8)
lbls <- c("US", "UK", "Australia", "Germany", "France")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels
pie(slices,labels = lbls, col=rainbow(length(lbls)),
  	main="Pie Chart of Countries")

#'
#'
#'
#' ##3D Pie Chart
#' The pie3D( ) function in the plotrix package provides 3D exploded pie charts.
#'
## ---- cache = TRUE-------------------------------------------------------
# 3D Exploded Pie Chart
library(plotrix)
slices <- c(10, 12, 4, 16, 8)
lbls <- c("US", "UK", "Australia", "Germany", "France")
pie3D(slices,labels=lbls,explode=0.1,
  	main="Pie Chart of Countries ")

#'
#'
#'
#' ##Creating Annotated Pies from a data frame
## ------------------------------------------------------------------------
# Pie Chart from data frame with Appended Sample Sizes
mytable <- table(iris$Species)
lbls <- paste(names(mytable), "\n", mytable, sep="")
pie(mytable, labels = lbls,
  	main="Pie Chart of Species\n (with sample sizes)")

#'
#'
#'
#' #Boxplots
#' Boxplots can be created for individual variables or for variables by group. The format is boxplot(x, data=), where x is a formula and data= denotes the data frame providing the data. An example of a formula is y~group where a separate boxplot for numeric variable y is generated for each value of group. Add varwidth=TRUE to make boxplot widths proportional to the square root of the samples sizes. Add horizontal=TRUE to reverse the axis orientation.
#'
## ---- cache = TRUE-------------------------------------------------------
# Boxplot of MPG by Car Cylinders
boxplot(mpg~cyl,data=mtcars, main="Car Milage Data",
  	xlab="Number of Cylinders", ylab="Miles Per Gallon")

#'
#'
#'
## ---- cache = TRUE-------------------------------------------------------
# Notched Boxplot of Tooth Growth Against 2 Crossed Factors
# boxes colored for ease of interpretation
boxplot(len~supp*dose, data=ToothGrowth, notch=TRUE,
  col=(c("gold","darkgreen")),
  main="Tooth Growth", xlab="Suppliment and Dose")


#'
#'
#' In the notched boxplot, if two boxes' notches do not overlap this is ‘strong evidence’ their medians differ (Chambers et al., 1983, p. 62).
#'
#' Colors recycle. In the example above, if I had listed 6 colors, each box would have its own color. Earl F. Glynn has created an easy to use list of colors is PDF format.
#'
#' ##Other Options
#' The boxplot.matrix( ) function in the sfsmisc package draws a boxplot for each column (row) in a matrix. The boxplot.n( ) function in the gplots package annotates each boxplot with its sample size. The bplot( ) function in the Rlab package offers many more options controlling the positioning and labeling of boxes in the output.
#'
#' ##Violin Plots
#' A violin plot is a combination of a boxplot and a kernel density plot. They can be created using the vioplot( ) function from vioplot package.
#'
## ---- cache = TRUE-------------------------------------------------------
# Violin Plots
library(vioplot)
x1 <- mtcars$mpg[mtcars$cyl==4]
x2 <- mtcars$mpg[mtcars$cyl==6]
x3 <- mtcars$mpg[mtcars$cyl==8]
vioplot(x1, x2, x3, names=c("4 cyl", "6 cyl", "8 cyl"),
   col="gold")
title("Violin Plots of Miles Per Gallon")

#'
#'
#'
#'
#' #Scatterplots
#' ##Simple Scatterplot
#' There are many ways to create a scatterplot in R. The basic function is plot(x, y), where x and y are numeric vectors denoting the (x,y) points to plot.
#'
## ---- cache = TRUE-------------------------------------------------------
# Simple Scatterplot
attach(mtcars)
plot(wt, mpg, main="Scatterplot Example",
  	xlab="Car Weight ", ylab="Miles Per Gallon ", pch=19)



# Add fit lines
abline(lm(mpg~wt), col="red") # regression line (y~x)
lines(lowess(wt,mpg), col="blue") # lowess line (x,y)

#'
#'
#'
#'
#' ##Scatterplot Matrices
#' There are at least 4 useful functions for creating scatterplot matrices. Analysts must love scatterplot matrices!
#'
## ---- cache = TRUE-------------------------------------------------------
# Basic Scatterplot Matrix
pairs(~mpg+disp+drat+wt,data=mtcars,
   main="Simple Scatterplot Matrix")

#'
#'
#'
#' The lattice package provides options to condition the scatterplot matrix on a factor.
#'
## ---- cache = TRUE-------------------------------------------------------
# Scatterplot Matrices from the lattice Package
library(lattice)
super.sym <- trellis.par.get("superpose.symbol")
splom(mtcars[c(1,3,5,6)], groups=cyl, data=mtcars,
  	panel=panel.superpose,
   key=list(title="Three Cylinder Options",
   columns=3,
   points=list(pch=super.sym$pch[1:3],
   col=super.sym$col[1:3]),
   text=list(c("4 Cylinder","6 Cylinder","8 Cylinder"))))

#'
#'
#'
#' The car package can condition the scatterplot matrix on a factor, and optionally include lowess and linear best fit lines, and boxplot, densities, or histograms in the principal diagonal, as well as rug plots in the margins of the cells.
#'
## ---- cache = TRUE-------------------------------------------------------
# Scatterplot Matrices from the car Package
library(car)
scatterplotMatrix(~mpg+disp+drat+wt|cyl, data=mtcars,
  	main="Three Cylinder Options")

#'
#'
#'
#' The gclus package provides options to rearrange the variables so that those with higher correlations are closer to the principal diagonal. It can also color code the cells to reflect the size of the correlations.
#'
## ---- cache = TRUE-------------------------------------------------------
# Scatterplot Matrices from the glus Package
library(gclus)
dta <- mtcars[c(1,3,5,6)] # get data
dta.r <- abs(cor(dta)) # get correlations
dta.col <- dmat.color(dta.r) # get colors
# reorder variables so those with highest correlation
# are closest to the diagonal
dta.o <- order.single(dta.r)
cpairs(dta, dta.o, panel.colors=dta.col, gap=.5,
main="Variables Ordered and Colored by Correlation" )

#'
#'
#' ##3D Scatterplots
#' You can create a 3D scatterplot with the scatterplot3d package. Use the function scatterplot3d(x, y, z).
#'
## ------------------------------------------------------------------------
# 3D Scatterplot
library(scatterplot3d)
attach(mtcars)
scatterplot3d(wt,disp,mpg, main="3D Scatterplot")

#'
#'
## ---- cache = TRUE-------------------------------------------------------
# 3D Scatterplot with Coloring and Vertical Drop Lines
library(scatterplot3d)
attach(mtcars)
scatterplot3d(wt,disp,mpg, pch=16, highlight.3d=TRUE,
  type="h", main="3D Scatterplot")


#'
#'
## ---- cache = TRUE-------------------------------------------------------
# 3D Scatterplot with Coloring and Vertical Lines
# and Regression Plane
library(scatterplot3d)
attach(mtcars)
s3d <-scatterplot3d(wt,disp,mpg, pch=16, highlight.3d=TRUE,
  type="h", main="3D Scatterplot")
fit <- lm(mpg ~ wt+disp)
s3d$plane3d(fit)

#'
#'
#' ##Spinning 3D Scatterplots
#' You can also create an interactive 3D scatterplot using the plot3D(x, y, z) function in the rgl package. It creates a spinning 3D scatterplot that can be rotated with the mouse. The first three arguments are the x, y, and z numeric vectors representing points. col= and size= control the color and size of the points respectively.
#'
## ---- eval = FALSE-------------------------------------------------------
## # Spinning 3d Scatterplot
## library(rgl)
##
## plot3d(wt, disp, mpg, col="red", size=3)

#'
#'
#' You can perform a similar function with the scatter3d(x, y, z) in the Rcmdr package.
#'
## ---- eval = FALSE-------------------------------------------------------
## # Another Spinning 3d Scatterplot
## library(Rcmdr)
## attach(mtcars)
## scatter3d(wt, disp, mpg)

#'
#'
#'
#'
