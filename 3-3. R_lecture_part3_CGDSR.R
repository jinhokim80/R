#' ---
#' title: "R을 활용한 의생명 데이터 분석 - Part 3. R을 활용한 TCGA 암유전체데이터 분석"
#' author: "Jinho Kim"
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

#' 
#' 
#' # CDGS R
#' 
#' ## Install cgdsr package
## ---- eval = FALSE-------------------------------------------------------
## #install.packages("cgdsr")

#' 
#' ## Loading the package
## ---- message = FALSE, warning = FALSE-----------------------------------
library(knitr)
library(cgdsr)

#' 
#' ## Downloading data
## ------------------------------------------------------------------------
## Create CGDS object
mycgds = CGDS("http://www.cbioportal.org/")

## Select a cancer study
#View(getCancerStudies(mycgds)) #View table of studies list
cancer.studies <- getCancerStudies(mycgds)
kable(cancer.studies[,c(1,2)])
mycancerstudy = cancer.studies[38,1] #Breast Invasive Carcinoma (TCGA, Cell 2015)
print (mycancerstudy)

## Get available case lists for a given cancer study
#View(getCaseLists(mycgds,mycancerstudy))
case.list <- getCaseLists(mycgds,mycancerstudy)
kable(case.list[, 1:2])
mycaselist = case.list[1,1] #All Complete Tumors
print (mycaselist)

## Get available genetic profiles
genetic.profile <- getGeneticProfiles(mycgds,mycancerstudy)
kable(genetic.profile[, 1:2])
cna <- genetic.profile[5,1] #linear_CNA
print (cna)
mrna <- genetic.profile[4,1] #rna_seq_v2_mrna
print (mrna)

#' 
#' # Plots
## ------------------------------------------------------------------------
## Get data slices for a specified list of genes, genetic profile and case list
cnadata=getProfileData(mycgds,c("ERBB2"),cna,mycaselist) #CNA data of ERBB2
kable(head(cnadata))
mrnadata=getProfileData(mycgds,c("ERBB2"),mrna,mycaselist) #mRNA data of ERBB2
kable(head(mrnadata))

##Histogram
hist(cnadata$ERBB2,main="ERBB2",xlab="CNA") # plot generation
hist(mrnadata$ERBB2,main="ERBB2",xlab="mRNA") 
hist(log(mrnadata$ERBB2),main="ERBB2",xlab="mRNA") # log transformation

##box plot
boxplot(cnadata$ERBB2,main="ERBB2",ylab="CNA")
boxplot(log(mrnadata$ERBB2),main="ERBB2",ylab="mRNA(log)")

##Scatter plot
co=cor(cnadata$ERBB2, log(mrnadata$ERBB2), use = "na.or.complete") #pearson correlation
cor(cnadata$ERBB2, log(mrnadata$ERBB2), method = "pearson", use = "na.or.complete")
cor(cnadata$ERBB2, log(mrnadata$ERBB2), method = "spearman", use = "na.or.complete")


plot(cnadata$ERBB2,log(mrnadata$ERBB2),main="ERBB2",xlab="CNA",ylab="mRNA(log)")
# Add text on the plot
text(0,12,sprintf("r= %.2f", co)) # add text

#' 
## ---- eval = FALSE-------------------------------------------------------
## pdf("TCGA_BRCA_ERBB2_correlation.pdf") #open PDF
## plot(cnadata$ERBB2,log(mrnadata$ERBB2),main="ERBB2",xlab="CNA",ylab="mRNA(log)")
## text(0,12,"r=0.85") # add text
## dev.off() #close PDF

#' 
#' 
#' #ggplot2
## ---- eval = FALSE-------------------------------------------------------
## #install.packages("ggplot2") ## library install
## 

## ---- message = FALSE, warning = FALSE-----------------------------------
library(ggplot2) ## Loading library

#' 
## ------------------------------------------------------------------------
input.cna=data.frame(cnadata,'CNA',stringsAsFactors = F)
colnames(input.cna)=c("CNA","Type")

plot1=ggplot(input.cna,aes(x=Type,y=CNA))+
  geom_boxplot(width=0.7,outlier.size = NA) +
  geom_jitter(width=0.35,colour="blue",alpha=0.5)

plot1

input.mrna=data.frame(log(mrnadata),'mRNA',stringsAsFactors = F)
colnames(input.mrna)=c("mRNA","Type")

plot2=ggplot(input.mrna,aes(x=Type,y=mRNA))+
  geom_boxplot(width=0.7,outlier.size = NA) +
  geom_jitter(width=0.35,colour="green",alpha=0.5)

plot2

input.data=data.frame(cnadata,log(mrnadata),stringsAsFactors = F)
colnames(input.data)=c("CNA","mRNA")

plot3=ggplot(input.data,aes(x=CNA,y=mRNA))+
  geom_point(color='red',alpha=0.5)+
  geom_text(aes(0, 12), label ="p=0.85")

plot3

#' 
## ---- eval = FALSE-------------------------------------------------------
## #install.packages("gridExtra")

#' 
#' 
## ---- message = FALSE, warning = FALSE-----------------------------------
library(gridExtra)

#' 
## ------------------------------------------------------------------------
grid.arrange(plot1,plot2,plot3, nrow=1, ncol=3)

#' 
## ---- eval = FALSE-------------------------------------------------------
## pdf("ggplot2.pdf",width=12,height=4)
## grid.arrange(plot1,plot2,plot3, nrow=1, ncol=3)
## dev.off()
## 

#' 
#' #Heatmap
#' 
## ---- message = FALSE, warning = FALSE-----------------------------------
library(ComplexHeatmap)

#' 
## ------------------------------------------------------------------------

geneset=c("RUNX1","PIK3CA","TP53","GATA3","FOXA1","SF3B1","PTEN","CBFB","CDH1","TBX3","MAP2K4","MAP3K1","ERBB2","KMT2C","NCOR1","FAM86B2","CDKN1B","HIST1H3B","THEM5","FAM86B1","GPS2","AQP12A","PIK3R1","ACTL6B","ZFP36L1","RB1","KRAS","EPDR1","C1QTNF5","ZFP36L2","CTCF","ASB10","FBXW7","RPGR","MYB","TBL1XR1","CASP8","TCP10","WSCD2","AARS","FAM20C","HIST1H2BC","ARID1A","PTHLH")

geneset_cnadata=getProfileData(mycgds,geneset,cna,mycaselist) #cna data

geneset_cnadata=geneset_cnadata[c(1:44),] #sample selection

geneset_cnadata=t(geneset_cnadata) # transpose data

geneset_cnadata[is.na(geneset_cnadata)] <- 0
#View(geneset_cnadata)


ann_col = HeatmapAnnotation(Group = c(rep('A',22),rep('B',22)))

Heatmap(geneset_cnadata, top_annotation = ann_col)

#' 
## ---- eval = FALSE-------------------------------------------------------
## pdf('heatmap.pdf',paper='a4')
## Heatmap(geneset_cnadata, top_annotation = ann_col)
## dev.off()

#' 
## ---- eval = FALSE-------------------------------------------------------
## write.table(geneset_cnadata[1:10, 1:10], "cnadata.10x10.txt")
## write.table(geneset_cnadata[1:10, 1:10], "cnadata.10x10.txt", sep = "\t", quote = FALSE)

#' 
#' 
#' 
#' 
#' 
