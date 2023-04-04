# Load the data
paysim<-
  read.csv("C:/Users/hbq/Desktop/paysim.csv", header=FALSE)

# Load libraries
library(ggplot2)
library(corrplot)

# Preprocess the data and distinguish attributes
data<-paysim
head(data)
colSums(is.na(data))
data<-data[,-4]
data<-data[,-6]
data<-data[,-2]

# Ubivariate analysis
table(paysim$isFraud)
paysim %>% 
  group_by(type) %>% 
  summarise(sum_isfraud = sum(isFraud))

ggplot(transactions,aes(x=isFraud,fill=isFraud))+
       geom_bar(stat = 'count')+labs(x = '0 and 1')+
       theme_grey(base_size = 11)

plot_histogram(data$oldbalanceOrg)
plot_histogram(data$amount)

data %>%
  ggplot(aes(x = factor(type))) + 
  geom_bar()

# Bi-varaite analysis
res<-cor(data)
corrplot(res, method="number")

# Outlier analysis
boxplot(data$amount) 
boxplot(data$oldbalanceOrg) 
boxplot(data$oldbalanceDest) 

# Load the data
creditcard<-
  read.csv("C:/Users/hbq/Desktop/creditcard.csv", header=FALSE)

# Preprocess the data
data_credit<-creditcard[,-31]
data_credit<-data_credit[-1,]
data_credit<-
  as.data.frame(lapply(data_credit,as.numeric))

# Find k
wss <- (nrow(data_credit)-1)*sum(apply(data_credit,2,var))
for (i in 2:15) wss[i] <- 
  sum(kmeans(data_credit,centers=i)$withinss)

plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares")

# Show groups
data1 <- kmeans(x= data_credit, centers = 2)
data_credit$CLUSTER <-  as.factor(data1$cluster)

data_credit  %>% 
  group_by(CLUSTER) %>% 
  summarise_all(mean)
