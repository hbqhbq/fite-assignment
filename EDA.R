#EDA
library(ggplot2)
library(corrplot)


#import dataset
train <- read.csv('../input/fraud-detection/fraudTrain.csv')
test <- read.csv('../input/fraud-detection/fraudTest.csv')

#Combine train and test into one data set named creditcard.
creditcard<-rbind(train,test)
dim(creditcard)

str(creditcard)
summary(creditcard)

#Count number of fraud and not fraud data
ggplot(data =creditcard, aes (x = factor (is_fraud),
                              fill = factor (is_fraud),
                              label = stat (count))) +
  geom_bar (position = "dodge") +
  geom_text (stat = 'count',
             position = position_dodge (.9),
             vjust = -0.5,
             size = 3) +
  scale_x_discrete (labels = c("no fraud", "fraud")) +
  labs(x= 'isFraud') +
  ggtitle("Count number of fraud and not fraud data")

#################
######Amount#####
#################

summary(creditcard$amt)
#Box Plot for credit card Amount
boxplot(creditcard$amt, col='maroon',xlab = 'credit card amount', main = 'Box Plot for credit card Amount')

#Histogram for credit card Amount
hist(creditcard$amt[creditcard$amt<500], 
     main="Histogram for credit card Amount", 
     xlab="card Amount", 
     border="red", 
     las=1, 
     breaks=20, prob = TRUE)

#Card Amount (log) groupby Fraud
ggplot (data = creditcard, aes(x=factor(is_fraud),y =log1p(amt),fill =factor(is_fraud) )) +
  geom_boxplot () +
  labs(x = 'isFraud', y = 'amount') +
  ggtitle ( 'Card Amount (log) groupby Fraud' )

#################
####Category#####
#################

#Category type
ggplot(data = creditcard, aes (x = factor (category), fill = factor (category)),width=20) +
  geom_bar (position = "dodge",width =0.7) +
  
  coord_flip()+
  labs(x= 'type') +
  ggtitle("Category type")

#isFraud groupby category type
ggplot(data =fraud , aes (x=factor(category), fill = factor (is_fraud),label = stat (count))) +
  geom_bar (position = "dodge",width=0.5) +
  geom_text (stat = 'count',
             position = position_dodge (.9),
             vjust = -0.5,
             size = 3) +
  
  coord_flip()+
  ggtitle("isFraud groupby category type")

###############
#####State#####
###############

#Number of Credit Card Transactions by State
ggplot(data = creditcard, aes (x = factor (state), fill = factor (state))) +
  geom_bar (position = "dodge") +
  labs(x= 'state') +
  coord_flip()+
  ggtitle("Number of Credit Card Transactions by State")

#Number of Credit Card Frauds by State
ggplot(data = fraud, aes (x = factor (state), fill = factor (state))) +
  geom_bar (position = "dodge") +
  labs(x= 'state') +
  coord_flip()+
  ggtitle("Number of Credit Card Frauds by State")

################
#####Gender#####
################

#isFraud groupby gender type
ggplot(data =creditcard , aes (x=factor(is_fraud), fill = factor (gender),label = stat (count))) +
  geom_bar (position = "dodge") +
  geom_text (stat = 'count',
             position = position_dodge (.9),
             vjust = -0.5,
             size = 3) +
  scale_x_discrete (labels = c("no fraud", "fraud"))+
  
  ggtitle("isFraud groupby gender type")

#Pie Chart for Gender
df<-data.frame(type=c("M_fraud","F_fraud"),value=c(4752,4899))
ggplot(df,aes(x='',y=value,fill=type))+
  geom_bar(stat = 'identity',width = 1,position = 'stack')+
  geom_text(aes(1.2,label=scales::percent(value/sum(value))),size=5,position = position_stack(vjust = 0.5))+
  scale_y_continuous(expand = c(0,0))+
  theme_bw()+
  labs(x=NULL,y=NULL,title = 'Pie Chart for Gender')+
  theme(legend.title = element_blank(),
        legend.position = 'bottom',
        legend.text = element_text(colour = 'black',size = 16),
        axis.text = element_blank(),
        axis.title = element_blank(),
        panel.border = element_blank(),
        panel.grid = element_blank(),
        plot.title = element_text(hjust = 0.5,size = 20)
  )+
  coord_polar(theta = 'y', start = 0, direction = 1)

#######################
#####Date of Birth#####
#######################

#Histogram and Density Plot for Date of Birth
ggplot(data=creditcard,aes(dob))+
  geom_histogram(aes(y=..density..),colour="black", fill="white")+
  geom_density(alpha=.2, fill="red") 

#Histogram and Density Plot for Date of Birth groupby is_fraud
ggplot(data=fraud,aes(x=dob))+
  geom_histogram(aes(y=..density..,fill=is_fraud),colour="black", fill="white")+
  geom_density(alpha=.2, fill="red") 

###############################

#corelation
corelation<-cor(creditcard[,sapply(creditcard,is.numeric)],use="complete.obs",method="pearson")
corrplot(corelation, method = 'number')
