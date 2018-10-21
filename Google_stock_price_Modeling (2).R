#Functions
Rsquare <- function(y,y_new)
{
  RSS <- sum((y-y_new)^2)
  TSS <- sum((y-mean(y))^2)
  return(1-RSS/TSS)
}


 #imported the data set
library(readxl)
SP <- read.csv("C:/Uconn/Summer 18/DataMining/PRoject/google_full_data_scale.csv")
#install.packages("usdm")
library(usdm)
#View(SP)
#class(SP)
sapply(SP, class)
SP$Date=  as.Date(SP$Date,format="%m/%d/%Y")
SP$StockMarketCrash_08 = factor(SP$StockMarketCrash_08)
SP$BlackMonday_11 = factor(SP$BlackMonday_11)
SP$Apple_Impact12 = factor(SP$Apple_Impact12)
SP$HurricaneSandy_12 = factor(SP$HurricaneSandy_12)
SP$GoogleInternetGrowth_13 = factor(SP$GoogleInternetGrowth_13)
SP$GlobalCrisis_14 = factor(SP$GlobalCrisis_14)
SP$PresElection_16 = factor(SP$PresElection_16)
SP$EUFine_17 = factor(SP$EUFine_17)
SP$ImportTariff_18 = factor(SP$ImportTariff_18)
SP$Sept_Low = factor(SP$Sept_Low)
SP$Quarterly.Impact = factor(SP$Quarterly.Impact)
attach(SP)


#Correlation
library(corrplot)
View(colnames(SP))

corr <- cor(SP[,c(                "openPrices"             
                                  ,"compoundSentiment",       "negSentiment"       ,     "neuSentiment"           
                                  ,"posSentiment"        ,    "Gold"             ,       "Fed_Fund_rate"          
                                  ,"Inflation"           ,                 "Net.Income"             
                                  ,  "Gross.profit.margine"   
                                  ,"Stock.volume"          ,  "SP500_open"             
)])
S =corrplot(corr)
#options(scipen = 999)

SP =SP[, 3:25]
#View(SP)
#View(Sp_new)
library(car)

#collinearity between SP 500  and stock volume and sentiment scores is checked 
model1  <- lm(SP500_open~ .- openPrices, data = SP)
vif(model1)

model2  <- lm(Stock.volume~ .- openPrices , data = SP)
vif(model2)
View(vif(model2))

model3  <- lm(neuSentiment~ .- openPrices, data = SP)
vif(model3)

model4  <- lm(negSentiment~ .- openPrices, data = SP)
vif(model4)

model5  <- lm(Stock.volume~ .- openPrices -neuSentiment, data = SP)
vif(model5)
#View(vif(model5))

model6  <- lm(Stock.volume~ .- openPrices -negSentiment, data = SP)
vif(model6)

# removing neutral  sentiment  as it decreased VIF vlaues 

colnames(SP) 
SP= SP[-4]
#View(SP)
colnames(SP)
SP=SP[-20]
## Splitting data in training and test set
#install.packages('caTools')


training_set = SP[1:2472,]
test_set = SP[2473:2735,]

#removing stock volume as it correlated with SP5 00
colnames(SP) 
SP= SP[-10]

training_set=training_set[-10]
test_set = test_set[-10]
colnames(training_set)

#Multiple linear regression model -FULL-All variables
full = lm(formula = openPrices ~ ., data = training_set)
summary(full)
null = lm(formula=openPrices ~ 1 ,data=training_set)

# using backward elimination method to fit our Multiple Linear regression model

regressor_1 = step(full,data = training_set,direction = "backward")
summary(regressor_1)

# using forward elimination method to fit our Multiple Linear regression model
regressor_2 = step(null,scope=list(lower=null, upper=full),data = training_set,direction = "forward")
summary(regressor_2)

#mixed stepwise regression
regressor_3 = step(null,scope=list(lower=null, upper=full),data = training_set,direction = "both")
summary(regressor_3)


#model build
model1 =lm(formula = openPrices ~ compoundSentiment + negSentiment + 
             posSentiment + Gold + Fed_Fund_rate + Inflation + Net.Income + 
             Gross.profit.margine + SP500_open + StockMarketCrash_08 + 
             Apple_Impact12 + HurricaneSandy_12 + GoogleInternetGrowth_13 + 
             GlobalCrisis_14 + PresElection_16 + Sept_Low +Sno, data = training_set)
summary(model1)
test_pred <- predict(model1, newdata = test_set)
rmse(test_set$openPrices,test_pred)
BIC(model1)



#adding significant variables from lass and ridge --Apple Impact 
model3 =lm(formula = openPrices ~ SP500_open + Inflation + StockMarketCrash_08 + 
             Net.Income + Gold + Fed_Fund_rate + BlackMonday_11 + PresElection_16 + 
             ImportTariff_18 + Gross.profit.margine + compoundSentiment + 
             negSentiment + posSentiment + EUFine_17 + Sept_Low+Apple_Impact12, data = training_set)
summary(model3)
test_rmse_model3 <- sqrt(mean((test_set$openPrices - test_pred)^2))
test_rmse_model3
BIC(model3)

library(Metrics)
r1 =lm(formula = openPrices ~ SP500_open + Inflation + StockMarketCrash_08 + 
Net.Income + Gold + Fed_Fund_rate + BlackMonday_11 + PresElection_16 + Gross.profit.margine + compoundSentiment + 
  negSentiment + posSentiment + EUFine_17 + Sept_Low+Apple_Impact12+SP500_open :Inflation+           SP500_open :Net.Income+ 
  SP500_open :Gold + SP500_open :Fed_Fund_rate+SP500_open :Gross.profit.margine+SP500_open : compoundSentiment +
  SP500_open :negSentiment + SP500_open:posSentiment+Inflation :Net.Income+ Inflation :Gold+
  Inflation :Fed_Fund_rate +
  Inflation :Gross.profit.margine+
  Inflation : compoundSentiment +
  Inflation :negSentiment +
  Inflation :posSentiment+
  Net.Income :Inflation+
  Net.Income :Gold +
  Net.Income :Fed_Fund_rate +
  Net.Income :Gross.profit.margine+
  Net.Income : compoundSentiment +
  Net.Income :negSentiment+
  Net.Income :posSentiment+
  Gold:Inflation+
  Gold :Net.Income+ 
  Gold :Fed_Fund_rate +
  Gold :Gross.profit.margine+
  Gold : compoundSentiment +
  Gold :negSentiment +
  Gold :posSentiment+
  Fed_Fund_rate:Net.Income+ 
  Fed_Fund_rate:Gross.profit.margine+
  Fed_Fund_rate: compoundSentiment +
  Fed_Fund_rate:negSentiment +
  Fed_Fund_rate:posSentiment+
  Gross.profit.margine:Net.Income+ 
  Gross.profit.margine:compoundSentiment +
  Gross.profit.margine:negSentiment +
  Gross.profit.margine:posSentiment+posSentiment:negSentiment+posSentiment:compoundSentiment+negSentiment:compoundSentiment, data = training_set)


summary(r1)
r1_pred =  predict(r1,newdata = test_set)
rmse(test_set$openPrices,r1_pred)


plot_df1 = data.frame(x = seq(1:263),"y1" = (-0.15*test_set$openPrices + 1)^(1/-0.15), "y2" = (-0.15*r1_pred + 1)^(1/-0.15))
p1 = ggplot()+
  geom_line(data = plot_df1, aes(x = x, y = y1), color = "blue")+
  geom_line(data = plot_df1, aes(x = x, y = y2), color = "red")
print(p1)


#Checking assumption of linear model on the best linear regression model

#test for heteroskedasticity
library(ggplot2)
install.packages("data.table")
library(data.table)
ggplot(data = data.table(Fitted_values = model1$fitted.values,
                         Residuals = model1$residuals),
       aes(Fitted_values, Residuals)) +
  geom_point(size = 1.7) +
  geom_smooth() +
  geom_hline(yintercept = 0, color = "red", size = 1) +
  labs(title = "Fitted values vs Residuals")

#QQ plot 

ggQQ <- function(lm){
  # extract standardized residuals from the fit
  d <- data.frame(std.resid = rstandard(lm))
  # calculate 1Q/4Q line
  y <- quantile(d$std.resid[!is.na(d$std.resid)], c(0.25, 0.75))
  x <- qnorm(c(0.25, 0.75))
  slope <- diff(y)/diff(x)
  int <- y[1L] - slope * x[1L]
  
  p <- ggplot(data = d, aes(sample = std.resid)) +
    stat_qq(shape = 1, size = 3) +         # open circles
    labs(title = "Normal Q-Q",             # plot title
         x = "Theoretical Quantiles",      # x-axis label
         y = "Standardized Residuals") +   # y-axis label
    geom_abline(slope = slope, intercept = int, linetype = "dashed",
                size = 1, col = "firebrick1") # dashed reference line
  return(p)
}

ggQQ(r1)




actual_test = test_set$openPrices
actual_test =(-0.15*actual_test  + 1)^(1/(-0.15))
pred_test = predict(model1,newdata = test_set)
pred_test =(-0.15*pred_test  + 1)^(1/(-0.15))
pred_test
ts.plot(actual_test,ts(pred_test),col=c("blue","red"))
library(Metrics)
rmse(actual_test,pred_test)
