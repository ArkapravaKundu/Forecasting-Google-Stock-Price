library(readxl)
library(forecast)
library(tseries)
library(Metrics)
library(prophet)
library(ggplot2)
library(moments)
google_traindata <- read_excel("C:/Users/arkak/Desktop/google_traindata.xlsx")
View(google_traindata)
google_testdata <- read_excel("C:/Users/arkak/Desktop/google_traindata.xlsx")
View(google_testdata)

attach(google_traindata)
open<-openPrices
tsopen<-ts(open)
test_series<-google_testdata$openPrices

par(bg="seagreen3")
plot(data.train$Date,tsopen,type="l",xlab="Time Period",col="mediumblue")

#Decomposing the data
decomposets<-decompose(ts(open, freq=100))
plot(decomposets)

adf.test(open, alternative = "stationary")

#Differencing the data
open_diff<-diff(open, differences = 1)
plot(ts(open_diff))
par(bg = "slategray3")
plot(ts(open_diff), col="indianred3",ylab="Open_diff",xlab="Time Period")
axis(1,  col.axis="mediumblue")
axis(2,  col.axis="mediumblue")
adf.test(open_diff, alternative = "stationary")

season_open<- diff(tsopen, differences = 365)
plot(season_open, col="mediumblue")
adf.test(season_open, alternative = "stationary")

#Holts seasonal model
Holt_seasonal<-HoltWinters(ts(open,frequency = 365),seasonal = c("additive","multiplicative"))
Holt_seasonal
Lh.pred<-predict(Holt_seasonal, n.ahead = 260)
P<-(-0.15* test_series + 1)^(1/-0.15)
rmse(p,Lh.pred_p)

#Holts exponential model
Holt_exp<- HoltWinters(tsopen, gamma = FALSE, beta = FALSE)
plot(Holt_exp)
Holt_exp_pred<-predict(Holt_exp, n.ahead=260)
rmse(test_series, Holt_exp_pred)


#ARIMA
arima_auto<-auto.arima(tsopen)
arima_auto
pred_arima<-predict(arima_auto, n.ahead = 260)
P<-(-0.15* test_series + 1)^(1/-0.15)
p1<-(-0.15*pred_arima$pred + 1)^(1/-0.15)
rmse(p,p1)

acf(season_open, main="Open seasonal ACF", col="mediumblue",ci.col="indianred3")
pacf(season_open, main="Open seasonal PACF", col="seagreen",ci.col="indianred3")

#ARIMA with seasonality
sarima_auto<-auto.arima(ts(open, frequency = 365), D=1)
sarima_auto
sarima_auto<-arima(ts(open,frequency = 365), order=c(1,1,1), seasonal=c(0,1,0))
sarima_auto_pred<-predict(sarima_auto, n.ahead = 260)

ts.plot(p,p1, col=c("blue","red"), main="Plot of ARIMA(1,1,1)(0,1,0)")
legend("topleft", c("Actual", "Predicted"), fill=c("blue", "red"))
p2<-(-0.15*sarima_auto_pred<$pred + 1)^(1/-0.15)
rmse(p,p2)


#ARIMA with all regressors

sarima_auto_reg<-arima(ts(open,frequency = 365), order=c(1,1,1), seasonal=c(0,1,0), xreg=cbind(compoundSentiment,negSentiment, Gold, Fed_Fund_rate, Infletion, `Net Income`, Gross profit margine, SP500_open ))
sarima_auto_reg_pred<-predict(sarima_auto_reg5,n.ahead = 260, newxreg=cbind(google_testdata$compoundSentiment,google_testdata$negSentiment, google_testdata$Gold, google_testdata$Infletion, google_testdata$`Net Income`, google_testdata$Fed_Fund_rate, google_testdata$Gross profit margine,google_testdata$SP500_open))
ts.plot(test_series, sarima_auto_reg_pred$pred, col=c("blue","red"))
p3<-(-0.15*sarima_auto_reg_pred$pred + 1)^(1/-0.15)
rmse(p,p3)


sarima_auto_reg5<-arima(ts(open,frequency = 365), order=c(1,1,1), seasonal=c(0,1,0), xreg=cbind(compoundSentiment,negSentiment, Gold, Infletion, `Net Income`, BlackMonday_11,PresElection_16))
sarima_auto_reg5_pred<-predict(sarima_auto_reg5,n.ahead = 260, newxreg=cbind(google_testdata$compoundSentiment,google_testdata$negSentiment, google_testdata$Gold, google_testdata$Infletion, google_testdata$`Net Income`,  google_testdata$BlackMonday_11,google_testdata$PresElection_16))
ts.plot(test_series, sarima_auto_reg5_pred$pred, col=c("blue","red"))
p4<-(-0.15*sarima_auto_reg5_pred$pred + 1)^(1/-0.15)
rmse(p,p4)

#ARIMA with regressors and events
sarima_auto_reg9<-arima(ts(open,frequency = 365), order=c(1,1,1), seasonal=c(0,1,0), xreg=cbind(compoundSentiment,negSentiment, posSentiment,  Infletion, HurricaneSandy_12, `Quarterly Impact`, BlackMonday_11,PresElection_16))
sarima_auto_reg9_pred<-predict(sarima_auto_reg9, n.ahead=260, newxreg=cbind( google_testdata$compoundSentiment, google_testdata$negSentiment,  google_testdata$posSentiment,  google_testdata$Infletion,google_testdata$HurricaneSandy_12,  google_testdata$BlackMonday_11, google_testdata$PresElection_16, google_testdata$`Quarterly Impact`))
ts.plot(test_series, sarima_auto_reg9_pred$pred, col=c("blue","red"))
p5<-(-0.15*sarima_auto_reg9_pred$pred + 1)^(1/-0.15)
rmse(p,p5)
ts.plot(p,p5, col=c("blue","red"), main="ARIMA Model with regressors and events")
legend("topleft", c("Actual", "Predicted"), fill=c("blue", "red"))

plot(sarima_auto_reg9$residuals, main="Residual plot",col="darkgreen")
adf.test(sarima_auto_reg9$residuals)
Box.test(google_testdata$openPrices,type='Ljung')


#Prophet Model

history <- data.frame(ds = google_traindata$Date,y = google_traindata$openPrices)
m<-prophet(history)
future <- make_future_dataframe(m, periods = 260)
forecast<-predict(m, future)
plot(m, forecast, main="Prophet Model Forecast", xlab="Time",ylabel ="Stock Price")
prop_pred<- forecast$yhat[2476:2735]
length(prop_pred)
p6<-(-0.15*prop_pred + 1)^(1/-0.15)
rmse(p,p6)
