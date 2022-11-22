#importazione dei dati del training set
df.train<-read.csv("C:/Users/miky1/OneDrive/Desktop/MLCUP-TR.csv", sep=";")
head(df.train)

#standardizzazione dei dati
df.train<-data.frame(scale(df.train[,-1]))
str(df.train)
head(df.train)

#importazione dati dl test set
df.test<-read.csv("C:/Users/miky1/OneDrive/Desktop/ML-CUP22-TS.csv", sep=";")

#standardizzazione
df.test<-data.frame(scale(df.test[,-1]))

#summary
summary(df.train)

head(df.test)
summary(df.test)


#install.packages(c("leaps", "rpart", "mcgv", "glmnet", "boot", "rpart.plot"))

library(nnet)
library(dplyr)
library(ggplot2)
library(GGally)
#library(MASS)
#library(ggplot2)
#library(dplyr)
#library(tidyverse)
library(corrplot)
#library(leaps)
#library(rpart)
#library(mgcv)
#library(glmnet)
#library(boot)
#library(rpart.plot)
#library(keras)
library(caret)

#Il grafico che segue mostra le relazioni tra tutte le colonne del data frame



dev.new();corrplot(cor(df.train), method = "number", type = "upper", diag = FALSE)
dev.new();ggpairs(df.train)


set.seed(12383010)


#creazione NN
nn0 <- nnet(output1+ output2  ~ input1+ input2+ input3+ input4+ input5 +input6+ input7+ input8 +input9, data = df.train,
            size=6, linout=TRUE, skip=TRUE, MaxNWts=1000, trace=FALSE, maxit=500000)

str(nn0)

summary(df.train$output1+df.train$output2)

#dicotomizzazione output
df.train$out<-factor(ifelse(df.train$output1+df.train$output2<0, "0","1"))

#adattamento dati valori predettei
df.train$pred <- as.vector(predict(nn0, type="raw"))

df.train$pred_class <- factor(ifelse(df.train$pred < 0, "0", "1"))

#valutazione modello
confusionMatrix(df.train$pred_class, reference = (df.train$out))

#unificazione dataset training+test
data_df <- df.train %>%
mutate(set="train") %>%
bind_rows(df.test %>% mutate(set="test"))

#previsione valori 
data_df$fit <- predict(nn0, data_df)

data_df$output1[1493:2021]<-data_df$fit[1493:2021]
data_df$output2[1493:2021]<-data_df$fit[1493:2021]

#mean absolute error
mae <- data_df %>%
filter(set=="train") %>%
summarise(mae = mean(abs(fit-(output2+output1)))) %>%
pull()
print(mae)


ggp <- ggplot(data = data_df, mapping = aes(x=fit, y=output1+output2)) +
  geom_point(aes(colour=set), alpha=0.6) +
  geom_abline(slope=1, intercept = 0) +
  geom_smooth(method = "lm", se = FALSE, aes(colour=set), alpha=0.6)
dev.new()
print(ggp)