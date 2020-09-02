# Load Libraries
library(stm)
library(quanteda)
library(tidyverse)
library(tm, SnowballC)

# Corpus and DFM
vcorpus <- VCorpus(VectorSource(df$text_lemma))
vcorpus_nostop <- tm_map(vcorpus, removeWords, eng_stop)
corpus <- corpus(corpus_nostop)
dfm_stm <- dfm(corpus_stm, verbose=TRUE, remove_numbers=TRUE)

# Meta Variables
chi <- df$chi_vakue
help <- df$label
meta <- data.frame(chi=chi, help=help)

# Perform STM
stm <- stm(documents=dfm_stm, K=50, prevalence=~help+chi,
           data=meta, seed=123)

est <- estimateEffect(~help, stm,
                      uncertainty="None")

summary(est, topics=1)

# Topic Coefficients 
coef <- se <- rep(NA, 50)
for (i in 1:50){
  coef[i] <- est$parameters[[i]][[1]]$est[2]
  se[i] <- sqrt(est$parameters[[i]][[1]]$vcov[2,2])
}

df <- data.frame(topic = 1:50, coef=coef, se=se)
df <- df[order(df$coef, decreasing=T),] 