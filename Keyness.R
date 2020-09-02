# Load Libraries
library(quanteda)
library(tidyverse)
library(tm, SnowballC)

# Corpus and DFM
corpus <- corpus(df, text_field = "Post_Lemma")
dfm_keyness <- dfm(corpus, groups=c("label"), verbose=TRUE)

# Keyness Analysis
help_keyness <- textstat_keyness(dfm_keyness, target="1",measure="chi2") 

# Keyness Plot
textplot_keyness(dfm_keyness, margin = .1, labelsize = 2, n=30, show_reference=T, show_legend = T, color = c("mediumseagreen", "indianred1"))