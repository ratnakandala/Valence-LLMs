# load required libraries (install if necessary)
library(readr)
library(lme4) 
library(lmerTest)
library(CorrMixed)
library(ggplot2)
library(polycor)
library(corrplot)
library(cocor)
library(rmcorr)
library(psych)

# read in data
d <- read.csv("chocollama-8B-instruct_valence_estimates_english_few_shot_prompt_entire_raw_text_LIWC_pattern.csv", fileEncoding='UTF-8-BOM')

# run Pearson correlations
corr1 <- cor.test(d$valence, d$polarity)
corr2 <- cor.test(d$valence, d$posemo)
corr3 <- cor.test(d$valence, d$negemo) #take abs
corr4 <- cor.test(d$polarity, d$negemo) #take abs
corr5 <- cor.test(d$valence, d$valence_score)
corr6 <- cor.test(d$polarity, d$valence_score)
corr7 <- cor.test(d$negemo, d$valence_score) #take abs

# compare correlation coefficients (https://www.rdocumentation.org/packages/psych/versions/2.5.3/topics/paired.r)
n <- nrow(d)
comp1 <- paired.r(corr1$estimate, abs(corr3$estimate), abs(corr4$estimate), n)
comp2 <- paired.r(corr1$estimate, corr5$estimate, corr6$estimate, n)


# compare repeated measures correlations (https://cran.r-project.org/web/packages/rmcorr/vignettes/compcor.html)
variables.overlap <- c("valence", "polarity", "negemo")
dist_rmc_mat_overlap <- rmcorr_mat(participant = connectionId, variables = variables.overlap, dataset = d, CI.level = 0.95)
model1.val.pol <- dist_rmc_mat_overlap$summary[1,]
model2.val.neg <- dist_rmc_mat_overlap$summary[2,]
model3.pol.neg <- dist_rmc_mat_overlap$summary[3,]
r.jk <- model1.val.pol$rmcorr.r
r.jh <- abs(model2.val.neg$rmcorr.r)
r.kh <- abs(model3.pol.neg$rmcorr.r)
n <- mean(dist_rmc_mat_overlap$summary$effective.N)
cocor.dep.groups.overlap(r.jk,r.jh,r.kh,n,alternative="two.sided",test="all",var.labels=variables.overlap)


# run polyserial correlations (https://search.r-project.org/CRAN/refmans/polycor/html/polyserial.html)
d <- subset(d, valence !="")
d <- d[order(d$valence_score), ]
pcorr5 <- polyserial(d$valence, d$valence_score)
pcorr6 <- polyserial(d$polarity, d$valence_score)
pcorr7 <- polyserial(d$negemo, d$valence_score) #take abs

# compare correlation coefficients
n <- nrow(d)
comp3 <- paired.r(pcorr1$estimate, pcorr5$estimate, pcorr6$estimate, n)


# run regression models (if I want to do this polyserial, then could degroup the data and run spearman...???)
m1 <- lmer(scale(valence) ~ scale(polarity) + (scale(polarity)|connectionId), data=d)
summary(m1)
m2 <- lmer(scale(valence) ~ scale(posemo) + (scale(posemo)|connectionId), data=d)
summary(m2)
m3 <- lmer(scale(valence) ~ scale(negemo) + (scale(negemo)|connectionId), data=d)
summary(m3)