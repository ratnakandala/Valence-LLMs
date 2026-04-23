# Valence Estimation Analysis
# ============================================================
# This script computes correlations between user self-reported valence ratings
# and valence estimates from three LLMs (ChocoLlama, Reynaerde, GEITje),
# as well as LIWC and Pattern lexicon-based tools.
# Presented at NeurIPS.


# ── Install packages (run once) ───────────────────────────────────────────────
# install.packages(c("tidyverse", "psych", "corrplot", "vroom", "lme4",
#                    "lmerTest", "CorrMixed", "polycor", "cocor", "rmcorr",
#                    "tidyr", "Rfast"))


# ── Load libraries ────────────────────────────────────────────────────────────
library(tidyverse)   # Data manipulation and visualisation
library(psych)       # Correlation tests (corr.test)
library(vroom)       # Fast CSV reading
library(corrplot)    # Correlation plots
library(readr)       # CSV reading utilities
library(lme4)        # Linear mixed-effects models
library(lmerTest)    # P-values for mixed models
library(CorrMixed)   # Correlations in mixed models
library(polycor)     # Polyserial correlation
library(cocor)       # Comparison of correlations
library(rmcorr)      # Repeated measures correlation
library(tidyr)       # Data tidying
library(Rfast)       # Fast statistical functions (poly.cor)


# ── Load data ─────────────────────────────────────────────────────────────────
# Update these paths to point to your local CSV files

LIWC_Pattern_estimates <- vroom("path/to/LIWC_Pattern_estimates.csv", col_names = TRUE, show_col_types = FALSE)
Chocollama_estimates   <- vroom("path/to/Chocollama_estimates.csv",   col_names = TRUE, show_col_types = FALSE)
Reynaerde_estimates    <- vroom("path/to/Reynaerde_estimates.csv",    col_names = TRUE, show_col_types = FALSE)
GEITje_estimates       <- vroom("path/to/GEITje_estimates.csv",       col_names = TRUE, show_col_types = FALSE)


# ── Pearson correlations: User valence vs LIWC and Pattern ────────────────────
# Pearson correlation is appropriate here as both variables are continuous.

# User valence vs LIWC Posemo (positive emotion)
corr.test(LIWC_Pattern_estimates$valence, LIWC_Pattern_estimates$posemo)

# User valence vs LIWC Negemo (negative emotion)
corr.test(LIWC_Pattern_estimates$valence, LIWC_Pattern_estimates$negemo)

# User valence vs Pattern polarity
corr.test(LIWC_Pattern_estimates$valence, LIWC_Pattern_estimates$polarity)


# ── Pearson correlation: User valence vs ChocoLlama ───────────────────────────
# Note: Pearson may not be fully appropriate here since LLM ratings are discrete
# (see polyserial section below for a more appropriate measure).

corr.test(Chocollama_estimates$valence, Chocollama_estimates$`llama3-chocollama-8B-instruct-valences-english`)


# ── Polyserial correlations: User valence vs LLMs ────────────────────────────
# Polyserial correlation is appropriate when one variable is continuous (user valence,
# ranging from -50 to +50) and the other is discrete/ordinal (LLM ratings, 1 to 7).


# User valence vs ChocoLlama
polyserial(Chocollama_estimates$valence, Chocollama_estimates$`llama3-chocollama-8B-instruct-valences-english`)

# Remove rows with missing values before computing p-value
df1 <- na.omit(Chocollama_estimates[, c("valence", "llama3-chocollama-8B-instruct-valences-english")])
poly.cor(df1$valence, df1$`llama3-chocollama-8B-instruct-valences-english`)


# User valence vs Reynaerde
polyserial(Reynaerde_estimates$valence, Reynaerde_estimates[["llama3-Reynaerde-7B-chat-valences-english"]])

dfRey <- na.omit(Reynaerde_estimates[, c("valence", "llama3-Reynaerde-7B-chat-valences-english")])
poly.cor(dfRey$valence, dfRey$`llama3-Reynaerde-7B-chat-valences-english`)


# User valence vs GEITje
polyserial(GEITje_estimates$valence, GEITje_estimates[["geitje-7B-ultra-valences-english"]])

dfGEIT <- na.omit(GEITje_estimates[, c("valence", "geitje-7B-ultra-valences-english")])
poly.cor(dfGEIT$valence, dfGEIT$`geitje-7B-ultra-valences-english`)


# ── Polyserial correlations: LIWC/Pattern vs LLMs ────────────────────────────
# LIWC and Pattern outputs are continuous; LLM ratings are ordinal.
# Datasets are merged on timeStampStart to align rows before correlating.


# ── LIWC/Pattern vs GEITje ────────────────────────────────────────────────────

# Ensure timeStampStart is character type for consistent merging
LIWC_Pattern_estimates$timeStampStart <- as.character(LIWC_Pattern_estimates$timeStampStart)
GEITje_estimates$timeStampStart       <- as.character(GEITje_estimates$timeStampStart)

# Drop rows where GEITje valence estimate is missing
GEITje_clean <- GEITje_estimates[!is.na(GEITje_estimates[["geitje-7B-ultra-valences-english"]]), ]

# Merge LIWC/Pattern and GEITje data on timeStampStart
merged_GEITje <- merge(
  LIWC_Pattern_estimates[, c("timeStampStart", "posemo", "negemo", "polarity")],
  GEITje_clean[, c("timeStampStart", "geitje-7B-ultra-valences-english")],
  by = "timeStampStart"
)

# Convert GEITje valence to ordered factor for polyserial correlation
merged_GEITje$geitje_ord <- as.ordered(merged_GEITje[["geitje-7B-ultra-valences-english"]])

# Polyserial: LIWC Posemo vs GEITje
polyserial(as.numeric(merged_GEITje$posemo), merged_GEITje$geitje_ord)
df <- na.omit(merged_GEITje[, c("posemo", "geitje-7B-ultra-valences-english")])
poly.cor(as.numeric(df$posemo), as.integer(factor(df[["geitje-7B-ultra-valences-english"]], ordered = TRUE)))

# Polyserial: LIWC Negemo vs GEITje
polyserial(as.numeric(merged_GEITje$negemo), merged_GEITje$geitje_ord)
df <- na.omit(merged_GEITje[, c("negemo", "geitje-7B-ultra-valences-english")])
poly.cor(as.numeric(df$negemo), as.integer(factor(df[["geitje-7B-ultra-valences-english"]], ordered = TRUE)))

# Polyserial: Pattern polarity vs GEITje
polyserial(as.numeric(merged_GEITje$polarity), merged_GEITje$geitje_ord)
df <- na.omit(merged_GEITje[, c("polarity", "geitje-7B-ultra-valences-english")])
poly.cor(as.numeric(df$polarity), as.integer(factor(df[["geitje-7B-ultra-valences-english"]], ordered = TRUE)))


# ── LIWC/Pattern vs Reynaerde ─────────────────────────────────────────────────

Reynaerde_estimates$timeStampStart <- as.character(Reynaerde_estimates$timeStampStart)

# Drop rows where Reynaerde valence estimate is missing
Reynaerde_clean <- Reynaerde_estimates[!is.na(Reynaerde_estimates[["llama3-Reynaerde-7B-chat-valences-english"]]), ]

# Merge LIWC/Pattern and Reynaerde data on timeStampStart
merged_Reynaerde <- merge(
  LIWC_Pattern_estimates[, c("timeStampStart", "posemo", "negemo", "polarity")],
  Reynaerde_clean[, c("timeStampStart", "llama3-Reynaerde-7B-chat-valences-english")],
  by = "timeStampStart"
)

# Convert Reynaerde valence to ordered factor
merged_Reynaerde$reynaerde_ord <- as.ordered(merged_Reynaerde[["llama3-Reynaerde-7B-chat-valences-english"]])

# Polyserial: LIWC Posemo vs Reynaerde
polyserial(as.numeric(merged_Reynaerde$posemo), merged_Reynaerde$reynaerde_ord)
df <- na.omit(merged_Reynaerde[, c("posemo", "llama3-Reynaerde-7B-chat-valences-english")])
poly.cor(as.numeric(df$posemo), as.integer(factor(df[["llama3-Reynaerde-7B-chat-valences-english"]], ordered = TRUE)))

# Polyserial: LIWC Negemo vs Reynaerde
polyserial(as.numeric(merged_Reynaerde$negemo), merged_Reynaerde$reynaerde_ord)
df <- na.omit(merged_Reynaerde[, c("negemo", "llama3-Reynaerde-7B-chat-valences-english")])
poly.cor(as.numeric(df$negemo), as.integer(factor(df[["llama3-Reynaerde-7B-chat-valences-english"]], ordered = TRUE)))

# Polyserial: Pattern polarity vs Reynaerde
polyserial(as.numeric(merged_Reynaerde$polarity), merged_Reynaerde$reynaerde_ord)
df <- na.omit(merged_Reynaerde[, c("polarity", "llama3-Reynaerde-7B-chat-valences-english")])
poly.cor(as.numeric(df$polarity), as.integer(factor(df[["llama3-Reynaerde-7B-chat-valences-english"]], ordered = TRUE)))


# ── LIWC/Pattern vs ChocoLlama ────────────────────────────────────────────────

# Ensure regex_output is character type for consistent merging
LIWC_Pattern_estimates$regex_output <- as.character(LIWC_Pattern_estimates$regex_output)
Chocollama_estimates$regex_output   <- as.character(Chocollama_estimates$regex_output)

# Drop rows where ChocoLlama valence estimate is missing
Chocollama_clean <- Chocollama_estimates[!is.na(Chocollama_estimates[["llama3-chocollama-8B-instruct-valences-english"]]), ]

# Merge LIWC/Pattern and ChocoLlama data on regex_output (text column)
merged_Chocollama <- merge(
  LIWC_Pattern_estimates[, c("regex_output", "posemo", "negemo", "polarity")],
  Chocollama_clean[, c("regex_output", "llama3-chocollama-8B-instruct-valences-english")],
  by = "regex_output"
)
merged_Chocollama <- merged_Chocollama[complete.cases(merged_Chocollama), ]

# Convert ChocoLlama valence to ordered factor
merged_Chocollama$chocollama_ord <- as.ordered(merged_Chocollama[["llama3-chocollama-8B-instruct-valences-english"]])

# Polyserial: LIWC Posemo vs ChocoLlama
polyserial(as.numeric(merged_Chocollama$posemo), merged_Chocollama$chocollama_ord)
df <- na.omit(merged_Chocollama[, c("posemo", "llama3-chocollama-8B-instruct-valences-english")])
poly.cor(as.numeric(df$posemo), as.integer(factor(df[["llama3-chocollama-8B-instruct-valences-english"]], ordered = TRUE)))

# Polyserial: LIWC Negemo vs ChocoLlama
polyserial(as.numeric(merged_Chocollama$negemo), merged_Chocollama$chocollama_ord)
df <- na.omit(merged_Chocollama[, c("negemo", "llama3-chocollama-8B-instruct-valences-english")])
poly.cor(as.numeric(df$negemo), as.integer(factor(df[["llama3-chocollama-8B-instruct-valences-english"]], ordered = TRUE)))

# Polyserial: Pattern polarity vs ChocoLlama
polyserial(as.numeric(merged_Chocollama$polarity), merged_Chocollama$chocollama_ord)
df <- na.omit(merged_Chocollama[, c("polarity", "llama3-chocollama-8B-instruct-valences-english")])
poly.cor(as.numeric(df$polarity), as.integer(factor(df[["llama3-chocollama-8B-instruct-valences-english"]], ordered = TRUE)))


# ── Distributions ─────────────────────────────────────────────────────────────

# User self-reported valence distribution (continuous, -50 to +50)
Chocollama_estimates %>%
  ggplot(aes(x = valence)) +
  geom_histogram(aes(y = ..density..), color = "black", fill = "lightblue") +
  geom_density() +
  scale_x_continuous(breaks = seq(-50, 50, by = 5), limits = c(-50, 50), expand = expansion(add = 0.5)) +
  theme_minimal() +
  labs(title = "Distribution of Users' Self-Reported Valence Ratings", x = "Users' Valence Ratings") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("Distribution_Users_Valence.png", width = 4800, height = 900, units = "px", bg = "white")

# LIWC Posemo and Negemo distributions overlaid
LIWC_Pattern_estimates %>%
  ggplot() +
  geom_histogram(aes(x = posemo, y = ..density.., fill = factor("Posemo")), color = "black", binwidth = 1, alpha = 0.5) +
  geom_density(aes(x = posemo, color = factor("Posemo")), size = 1) +
  geom_histogram(aes(x = negemo, y = ..density.., fill = factor("Negemo")), color = "black", binwidth = 1, alpha = 0.5) +
  geom_density(aes(x = negemo, color = factor("Negemo")), size = 1) +
  scale_fill_manual(name = "Emotion Type", values = c("Posemo" = "#0072B2", "Negemo" = "#E69F00")) +
  scale_color_manual(name = "Emotion Type", values = c("Posemo" = "#0072B2", "Negemo" = "#E69F00")) +
  scale_x_continuous(breaks = seq(0, 50, by = 2), limits = c(0, 30)) +
  scale_y_continuous(limits = c(0, 0.4)) +
  theme_minimal() +
  labs(title = "Distribution of LIWC's Valence Ratings", x = "LIWC's ratings", y = "Density") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = "top")
ggsave("Distribution_LIWC_Valence.png", width = 2000, height = 800, units = "px", dpi = 300, bg = "white")

# Pattern polarity distribution
LIWC_Pattern_estimates %>%
  ggplot(aes(x = polarity)) +
  geom_histogram(aes(y = ..density..), color = "black", fill = "lightblue") +
  geom_density() +
  theme_minimal() +
  labs(title = "Distribution of Pattern's Valence Ratings", x = "Pattern Polarity") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("Distribution_Pattern_Valence.png", width = 2000, height = 800, units = "px", bg = "white")


# ── LLM predicted valence distributions ──────────────────────────────────────

# ChocoLlama predicted valence distribution (discrete bar chart)
Chocollama_estimates %>%
  ggplot(aes(x = `llama3-chocollama-8B-instruct-valences-english`)) +
  geom_bar(fill = "lightblue", color = "black", size = 0.5) +
  scale_x_continuous(
    breaks = seq(
      floor(min(Chocollama_estimates$`llama3-chocollama-8B-instruct-valences-english`, na.rm = TRUE)),
      ceiling(max(Chocollama_estimates$`llama3-chocollama-8B-instruct-valences-english`, na.rm = TRUE)),
      by = 1
    ),
    expand = expansion(add = 0.5)
  ) +
  theme_minimal() +
  labs(title = "Distribution of ChocoLlama-8B Predicted Valence Scores", x = "Predicted Valence", y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("Distribution_Chocollama_Valence.png", width = 2000, height = 800, units = "px", bg = "white")

# Reynaerde predicted valence distribution
Reynaerde_estimates %>%
  ggplot(aes(x = `llama3-Reynaerde-7B-chat-valences-english`)) +
  geom_bar(fill = "lightblue", color = "black", size = 0.5) +
  scale_x_continuous(
    breaks = seq(
      floor(min(Reynaerde_estimates[["llama3-Reynaerde-7B-chat-valences-english"]], na.rm = TRUE)),
      ceiling(max(Reynaerde_estimates[["llama3-Reynaerde-7B-chat-valences-english"]], na.rm = TRUE)),
      by = 1
    ),
    expand = expansion(add = 0.5)
  ) +
  theme_minimal() +
  labs(title = "Distribution of Reynaerde-7B Predicted Valence Scores", x = "Predicted Valence", y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("Distribution_Reynaerde_Valence.png", width = 2000, height = 800, units = "px", bg = "white")

# GEITje predicted valence distribution
GEITje_estimates %>%
  ggplot(aes(x = `geitje-7B-ultra-valences-english`)) +
  geom_bar(fill = "lightblue", color = "black", size = 0.5) +
  scale_x_continuous(
    breaks = seq(
      floor(min(GEITje_estimates[["geitje-7B-ultra-valences-english"]], na.rm = TRUE)),
      ceiling(max(GEITje_estimates[["geitje-7B-ultra-valences-english"]], na.rm = TRUE)),
      by = 1
    ),
    expand = expansion(add = 0.5)
  ) +
  theme_minimal() +
  labs(title = "Distribution of GEITje-7B Predicted Valence Scores", x = "Predicted Valence", y = "Count") +
  theme(plot.title = element_text(hjust = 0.5))
ggsave("Distribution_GEITje_Valence.png", width = 2000, height = 800, units = "px", bg = "white")


# ── Scatter plots: User valence vs LLM predictions (linear fit) ──────────────

# User valence vs ChocoLlama — linear fit with jittered points
Chocollama_estimates %>%
  filter(is.finite(`llama3-chocollama-8B-instruct-valences-english`), is.finite(valence)) %>%
  ggplot(aes(x = `llama3-chocollama-8B-instruct-valences-english`, y = valence)) +
  geom_smooth(method = "lm", color = "#0072B2", se = TRUE) +
  geom_jitter(width = 0.25, alpha = 0.1, color = "black") +
  theme_minimal() +
  labs(title = "User vs. ChocoLlama-8B Predictions (Linear Fit)", x = "ChocoLlama-8B Predicted Valence", y = "User Valence Rating")
ggsave("Chocollama_Valence_LinearFit.png", width = 2000, height = 800, units = "px", dpi = 300, bg = "white")

# User valence vs Reynaerde — linear fit with jittered points
Reynaerde_estimates %>%
  filter(is.finite(`llama3-Reynaerde-7B-chat-valences-english`), is.finite(valence)) %>%
  ggplot(aes(x = `llama3-Reynaerde-7B-chat-valences-english`, y = valence)) +
  geom_smooth(method = "lm", color = "#0072B2", se = TRUE) +
  geom_jitter(width = 0.25, alpha = 0.1, color = "black") +
  theme_minimal() +
  labs(title = "User vs. Reynaerde-7B Predictions (Linear Fit)", x = "Reynaerde-7B Predicted Valence", y = "User Valence Rating")
ggsave("Reynaerde_Valence_LinearFit.png", width = 2000, height = 800, units = "px", dpi = 300, bg = "white")

# User valence vs GEITje — linear fit with jittered points
GEITje_estimates %>%
  filter(is.finite(`geitje-7B-ultra-valences-english`), is.finite(valence)) %>%
  ggplot(aes(x = `geitje-7B-ultra-valences-english`, y = valence)) +
  geom_smooth(method = "lm", color = "#0072B2", se = TRUE) +
  geom_jitter(width = 0.25, alpha = 0.1, color = "black") +
  theme_minimal() +
  labs(title = "User vs. GEITje-7B Predictions (Linear Fit)", x = "GEITje-7B Predicted Valence", y = "User Valence Rating")
ggsave("GEITje_Valence_LinearFit.png", width = 2000, height = 800, units = "px", dpi = 300, bg = "white")


# ── Boxplots: User valence distribution by LLM predicted score ───────────────

# Each box shows the spread of user valence ratings for a given LLM score bin

# ChocoLlama boxplot
Chocollama_estimates %>%
  filter(is.finite(`llama3-chocollama-8B-instruct-valences-english`), is.finite(valence)) %>%
  ggplot(aes(x = as.factor(`llama3-chocollama-8B-instruct-valences-english`), y = valence)) +
  geom_boxplot(fill = "lightblue", color = "black", outlier.alpha = 0.2) +
  theme_minimal() +
  labs(title = "User Valence by ChocoLlama-8B Predicted Score", x = "ChocoLlama-8B Predicted Valence (Binned)", y = "User Valence Rating")
ggsave("Chocollama_Valence_Boxplot.png", width = 2000, height = 800, units = "px", dpi = 300, bg = "white")

# Reynaerde boxplot
Reynaerde_estimates %>%
  filter(is.finite(`llama3-Reynaerde-7B-chat-valences-english`), is.finite(valence)) %>%
  ggplot(aes(x = as.factor(`llama3-Reynaerde-7B-chat-valences-english`), y = valence)) +
  geom_boxplot(fill = "lightblue", color = "black", outlier.alpha = 0.2) +
  theme_minimal() +
  labs(title = "User Valence by Reynaerde-7B Predicted Score", x = "Reynaerde-7B Predicted Valence (Binned)", y = "User Valence Rating")
ggsave("Reynaerde_Valence_Boxplot.png", width = 2000, height = 800, units = "px", dpi = 300, bg = "white")

# GEITje boxplot
GEITje_estimates %>%
  filter(is.finite(`geitje-7B-ultra-valences-english`), is.finite(valence)) %>%
  ggplot(aes(x = as.factor(`geitje-7B-ultra-valences-english`), y = valence)) +
  geom_boxplot(fill = "lightblue", color = "black", outlier.alpha = 0.2) +
  theme_minimal() +
  labs(title = "User Valence by GEITje-7B Predicted Score", x = "GEITje-7B Predicted Valence (Binned)", y = "User Valence Rating")
ggsave("GEITje_Valence_Boxplot.png", width = 2000, height = 800, units = "px", dpi = 300, bg = "white")


# ── Linear mixed-effects model ────────────────────────────────────────────────
# Predicts user valence from Pattern polarity, accounting for repeated measures
# per participant (connectionId) with a random slope for polarity.

m1 <- lmer(scale(valence) ~ scale(polarity) + (scale(polarity) | connectionId), data = LIWC_Pattern_estimates)
summary(m1)
