# Setup
library(tidyverse)
library(viridis)
library(knitr)
library(gridExtra)
library(lmerTest)
library(scales)

# Load data
file_path <- "TalkLab/data/Talklab_CANDOR_take_home_data.csv"

df <- read_csv(file_path) |>
  slice(-1) |>
  mutate(
    sequence_start = as.numeric(sequence_start),
    sequence_end = as.numeric(sequence_end),
    sequence_duration = as.numeric(sequence_duration),
    sequence_type = str_trim(sequence_type)
  ) |>
  arrange(interaction_id, sequence_start)

df <- df |>
  group_by(interaction_id) |>
  mutate(
    next_type = lead(sequence_type),
    next_speaker = lead(initiating_speaker),
    speaker_change = ifelse(initiating_speaker == next_speaker, "Same Speaker", "Switch Speaker")
  ) |>
  ungroup()


# Summary stats
kable(df |>
        group_by(interaction_id) |>
        summarize(
          total_Q = sum(sequence_type == "question"),
          total_D = sum(sequence_type == "disclosure")
        ) |>
        summarize(
          avg_Q = mean(total_Q),
          avg_D = mean(total_D),
          sd_Q = sd(total_Q),
          sd_D = sd(total_D)
        ))

counts_by_interaction <- df |>
  group_by(interaction_id) |>
  summarize(
    total_Q = sum(sequence_type == "question"),
    total_D = sum(sequence_type == "disclosure")
  )

counts_by_interaction |>
  ggplot(aes(x = total_Q, y = total_D)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal() +
  labs(
    x = "Total Questions",
    y = "Total Disclosures"
  )

# Sample rhythm visualization
set.seed(42)
sampled_ids <- sample(unique(df$interaction_id), 6)

df |>
  filter(interaction_id %in% sampled_ids) |>
  ggplot(aes(y = initiating_speaker, x = sequence_start, xend = sequence_end, color = sequence_type)) +
  geom_segment(linewidth = 4, alpha = 0.8) +
  scale_color_manual(values = c("question" = "blue", "disclosure" = "orange")) +
  facet_wrap(~ interaction_id, ncol = 3, scales = "free_x") +
  theme_minimal()

# Micro-structural transitions
trans_data <- df |>
  filter(!is.na(next_type) & !is.na(speaker_change)) |>
  count(sequence_type, speaker_change, next_type) |>
  group_by(sequence_type) |>
  mutate(prob = n / sum(n)) |>
  ungroup()

ggplot(trans_data, aes(x = sequence_type, y = prob, fill = next_type, alpha = speaker_change)) +
  geom_bar(stat = "identity", position = "fill", color = "black") +
  scale_alpha_manual(values = c("Same Speaker" = 0.7, "Switch Speaker" = 1.0)) +
  scale_fill_manual(values = c("question" = "blue", "disclosure" = "orange")) +
  geom_text(aes(label = ifelse(prob > 0.05, scales::percent(prob, accuracy = 1), "")),
            position = position_stack(vjust = 0.5), size = 4, color = "white") +
  theme_minimal()


# Temporal analysis: bin into phases
df_temporal <- df |>
  group_by(interaction_id) |>
  mutate(
    progress = (sequence_start - min(sequence_start)) /
      (max(sequence_end) - min(sequence_start)),
    phase = cut(progress, breaks = 5, labels = FALSE)
  ) |>
  group_by(interaction_id, phase) |>
  summarize(
    disclosure_ratio = mean(sequence_type == "disclosure"),
    .groups = "drop"
  )

ggplot(df_temporal, aes(x = factor(phase), y = disclosure_ratio)) +
  geom_line(aes(group = interaction_id), color = "gray", alpha = 0.2) +
  stat_summary(fun = mean, geom = "line", aes(group = 1), color = "orange", linewidth = 2) +
  stat_summary(fun = mean, geom = "point", color = "orange", size = 4) +
  theme_minimal()


# Temporal analysis by speaker
df_temporal_speaker <- df |>
  group_by(interaction_id) |>
  mutate(
    progress = (sequence_start - min(sequence_start)) /
      (max(sequence_end) - min(sequence_start)),
    phase = cut(progress, breaks = 5, labels = FALSE)
  ) |>
  group_by(interaction_id, initiating_speaker, phase) |>
  summarize(
    disclosure_ratio = mean(sequence_type == "disclosure"),
    .groups = "drop"
  )

speaker_slopes <- df_temporal_speaker |>
  group_by(interaction_id, initiating_speaker) |>
  summarize(
    escalation_slope = coef(lm(disclosure_ratio ~ phase))[2],
    avg_disclosure = mean(disclosure_ratio),
    .groups = "drop"
  )

sample_ids <- sample(unique(df$interaction_id), 6)

df_temporal_speaker |>
  filter(interaction_id %in% sample_ids) |>
  ggplot(aes(x = factor(phase), y = disclosure_ratio, color = initiating_speaker, group = initiating_speaker)) +
  geom_line(linewidth = 1.5, alpha = 0.8) +
  facet_wrap(~interaction_id) +
  scale_color_manual(values = c("left" = "darkgreen", "right" = "maroon")) +
  theme_minimal()


# Models (not evaluated)
# model_h1 <- lmer(partner_rating ~ reciprocity_score + responsiveness_score +
#                    (1 | interaction_id), data = speaker_level_data)
