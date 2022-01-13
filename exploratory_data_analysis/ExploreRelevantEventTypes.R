library(tidyverse)
library(reshape2)
library(gridExtra)

# The purpose of this script is to assess which Event Types, if any, may be appropriate to filter out

results <- read_csv('resources/recent_olympic_data.csv')

# Remove gender and sport from event
results_cleaned <- results %>% mutate(event = str_remove(event, str_c(sport, " Men's |", sport, " Women's ")))

# How many Olympics Games is each event present in?
event_counts <- results_cleaned %>% select(year, season, sport, event) %>% group_by(season) %>%
    mutate(n_games_by_season = length(unique(year))) %>%
    group_by(season, sport, event) %>% 
    summarize(times_event_present = length(unique(year))) %>% distinct()

# How many Olympic Games is each sport present in and how many participants?
sport_counts <- results_cleaned %>% group_by(season) %>% 
    mutate(total_season_participants = n()) %>% 
    group_by(season, sport) %>%
    summarize(n_games_present               = n_distinct(year),
              n_participants                = n(),
              percent_participants_of_sport = unique(n_participants / total_season_participants * 100) 
              ) %>%
    arrange(n_games_present)

# How many events and participants are there per sport and how many years and participants are present by event?
summary_by_sport_df <- results_cleaned %>% group_by(season, sport) %>%
    mutate(n_events_sport_present = n_distinct(event),
           n_participants_sport   = n()) %>%
    group_by(season, sport, n_events_sport_present, n_participants_sport, event) %>%
    summarize(n_years_event_present  = n_distinct(year),
              n_participants_event   = n())

# What do the averages of the numeric columns look like for "Athletics"?
avg_bio_data_athletics <- results_cleaned %>% filter(sport == 'Athletics') %>% group_by(sex, event) %>%
    summarize(avg_age = mean(age, na.rm = T),
              avg_height = mean(height, na.rm = T),
              avg_weight = mean(weight, na.rm = T))

# Create "fill" vector to show avg_weight discrepancy in certain field events
fill = c()
for (ev in avg_bio_data_athletics$event) {
    
    if (ev %in% c('Discus Throw', 'Hammer Throw', 'Shot Put')) {
        fill = c(fill, 'firebrick')
    } else {
        fill = c(fill, 'forestgreen')
    }
    
}

# Plot avg_weight by event, faceted on sex
avg_bio_data_athletics %>% ggplot(aes(x=event, y=avg_weight)) +
    geom_col(color='black', fill=fill) +
    facet_wrap(~sex, ncol=1) +
    theme(axis.text.x = element_text(angle=60, vjust=.75))

# Create a list of ggplots looking at avg_weight per event within a given sport
sport_bio_plot <- list()
for (sp in unique(results_cleaned$sport)) {
    
    avg_bio_data_by_event <- results_cleaned %>% filter(sport == sp) %>% group_by(sex, event) %>%
        summarize(avg_age = mean(age, na.rm = T),
                  avg_height = mean(height, na.rm = T),
                  avg_weight = mean(weight, na.rm = T))
    
    sport_bio_plot[[sp]] <- avg_bio_data_by_event %>% ggplot(aes(x=event, y=avg_weight)) +
                                                      geom_col(color='black', fill='forestgreen') +
                                                      facet_wrap(~sex, ncol=1) +
                                                      labs(x='Event', y='Average Weight', title=str_c(sp, ' - Avg Weight by Event')) +
                                                      theme(axis.text.x = element_text(angle=90, vjust=.75), plot.title = element_text(hjust=.5, size=14))
    
}

# Save the plots to a PDF
ggsave(
    filename = 'exploratory_data_analysis/plots/sport_bio_plots.pdf',
    plot = marrangeGrob(sport_bio_plot, nrow=1, ncol=1)
)

# Remove Team Sports from the dataset
sports_to_remove <- c('Basketball', 'Handball', 'Water Polo', 'Hockey', 'Football', 'Softball',
                      'Volleyball', 'Baseball', 'Rugby Sevens', 'Beach Volleyball', 'Golf')

# Remove sports with less than 1% of the participants for its respective season
sports_to_remove <- c(sports_to_remove, 'Skeleton', 'Taekwondo', 'Trampolining', 'Triathlon',
                      'Curling', 'Badminton', 'Rhythmic Gymnastics', 'Synchronized Swimming',
                      'Modern Pentathlon')

# Summarize impact of sports cleanup
sports_clean_summary <- results_cleaned %>% group_by(season) %>% 
    mutate(n_part_old  = n(),
           n_sport_old = n_distinct(sport)) %>%
    filter(!sport %in% sports_to_remove) %>% group_by(season, n_part_old, n_sport_old) %>%
    summarize(n_part_new  = n(),
              n_sport_new = n_distinct(sport)) %>% 
    mutate(pct_part_removed = round((1 - n_part_new / n_part_old) * 100, 2))

# Execute sport cleanup and save to new df
results_cleaned_sports <- results_cleaned %>% filter(!sport %in% sports_to_remove)

# Extract data for sports with weight discrepancies to determine where we might want to section off a group of events into their own category
sports_w_weight_discrepancies <- c('Judo', 'Sailing', 'Athletics', 'Weightlifting', 'Wrestling',
                                   'Boxing', 'Canoeing', 'Cycling')
sports_w_weight_discrepancies_df <- results_cleaned_sports %>% filter(sport %in% sports_w_weight_discrepancies)


# Create a new list of ggplots looking at avg_weight per event within updated sports
sport_bio_boxplot <- list()
for (sp in unique(sports_w_weight_discrepancies_df$sport)) {
    
    avg_bio_data_by_event <- sports_w_weight_discrepancies_df %>% filter(sport == sp)
    
    sport_bio_boxplot[[sp]] <- avg_bio_data_by_event %>% ggplot(aes(x=event, y=weight)) +
        geom_boxplot() +
        facet_wrap(~sex, ncol=1, scales='free_y') +
        labs(x='Event', y='Average Weight', title=str_c(sp, ' - Weight by Event')) +
        theme(axis.text.x = element_text(angle=90, vjust=.75), plot.title = element_text(hjust=.5, size=14))
    
}

# Save the plots to a PDF
ggsave(
    filename = 'exploratory_data_analysis/plots/sport_bio_boxplots.pdf',
    plot = marrangeGrob(sport_bio_boxplot, nrow=1, ncol=1)
)

















