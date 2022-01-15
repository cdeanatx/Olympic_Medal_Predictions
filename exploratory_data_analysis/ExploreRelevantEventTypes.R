library(tidyverse)
library(reshape2)
library(gridExtra)

# The purpose of this script is to assess which Event Types, if any, may be appropriate to filter out

results <- read_csv('resources/recent_olympic_data.csv')

# Remove gender and sport from event
results_cleaned <- results %>% mutate(event = str_remove(event, str_c(sport, " Men's |", sport, " Women's |", sport, " Mixed ")))

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
for (sp in sports_w_weight_discrepancies) {
    
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

# Create dfs showing summary statistics for each sport
sports_weight_analysis_female <- list()
sports_weight_analysis_male <- list()
for (sp in sports_w_weight_discrepancies) {
    
    sports_weight_analysis_female[[sp]] <- sports_w_weight_discrepancies_df %>% filter(sport == sp, sex == 'F') %>% group_by(event) %>%
        summarize(n_part       = n(),
                  min_weight   = min(weight, na.rm = T),
                  Q1           = quantile(weight, probs = .25, na.rm = T),
                  med_weight   = median(weight, na.rm = T),
                  avg_weight   = mean(weight, na.rm = T),
                  Q3           = quantile(weight, probs = .75, na.rm = T),
                  max_weight   = max(weight, na.rm = T),
                  missing_vals = sum(is.na(weight))
        ) %>% arrange(desc(med_weight))
    
    sports_weight_analysis_male[[sp]] <- sports_w_weight_discrepancies_df %>% filter(sport == sp, sex == 'M') %>% group_by(event) %>%
        summarize(n_part       = n(),
                  min_weight   = min(weight, na.rm = T),
                  Q1           = quantile(weight, probs = .25, na.rm = T),
                  med_weight   = median(weight, na.rm = T),
                  avg_weight   = mean(weight, na.rm = T),
                  Q3           = quantile(weight, probs = .75, na.rm = T),
                  max_weight   = max(weight, na.rm = T),
                  missing_vals = sum(is.na(weight))
        ) %>% arrange(desc(med_weight))
}

# Let's look at the last time each event was held for sports that need more granular categorization
last_held <- results_cleaned_sports %>%
    filter(sport %in% c('Athletics', 'Boxing', 'Canoeing', 'Cycling', 'Judo', 'Sailing', 'Weightlifting', 'Wrestling')) %>% 
    group_by(season, sport, event) %>%
    summarize(last_year_held = max(year)) %>%
    arrange(sport, last_year_held)

# Create a nested list where first level is sport and second level is event category, which contains vector of events pertaining to that category
event_categories <- list(
    
    # Events are categorized based on weight of participants and "type" of event, e.g. "Track" or "Field"
    athletics = list(
        
        # These events are no longer held
        drop_events           = c('80 metres Hurdles', 'Pentathlon', '3,000 metres', '10 kilometres Walk'),
        heavyset_field_events = c('Shot Put', 'Discus Throw', 'Hammer Throw', 'Javelin Throw'),
        field_events          = c('Heptathlon', 'High Jump', 'Triple Jump', 'Long Jump', 'Pole Vault', 'Decathlon'),
        track                 = c('1,500 metres', '10,000 metres', '100 metres', '100 metres Hurdles', '110 metres Hurdles',
                                  '20 kilometres Walk', '200 metres', '3,000 metres Steeplechase', '4 x 100 metres Relay',
                                  '4 x 400 metres Relay', '400 metres', '400 metres Hurdles', '5,000 metres',
                                  '50 kilometres Walk', '800 metres', 'Marathon')
        
    ),
    
    # Events are categorized based on weight of participants.
    # Events no longer held can be rolled up into a weight category because they are similar enough
    boxing = list(
        
        flyweight    = c('Light-Flyweight', 'Flyweight'), # 3 kg variance in median
        lightweight  = c('Bantamweight', 'Featherweight', 'Lightweight'), # 6 kg variance in median
        welterweight = c('Light-Welterweight', 'Welterweight', 'Light-Middleweight'), # 7 kg variance in median
        middleweight = c('Middleweight', 'Light-Heavyweight'), # 6 kg variance in median
        heavyweight  = c('Heavyweight', 'Super-Heavyweight') # 7 kg variance in median
        
    ),
    
    # Events are categorized based on "type"/length of race. This resulted in loosely correlated median weights
    canoeing = list(
        
        # Not enough data, only present for 2 olympic games
        drop_events        = c('Kayak Singles, 200 metres', 'Canadian Singles, 200 metres', 'Kayak Doubles, 200 metres'),
        slalom             = c('Kayak Singles, Slalom', 'Canadian Doubles, Slalom', 'Canadian Singles, Slalom'),
        straight_race      = c('Kayak Singles, 500 metres', 'Kayak Doubles, 500 metres', 'Canadian Singles, 500 metres',
                               'Canadian Doubles, 500 metres', 'Kayak Fours, 500 metres', 'Kayak Singles, 1,000 metres',
                               'Kayak Fours, 1,000 metres', 'Canadian Singles, 1,000 metres',
                               'Kayak Doubles, 1,000 metres', 'Canadian Doubles, 1,000 metres')

    ),
    
    # Events are categorized based on "type". This resulted in loosely correlated median weights
    cycling = list(
        
        # The first 3 events have not been held recently
        # The last 4 events are new and have large variance in median weight from other cycling events
        drop_events   = c('Tandem Sprint, 2,000 metres', '1,000 metres Time Trial', '500 metres Time Trial',
                          'Keirin', 'Team Sprint', 'BMX', 'Mountainbike, Cross-Country'),
        road_cycling  = c('Road Race, Individual', 'Individual Time Trial', '100 kilometres Team Time Trial'),
        track_cycling = c('Points Race', 'Madison', 'Team Pursuit, 4,000 metres', 'Omnium', 'Individual Pursuit, 3,000 metres',
                          'Individual Pursuit, 4,000 metres', 'Sprint', 'Team Pursuit')
        
    ),
    
    # Events are categorized based on weight of participants.
    # Events no longer held can be rolled up into a weight category because they are similar enough
    judo = list(
        
        half_lightweight = c('Extra-Lightweight', 'Half-Lightweight'), # 6 kg variance in median weights
        lightweight  = c('Lightweight', 'Half-Middleweight'), # 8 kg variance in median weights
        middleweight = c('Middleweight'),
        half_heavyweight = c('Half-Heavyweight', 'Open Class'),
        heavyweight  = c('Heavyweight')
        
    ),
    
    
    sailing = list(
        
        # Most of these events are no longer held. A couple are drastically different in median weight.
        drop_events = c('5.5 metres', 'Two Person Heavyweight Dinghy', 'One Person Heavyweight Dinghy',
                              'Two Person Keelboat', 'Three Person Keelboat', 'One Person Dinghy'),
        sailing     = c('Skiff', 'Multihull', 'Windsurfer', 'Two Person Dinghy')
        
    ),
    
    weightlifting = list(
        
        featherweight            = c('Flyweight', 'Bantamweight', 'Featherweight'), # 9 kg variance in median weights
        light_middleweight       = c('Lightweight', 'Middleweight'), # 6.5 kg variance in median weights
        heavy_middleweight       = c('Light-Heavyweight', 'Middle-Heavyweight'), # 7 kg variance in median weights
        heavyweight              = c('Heavyweight I', 'Heavyweight II', 'Heavyweight'), # 8 kg variance in median weights
        super_heavyweight        = c('Super-Heavyweight')
        
    ),
    
    wrestling = list(
        
        flyweight         = c('Light-Flyweight, Greco-Roman', 'Light-Flyweight, Freestyle', 'Flyweight, Greco-Roman',
                              'Flyweight, Freestyle'), # 4 kg variance in median weights
        featherweight     = c('Bantamweight, Greco-Roman', 'Bantamweight, Freestyle', 'Featherweight, Greco-Roman',
                              'Featherweight, Freestyle'), # 5 kg variance in median weights
        welterweight      = c('Lightweight, Greco-Roman', 'Lightweight, Freestyle', 'Welterweight, Greco-Roman',
                              'Welterweight, Freestyle'), # 6 kg variance in median weights
        middleweight      = c('Middleweight, Greco-Roman', 'Middleweight, Freestyle', 'Light-Heavyweight, Greco-Roman',
                              'Light-Heavyweight, Freestyle'), # 8 kg variance in median weights
        heavyweight       = c('Heavyweight, Greco-Roman', 'Heavyweight, Freestyle'), # 3 kg variance in median weights
        super_heavyweight = c('Super-Heavyweight, Greco-Roman', 'Super-Heavyweight, Freestyle')
        
    )

)

# Loop through each row of the cleaned participants df and
# determine an updated event category that will be used in place of "sport" in the machine learning model
event_cat = c()
for (row in 1:nrow(results_cleaned_sports)) {
    
    this_sport <- tolower(gsub(' ', '_', results_cleaned_sports$sport[row]))
    this_event <- results_cleaned_sports$event[row]
    
    if (is.null(event_categories[[this_sport]])) {
        event_cat <- c(event_cat, this_sport)
    } 
    else {
        event_found = F
        for (cat in names(event_categories[[this_sport]])) {
            
            if (this_event %in% event_categories[[this_sport]][[cat]]) {
                event_found = T
                event_cat <-  c(event_cat, cat)
            }
        }
        
        if (!event_found) {
            print(c(row, this_sport, this_event))
        }
    }
}

# Add event category as a column to the cleaned participant df
clean_participant_df <- results_cleaned_sports %>%
    mutate(event_category = event_cat) %>%
    filter(event_category != 'drop_events') %>%
    relocate(id:sport, event_category)

# Concatenate sport to the event category for sports with weight-based categories
concat_sports = c('Boxing', 'Judo', 'Weightlifting', 'Wrestling')
clean_participant_df <- clean_participant_df %>%
    mutate(event_category = ifelse(sport %in% concat_sports, str_c(tolower(sport), '_', event_category), event_category))

# Save finalized participant dataset to CSV
write_csv(clean_participant_df, 'resources/cleaned_olympic_data.csv')
