library(tidyverse)
library(reshape2)

# The purpose of this script is to assess which Event Types, if any, may be appropriate to filter out

results <- read.csv('../resources/athlete_events.csv')
regions <- read.csv('../resources/noc_regions.csv')

# Join CSVs on NOC and change column names to lowercase
historical_results <- left_join(results, regions, by = "NOC")
names(historical_results) <- tolower(names(historical_results))

# Drop unnecessary columns and filter to show only 1964 or later
historical_results <- historical_results %>% select(-c(name, team, games, notes)) %>% filter(year >= 1964)

# How many Olympics Games is each event present in?
event_counts <- historical_results %>% select(year, season, sport, event) %>% group_by(season) %>%
    mutate(n_games_by_season = length(unique(year))) %>%
    group_by(season, sport, event) %>% 
    summarize(times_event_present = length(unique(year))) %>% distinct()

event_histogram <- event_counts %>% ggplot(aes(x=times_event_present)) +
                                    geom_histogram(binwidth = 1, fill = 'grey57', color = 'black') +
                                    facet_wrap(~season, ncol=1, scales='free_y') +
                                    scale_x_continuous(breaks=c(0:14), minor_breaks = NULL) +
                                    stat_bin(aes(y=..count.., label=..count..), bins=14, geom = 'text', vjust=-.5) +
                                    labs(x='Years Present', y='Number of Events', title='Event Histogram by Years Present Since 1964') +
                                    theme(plot.title = element_text(hjust=.5, size=16))

# Let's start looking at events that have appeared less often. Maybe we can combine some events?
event_counts %>% filter(times_event_present == 1) %>% arrange(season, sport) %>% print(n=nrow(event_counts))

historical_results %>% filter(sport == 'Golf') %>% group_by(year) %>% distinct(year) # The entire "Golf" sport is only present in 2016
historical_results %>% filter(sport == 'Rowing') %>% group_by(year) %>% distinct(year, event)
