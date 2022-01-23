library(tidyverse)
library(reshape2)

results <- read.csv('../resources/athlete_events.csv')
regions <- read.csv('../resources/noc_regions.csv')

# Join CSVs on NOC and change column names to lowercase
historical_results <- left_join(results, regions, by = "NOC")
names(historical_results) <- tolower(names(historical_results))

# How often are "notes" present?
sum(historical_results$notes %in% c('', NA)) # 266,077 are empty or missing out of 271,116

# Drop unnecessary columns
historical_results <- historical_results %>% select(-c(name, team, games, notes))

# Get counts and percentages of biometric NA values by year and season
biometric_na_by_olympics <- historical_results %>% group_by(season) %>%
    mutate(n_records_season       = n(),
           age_na_season_count    = sum(is.na(age)),
           height_na_season_count = sum(is.na(height)),
           weight_na_season_count = sum(is.na(weight))
           ) %>%
    group_by(season, year) %>% 
    summarize(
        n_records_season          = n_records_season,
        age_na_season_count       = age_na_season_count,
        height_na_season_count    = height_na_season_count,
        weight_na_season_count    = weight_na_season_count,
        n_records_season_year     = n(),
        percent_records           = round(n_records_season_year / n_records_season * 100, 2),
        age_na_count              = sum(is.na(age)),
        age_percent_na_year       = round(age_na_count / n_records_season_year * 100, 2),
        age_percent_na_season     = round(age_na_count / age_na_season_count * 100, 2),
        height_na_count           = sum(is.na(height)),
        height_percent_na_year    = round(height_na_count / n_records_season_year * 100, 2),
        height_percent_na_season  = round(height_na_count / height_na_season_count * 100, 2),
        weight_na_count           = sum(is.na(weight)),
        weight_percent_na_year    = round(weight_na_count / n_records_season_year * 100, 2),
        weight_percent_na_season  = round(weight_na_count / weight_na_season_count * 100, 2)
        ) %>% 
    arrange(season, year) %>% distinct() %>% ungroup()


# Prepare data for plotting by converting into long format
biometric_na_by_olympics_long <- melt(biometric_na_by_olympics[,c('year', 'season', 'age_percent_na_year', 'height_percent_na_year', 'weight_percent_na_year')], id.vars=c(1,2)) %>% filter(year <= 1992)

# Plot the data
plot_na_over_time <- biometric_na_by_olympics_long %>% ggplot(aes(x=factor(year), y=value)) +
                       geom_bar(aes(fill=variable), stat='identity', color='black', position='dodge') +
                       facet_wrap(~season, ncol=1) +
                       labs(x='Year', y='Percent Athletes w/ NA', title='% of NA in Biometric Data by Year and Season') +
                       theme(axis.text.x = element_text(angle=45), plot.title = element_text(hjust=.5, size=16), strip.text = element_text(size=14), axis.title = element_text(size=14))

# Get running NA percentages for summer
running_na_percent_summer <- biometric_na_by_olympics %>% filter(season == 'Summer') %>% 
    summarize(year = year,
              running_percent_records   = cumsum(percent_records),
              running_percent_na_age    = cumsum(age_percent_na_season),
              running_percent_na_height = cumsum(height_percent_na_season),
              running_percent_na_weight = cumsum(weight_percent_na_season)
    ) %>% 
    mutate(season = 'Summer')

# Get running NA percentages for winter
running_na_percent_winter <- biometric_na_by_olympics %>% filter(season == 'Winter') %>% 
    summarize(year = year,
              running_percent_records   = cumsum(percent_records),
              running_percent_na_age    = cumsum(age_percent_na_season),
              running_percent_na_height = cumsum(height_percent_na_season),
              running_percent_na_weight = cumsum(weight_percent_na_season)
    ) %>% 
    mutate(season = 'Winter')

# Combine NA percentage tables to prepare for plotting with facet
running_na_percent <- rbind(running_na_percent_summer, running_na_percent_winter) %>% relocate(year, season)

# Prepare data for plotting by converting into long format
running_na_percent_long <- melt(running_na_percent[,c('year', 'season', 'running_percent_records', 'running_percent_na_age', 'running_percent_na_height', 'running_percent_na_weight')], id.vars = c(1,2))

# Plot running percent NAs by season
plot_running_na_percent <- running_na_percent_long %>% ggplot(aes(x=factor(year), y=value)) +
    geom_line(aes(x='1964'), size=3, color='firebrick4') +
    geom_bar(aes(fill=variable), stat='identity', color='black', position='dodge') +
    theme(axis.text.x = element_text(angle=45), plot.title = element_text(hjust=.5, size=16), strip.text = element_text(size=14), axis.title = element_text(size=14)) +                       
    facet_wrap(~season, ncol=1, scales='free') +
    labs(x='Year', y='Running Percentage of NAs', title='Running % of NA by Year and Season')


