OlympicData <- read.csv(file = 'Desktop/Class/olympic-kaggle-dataset/recent_olympic_data.csv')
head(OlympicData)


#To select a sport and only have it in the data table

OlympicData <- subset(OlympicData, sport =="Athletics")

#Rerun the first line to gain full data table again



Year <- OlympicData$year
Sex <- OlympicData$sex
Height <- OlympicData$height
Weight <- OlympicData$weight
Age <- OlympicData$age
Event <- OlympicData$event

# count number of athletes, nations, & events, excluding the Art Competitions
counts <- OlympicData %>% group_by(year, season) %>%
  summarize(
    Athletes = length(unique(id)),
    Nations = length(unique(noc)),
    Events = length(unique(event))
  )


p1 <- ggplot(counts, aes(x=year, y=Athletes, group=season, color=season)) +
  geom_point(size=2) +
  geom_line() +
  scale_color_manual(values=c("darkorange","navyblue")) 
p2 <- ggplot(counts, aes(x=year, y=Nations, group=season, color=season)) +
  geom_point(size=2) +
  geom_line() +
  scale_color_manual(values=c("darkorange","navyblue"))
grid.arrange(p1, p2, ncol=1)

# Plot Total Number of Events per year
ggplot(counts, aes(x=year, y=Events, group=season, color=season)) +
  geom_point(size=2) +
  geom_line() +
  scale_color_manual(values=c("darkorange","navyblue")) + 
  labs(title = "Number of Olympics Events in Given Year") +
  theme(plot.title = element_text(hjust = 0.5))


# Recode year of Winter Games after 1992 to match the next Summer Games
# Thus, "Year" now applies to the Olympiad in which each Olympics occurred 
original <- c(1994,1998,2002,2006,2010,2014)
new <- c(1996,2000,2004,2008,2012,2016)
for (i in 1:length(original)) {
  OlympicData$year <- gsub(original[i], new[i], OlympicData$year)
}
OlympicData$year <- as.integer(OlympicData$year)

# Table counting number of athletes by Year and Sex
counts_sex <- OlympicData %>% group_by(year, sex) %>%
  summarize(Athletes = length(unique(id)))
counts_sex$year <- as.integer(counts_sex$year)

# Plot number of male/female athletes vs time
ggplot(counts_sex, aes(x=year, y=Athletes, group=sex, color=sex)) +
  geom_point(size=2) +
  geom_line()  +
  scale_color_manual(values=c("maroon","darkblue")) +
  labs(title = "Number of Olympians by Given Gender per Year") +
  theme(plot.title = element_text(hjust = 0.5))

#Change as.factor(Year) to as.factor(Event)
#to view by event
#Don't forget to change the axes!

p3 <- ggplot(OlympicData, aes(x = as.factor(Year), y = Height, fill = Sex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympic Year") + ylab("Height") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=12, angle=90))
p4 <- ggplot(OlympicData, aes(x = as.factor(Year), y = Weight, fill = Sex)) +
  geom_boxplot(alpha = 0.90) +
  xlab("Olympic Year") + ylab("Weight") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=12, angle=90))
p5 <- ggplot(OlympicData, aes(x = as.factor(Year), y = Age, fill = Sex)) +
  geom_boxplot(alpha = 0.90) +
  xlab("Olympic Year") + ylab("Age") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=12, angle=90))

grid.arrange(p3, p4, p5, ncol=1)