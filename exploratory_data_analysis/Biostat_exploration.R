# Load packages
library("plotly")
library("tidyverse")
library("gridExtra")
library("knitr")


OlympicData <- read.csv(file = 'Desktop/Class/olympic-kaggle-dataset/recent_olympic_data.csv')
head(OlympicData)


Year <- OlympicData$year
Sex <- OlympicData$sex
Height <- OlympicData$height
Weight <- OlympicData$weight
Age <- OlympicData$age

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

p3 <- ggplot(OlympicData, aes(x = as.factor(Year), y = Height, fill = Sex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympics Year") + ylab("Height") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))
p4 <- ggplot(OlympicData, aes(x = as.factor(Year), y = Weight, fill = Sex)) +
  geom_boxplot(alpha = 0.90) +
  xlab("Olympics Year") + ylab("Weight") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))
p5 <- ggplot(OlympicData, aes(x = as.factor(Year), y = Age, fill = Sex)) +
  geom_boxplot(alpha = 0.90) +
  xlab("Olympics Year") + ylab("Age") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))

grid.arrange(p3, p4, p5, ncol=1)

Wrestling <- subset(OlympicData, sport =="Wrestling")

WYear <- Wrestling$year
WSex <- Wrestling$sex
WHeight <- Wrestling$height
WWeight <- Wrestling$weight
WAge <- Wrestling$age


p6 <- ggplot(Wrestling, aes(x = as.factor(WYear), y = WHeight, fill = WSex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympics Year") + ylab("Height") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))

p7 <- ggplot(Wrestling, aes(x = as.factor(WYear), y = WWeight, fill = WSex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympics Year") + ylab("Weight") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))
p8 <- ggplot(Wrestling, aes(x = as.factor(WYear), y = WAge, fill = WSex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympics Year") + ylab("Age") +
  scale_fill_manual(values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))

grid.arrange(p6, p7, p8, ncol=1)


Athletics <- subset(OlympicData, sport =="Athletics")

AYear <- Athletics$year
ASex <- Athletics$sex
AHeight <- Athletics$height
AWeight <- Athletics$weight
AAge <- Athletics$age
AEvent <- Athletics$event


legend_title = "Sex"
p9 <- ggplot(Athletics, aes(x = as.factor(AEvent), y = AHeight, fill = ASex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympics Year") + ylab("Height") +
  scale_fill_manual(legend_title, values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))

p10 <- ggplot(Athletics, aes(x = as.factor(AEvent), y = AWeight, fill = ASex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympics Year") + ylab("Weight") +
  scale_fill_manual(legend_title, values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))
p11 <- ggplot(Athletics, aes(x = as.factor(AEvent), y = AAge, fill = ASex)) +
  geom_boxplot(alpha = 0.9) +
  xlab("Olympics Year") + ylab("Age") +
  scale_fill_manual(legend_title, values = c("maroon", "blue")) + theme(axis.text.x = element_text(size=8, angle=90))

grid.arrange(p9, p10, p11, ncol=1)


