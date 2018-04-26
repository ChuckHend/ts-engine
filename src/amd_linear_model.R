require(ggplot2)

setwd('/media/ahendel/Data/Dropbox/Projects')

d <- read.csv("amd_df.csv")

ggplot(d, aes(x=timeSeq, y=scale_close)) +
  geom_point() +
  geom_smooth()

mod <- lm(scale_close ~ timeSeq, data=d)

summary(mod)

?lm
