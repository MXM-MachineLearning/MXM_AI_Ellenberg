library(tidyverse)

df = read_csv("viz_a_b_sqrt_c.csv")

ggplot(df, aes(x=min_a_b_sqrt_c, y=num_moves_Q_learning_needs)) + 
  geom_point(color="blue", alpha=0.05) +
  geom_smooth(method = "lm", color="black", se = FALSE) +
  geom_smooth(color="red", se = FALSE) +
  xlab("Min(a, b, sqrt(c))") + 
  ylab("Number of moves Q learning needs") + 
  theme(
    panel.grid.major.x = element_blank(),  
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),  
    panel.grid.minor.y = element_blank()
  ) + 
  ggtitle("Vizualizing number of moves by the matrix values")
  
  
       