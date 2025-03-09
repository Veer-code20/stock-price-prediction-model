# Load necessary libraries
library(ggplot2)
library(readr)

# Load the predictions from the CSV file
results <- read_csv("stock_predictions.csv")

# Plot Actual vs Predicted stock prices
ggplot(results, aes(x = 1:nrow(results))) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "Actual vs Predicted Stock Prices", x = "Time", y = "Stock Price") +
  scale_color_manual(name = "Legend", values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()
# Save the plot as a PNG file
ggsave("stock_plot.png", width = 8, height = 6)

