# Project-4
By Lorenzo Davis, Chelsea Cullen, Robin Ryan, and Mark Olson <br/>
# Customer Segmentation for an Online Retailer 
# Using RFM (Recency, Frequency, Monetary) Analysis, K-Means Clustering, and Neural Networks


# Project Overview
 - **Objective:** To segment customers based on their purchase behavior to identify distinct customer groups for targeted marketing strategies. 
 - **Dataset:** UCI Online Retail Dataset https://archive.ics.uci.edu/dataset/352/online+retail
     - Key attributes include:
         - **InvoiceNo:** Unique invoice number.
         - **StockCode:** Product/item code.
         - **Description:** Description of the product.
         - **Quantity:** Number of units of the product purchased.
         - **InvoiceDate:** Date of the transaction.
         - **UnitPrice:** Price per unit of the product.
         - **CustomerID:** Unique identifier for each customer.
         - **Country:** Country where the customer resides.
 - **Skills Applied:**
     - **Data Preprocessing:** Cleaning and preparing the dataset for analysis (e.g., handling missing values, removing duplicates, and filtering out irrelevant transactions).
     - **RFM Analysis:** Using Recency, Frequency, and Monetary values to group customers based on purchasing behavior.
     - **K-Means Clustering:** Segmenting customers into distinct groups based on RFM metrics for targeted marketing.
     - **Neural Network for CLV Prediction:** Predicting future customer lifetime value (CLV) using a neural network model with behavioral features.
     - **Data Visualization and Evaluation:** Visualizing the customer segments and analyzing the performance of the models using various metrics.

#
# Part 1: Data Cleaning, Preprocessing, and Analysis

    Open the jupyter notebook Project4_CustomerSegmentation.ipynb and extract/save the file Online_Retail_Data.csv in the same folder.
    The raw dataset is a csv file that contains 8 columns and 541,909 rows of transactional data from Dec 2010-Dec 2011.

**Data Cleaning and Preprocessing Steps:**
 - Drop any rows without a Customer ID
 - Drop any rows where the Quantity or Unit Price is less than or equal to $0.00
 - Format the Invoice Date as DateTime
 - Format the Customer ID and Invoice No as string values
 - Create a new column 'Total Price' that is Quantity * Unit Price
 - Create a new column 'Recency' that calculates the total number of days between purchases for each InvoiceNo
 - Create a new column 'Frequency' that is based on the total number of unique invoice numbers per customer 
 - Create a new column 'Monetary' that is the sum of the Total Price for each customer

    After data cleaning and preprocessing the dataset will contain 8 columns and 392,692 rows of transactional data

## Data Analysis:
We previewed the raw cleaned data to get an overview of what type of customer behaviors are represented.  We specifically looked at Distribution of Time Since First Purchase, Total Number of TRansactions per Customer, and Unique Products Purchased by Customers.

We decided to use RFM (Recency, Frequency and Monetary) Analysis as it allowed us to use data based on existing customer behavior to predict how a new customer is likely to act in the future.  
The data was aggregated using the Customer ID to calculate the RFM values for each customer.

The data was then scaled to be used for further analysis in Parts 2 and 3

# Part 2: Create Retail Clusters using Machine Learning K-Means Model

K-Means was chosen to group the customers based on ease of implementation and effectiveness of segmentation tasks. 
We also tried to Birch Means method but it didn't give us anything different than K-Means Model so we chose to continue with the K-Means model.

We used the elbow method to determine the optimal number of clusters for our K-Means model.  We settled on 4 clusters after reviewing the elbow graph and the K-Means cluster graphs.  With 4 clusters we were better able to segment the outliers.

Once we identified the clusters we transformed the scaled data back to the original values to analyze the different segments and provide Cluster Profiles based on RFM average values.

## Cluster Interpretation and Profiles:

Cluster Profiles (based on average, original data values)
Cluster 0 (Recent Moderate Spenders): 
3,054 customers (70.4%)
~43 days since last purchase
3.68 purchases
~$1,353 total spend
Cluster 1 (Infrequent Low Spenders): 
1,067 customers (24.5%)
~247 days since last purchase
1.55 purchases
~$478 total spend
Cluster 2 (Loyal High Spenders): 
13 customers (0.29%) - these are most likely wholesale retailers
~6 days since last purchase
82.54 purchases
~$127,188 total spend
Cluster 3 (Frequent High Spenders): 
204 customers (4.7%)
~15 days since last purchase
22.33 purchases
~$12,690 total spend

Based on our segmentation analysis using the RFM data for each cluster and viewing the plotted data, we can advise the company on different ways the segments can be marketed to such as:
Marketing Campaigns:
Focus special promotions or loyalty programs on high-value clusters, like Cluster 2 (Loyal High Spenders), to retain and grow their loyalty.
Design reactivation campaigns for Cluster 1 (Infrequent Low Spenders) to bring them back with targeted offers.
Frequent buyers might appreciate early access to sales, while occasional buyers might be enticed with special promotions to increase their purchase frequency.
Resource Allocation: Allocate marketing budgets efficiently by identifying which customer segments to invest in. For example, investing more in Cluster 3 (Frequent High Spenders) may yield higher returns.
Key Point: RFM analysis provided the foundation for our customer segmentation, which allowed us to understand distinct customer behaviors and enable the business to create actionable strategies to engage each group more effectively.

At this point in our analysis we decided to circle back to the original dataset and further complete further investigation at an item detail level.  Using Spark sql the item data was queried to determine the top items purchsed in each cluster.  The items in each cluster were compared to see what overlapping items were purchased.  This provided us with further insight into how to move forward creating our predictions of future customer behavior.

# Part 3: Neural Network for Customer Lifetime Value (CLV) Prediction

What is CLV: CLV is the total revenue a business can expect from a single customer over their entire relationship. Understanding CLV is crucial for long-term business growth and customer relationship management.
Approach to Predicting CLV Using Neural Networks
Why Neural Network?
A neural network was chosen because:
Nonlinear Relationships: CLV is influenced by various factors (recency, frequency, monetary value, and other behavioral variables), and the neural network seemed to perform the best at modeling these nonlinear relationships.
Feature Engineering: Additional customer behavior features (e.g., average purchase value, time since first purchase, number of unique products) enhance the model's predictive power.
Key Point: This neural network is designed to predict the monetary value of future customer transactions using behavioral and transactional features.
Model Architecture:
Input: 7 features (Recency, Frequency, Monetary, Avg Purchase Value, Time Since First Purchase, Unique Products Purchased, Avg Quantity per Transaction).
Layers:
64 neurons in the first layer (ReLU activation).
32 neurons in the second layer (ReLU).
16 neurons in the third layer (ReLU).
8 neurons in the fourth layer (ReLU).
1 neuron in the output layer (Linear activation for regression).
Handling Outliers:
We Capped at the 80th Percentile
During the CLV analysis, we observed that the monetary values had significant outliers that distorted the model's predictions. To ensure the model performed optimally, we capped the Monetary variable at the 80th percentile to limit the influence of extreme values.
Effect: This helped the model predict more typical customer behavior without being skewed by abnormally high-value customers, resulting in a more accurate and generalized prediction model.

## Results of CLV Prediction
Model Evaluation Metrics:
R-Squared (R²): 0.88
Mean Absolute Percentage Error (MAPE): 28.17% 
Root Mean Squared Error (RMSE): $250.00 (the model's average prediction error).
Accuracy: 59.56% within ±20% tolerance
What the Results Mean:
R² = 0.88: The model is effective in explaining CLV, with only 12% of the variance left unexplained.
MAPE = 28.17%: The predictions are off by 28%.
Accuracy within ±20% tolerance = 59.56%: Over half the predictions are within 20% of the actual CLV value, showing the model can provide approximate predictions for many customers.
Reason for 20%Tolerance Buffer: To accommodate for predicting unpredictable consumer behavior (e.g. many influences that contribute to purchasing power), we set a 20% buffer to capture predictions that are close enough to the actual values to still be actionable.
Opportunities for Improvement
Model Refinement:
Additional Features: Including more customer behavioral data or external factors could further improve the model's accuracy.
Hyperparameter Tuning: Exploring different architectures, learning rates, and optimization 

## How CLV Prediction Can Be Used
Marketing & Promotions:
Identify High-Value Customers: Use CLV predictions to identify which customers are most likely to generate the most revenue, enabling focused marketing efforts and personalized promotions on high-value individuals.
Customer Retention Strategies:
Target At-Risk Customers: Identify customers who may have high potential value but are at risk of churn. Personalized offers or loyalty programs can help retain these customers.
Customer Segmentation:
Segment by CLV: Beyond RFM segmentation, customers can be segmented by predicted CLV to create even more focused marketing and sales strategies.
Key Point: CLV prediction enables proactive engagement with customers and the ability to make data-driven decisions for profitability.

## Limitations 
Outliers: We achieved 80% accuracy with the model, but this required excluding the top 20% of spenders, who are likely wholesalers. 
Additionally, it was challenging to clearly present each individual's spending habits. We had to work with a smaller dataset (Clusters) to ensure that the graphs remained clear and informative.
Unaccounted external factors: This does not account for outside factors and assumes the market will stay stable. (e.g., economic conditions, seasonal trends) which could influence spending patterns and impact the model's performance

## Conclusion

Our analysis provided our client with valuable insights into key marketing opportunities, optimal budget allocation, and predictive customer behavior. This enabled them to effectively target future customers and identify top-selling items. 
Additionally, our findings offered a comprehensive view of both short-term and long-term gains through customer lifetime value (CLV), supporting strategic decision-making for sustained business growth.
