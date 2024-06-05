# nyc_taxi
# TaxiNavigator: A Guide for Maximizing Earnings

## Problem Statement
The investigation aimes to conduct regression task on the New York City Yellow Taxi Dataset(January, 2024). The overall goal of this project is to develop a model that predict, where and when the taxi driver can make most money. This involves building different model and finding the most predictive features.

When dealing with the New York City Taxi Dataset, one must first define the specific problem to be solved. In the context provided, the problem is twofold: predicting oral temperature from infrared thermographic images and determining whether the temperature indicates a fever. The dataset, which could be sourced from NYC Taxi and Limousine Commission website, contains data from all the yellow taxi trip of January 2024.

![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/6fdc87d2-26d4-43a7-9705-8a172f84b59d)

This is the detailed explaination of each feature:
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/0776b1a4-cb9c-44ac-976c-5b52ed09bc46)



The output variable is total taxi fare
**Total taxi fare**: The total amount of the tax fare that one trip.
My project aims to forecast total taxi fare, which is the total amount a passenger is charged for a taxi trip. By analyzing this output variable, I can create a model that predicts the fare based on several features, such as trip distance, travel time, and travel date. This framework helps address regression problems where the goal is to predict a continuous value like total fare.

To implement this solution, I will develop a robust machine learning architecture capable of accurately predicting the total taxi fare for a given trip. To ensure the model's effectiveness, I select appropriate evaluation metrics such as **mean absolute error, mean squared error, and R-squared** to gauge its accuracy and reliability. I will assess the model's performance using **cross-validation**, a robust statistical method that maximizes the use of available data by iteratively splitting the dataset into training and validation subsets. This practice helps in estimating the performance of the model on unseen data, thereby ensuring that the model generalizes well and does not overfit to particular quirks of the training data.

Given this setup, the overarching objective is to provide actionable insights for taxi drivers and passengers to understand the factors contributing to fare fluctuations and identify ways to optimize travel costs. The model's outcomes can also be used to inform dynamic pricing strategies, optimize taxi routes, and enhance customer satisfaction by providing more predictable fare estimates.

By the end of this investigation, I aim to have a optimal model tuned to offer the highest predictive accuracy on the Dataset. This report will detail the steps taken in the investigation, the implementation of training, evaluation, the results obtained, and discussion.

## Methodology
### Step 1: Assemble the Dataset
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/3970553d-3e31-4bc4-a76b-f259884d82c4)

the raw dataset is 2964624 entries, it would consume intensive computing resources for training.  I use the systematically sample it by 100 to make it 29646. 

### Step 2: Choose a Evaluation Metric
Choosing the right evaluation metrics is essential to assess the effectiveness of our model in predicting taxi fare and suggesting optimal pickup zones. For the regression aspect, metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Squared Error (MSE) are suitable. These metrics help understand how close the predicted fare is to the actual fare, providing an insight into the model's accuracy.

### Step 3: Decide on a Evaluation Protocol
The evaluation protocol defines how the model will be assessed. This often includes splitting the dataset into training, validation, and test sets to ensure that the model can generalize well to new, unseen data. Cross-validation techniques such as k-fold cross-validation may also be employed to make the most efficient use of the data for training and evaluation.

### Step 4: Prepare the Data
**4.1 Cleaning the data**
First, remove the null value. Drop the columns that is not related to the question and use the mean value to fill the feature that might need.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/c1f13d0e-3818-448a-9668-f3a8dce53f4c)


Then, remove the outliers.
Visualise the data distribution. And saw there might be some extreme values on the upper end.
Calculating the values at each percentile from 1 - 100. And discovered there is incontinuty between 90 and 100.
Calculating the values at each percentile from 90 - 100 to see the trends more closely. And discovered there is incontinuty between 90 and 100.
Take a closer look into the data by calculating the values at each percentile from 99 - 100. Found the gap between 99.9 - 100. Remove the outliers from 99.9.
Repeat the process for the other needed features.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/6c7b14a2-7b29-4f0f-99cb-ad2f5ee15ea8)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/91d7f897-122c-4455-8b1c-052098c39498)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/e4795061-0072-4a1c-af56-25da844d0d3d)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/8b2e057d-e992-45ee-a0c9-518d33a94ef2)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/d3a13d89-8e1b-4916-b03d-6d1d6785eee0)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/605f3620-275a-4b07-80e0-85782e33ed10)

**4.2 Exploratory Data Analysis**
At this stage, we try to visualise the data to see spot trends in the data. So to choose the features we need for later training.
First we create two new columns, time and weekday, out of the date data that already exists.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/4fed1361-c480-4ff8-b52f-fcbdce82f15e)

Map the data to the New York City map. Since the original pickup location and drop off location is in ID form. We need a look up csv to find out what those number presents. I have done this by merge the dataframe.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/140eea7f-3df2-4955-a892-61760065c52b)


and I double check if the merge has been done properly
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/2acda5b9-6a23-4429-9f3d-43fc3ef69012)

Then I try to plot some data to better spot the relation between different features.
trip distance and time. In order to calculate the total fare taxi fare of a given time, trip distance is an importance factor. This graph shows that the trip distance demand would varied with a day. That's to say, at certain time of the day, the total fare would be higher.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/0444da6e-3171-4c5a-b783-79aa7874077b)

Trip distance and weekday. By plotting the trends, I discovered on certain days of the week, like Wednesday or Friday, people tend to travel further. Therefore, trip distance is a feature that can't be neglect.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/2e805c89-8ca2-445f-af4c-ccd7d6633e7d)

Trip distance and zone. There is a significant relation of the trip distance and where people live.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/596aa9c9-2716-4458-91e1-fcefafdc8a9e)

**Visualise the business**
Calculate the business of each zone in different weekday and specific hour
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/9983e8f2-f756-4855-b7a0-3dbd24492197)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/1f3407e8-9fd3-433b-a6da-b2e4311e7a49)

Plot the data onto a map. This requires a shape file, where the geometry data is available so you can use the name of the zone to map the data onto the image.
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/90412d47-cc46-4b1d-bf07-6d0a7c2d2062)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/c332f424-2573-4a46-8da3-e869fcbd2561)

The business of the whole week in each zone
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/4893ef96-e6cc-408b-a202-fbf5e5f97754)

The business of Monday in each zone
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/f5f40a69-fdea-43f4-a3f2-3b8f8f61ac2c)

**4.3 Train test data split**
After the data exploratory process, we have a better understanding of our data. It is time to do the feature engineering and choose the most related feature to build our model. In this process, we save more time on the training process.

![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/6534d001-6565-47c2-8df7-5064d6b1cf8f)

In this case, I decided pickup hour, pickup weekday, trip distance, tip amount, extra might be the factors that affect the total amount that a taxi driver can earn. 
I use a 70%/30% split because it is a relatively bigger dataset.

### Step 5: Train and Evaluate the Model
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/889f0a41-85f9-40db-8170-f67946ae6415)

I have tried 6 machine learning model and 2 neural network.

**Why I choose the four algorithms?**

**Linear Regression:** Linear Regression is a simple, interpretable algorithm that assumes a linear relationship between the features and the target variable. It's often a good starting point for regression tasks. When have continuous variables and a linear relationship might exist, Linear Regression can quickly give us an understanding of how changes in the input features affect the output. It can be easily visualized and explained, making it ideal for basic models or as a benchmark for comparison with more complex algorithms.

**K-Nearest Neighbour (KNN):** KNN is a non-parametric, instance-based learning algorithm. It doesn't make any assumptions about the underlying data distribution. Instead, it relies on the principle that similar instances should have similar outcomes. KNN is useful for capturing non-linear relationships in the data and can be adjusted by changing the number of neighbours. It is a versatile method that can help when the data has clusters or complex patterns.

**Random Forest:** Random Forest is an ensemble learning technique that combines multiple decision trees to improve performance and reduce overfitting. By randomly selecting subsets of features and instances to build each tree, Random Forest enhances generalization and robustness. This algorithm is suitable for complex datasets with many features and interactions. It can handle non-linear relationships and provide feature importance, helping to identify key contributors to the target variable.

**XGBoost:** XGBoost (Extreme Gradient Boosting) is an advanced gradient boosting algorithm designed for high performance and scalability. It uses a series of trees to make predictions, where each tree aims to correct the errors of the previous ones. XGBoost offers extensive parameter tuning options, making it highly adaptable to different datasets. It is known for its accuracy and speed, especially in competitions and real-world scenarios where the data is complex and large.

Moreover, I conducted experiments on neural network. Neural networks are used in this project because they excel at modeling complex, non-linear relationships and can handle large datasets. 

Keras Tuner is chosen for hyperparameter optimization due to its ability to automate the tuning process, reducing manual effort. It offers different search strategies to find the best hyperparameters forthe neural network. This automation helps streamline the workflow. However, the performance of the finetuned model is not significantly improved when compared to the regular model.

As both random forest and XGBoost model performs well to the data, I chose to finetune them using grid search and random search. The finetuned Random Forest model outbeat the XGBoost model, I chose it as the final model.


### Step 6: Finalise the Model
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/fe73d6d9-d20a-4c28-8c74-385ace8175be)

The final model achieved a Mean Absolute Error (MAE) of 2.1 and an R-squared value of 0.93, indicating a strong ability to predict and explain the variation in the data. These results suggest that the model is generally effective at capturing the underlying patterns in the dataset.

However, the slightly higher performance on the training set compared to the test set indicates potential overfitting. This occurs when the model learns the training data too well, including its noise, and struggles to generalize to new, unseen data. To mitigate this, you might consider techniques like regularization, dropout, or reducing the model's complexity to improve generalization.


![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/ed4257cb-8c2e-403b-993e-53dd387b3adb)
![image](https://github.com/weather-sketch/nyc_taxi/assets/138662766/204eee6d-413b-4dea-abdb-5b512fb00105)

The final model delivers on the proposed functionality by allowing users to enter a specific time and weekday to obtain insights into expected fare amounts and recommendations on the optimal locations to maximize earnings. It provides taxi drivers with data-driven guidance on where and when to go to maximize their revenue during their shifts. This functionality helps drivers make informed decisions and strategically plan their routes to optimize income.

# Limitation
## Inexperience in Large Dataset
Despite neural networks' reputation for excellent performance on large datasets, my experience has primarily focused on smaller datasets where overfitting is a common issue. Transitioning to larger datasets, I've encountered underfitting problems, which can be challenging to resolve. To address this, I used Keras Tuner to automatically fine-tune the hyperparameters. However, the performance did not improve as much as expected, suggesting that I have not yet harnessed the full potential of neural networks.

## Fail to use Google Map API
Additionally, I had difficulty using the Google Maps API to convert pickup and drop-off locations into longitude and latitude coordinates. This limited my ability to visualize popular routes at different times of the week effectively. If I could obtain accurate geographic data, I could create more insightful visualizations to understand taxi travel patterns in the dataset better.

# Discussion
In this study, we conducted a thorough investigation into the New York City Taxi dataset, focusing on building a predictive model that not only estimates fare amounts but also suggests optimal zones for pickups based on time and day of the week. The models employed, including linear regression, K-nearest neighbors, random forest, and XGBoost, were selected based on their ability to handle the complexities and non-linear relationships inherent in urban mobility data. **The final model, using a refined random forest approach, demonstrated significant predictive power, achieving an R-squared of 0.93 and a Mean Absolute Error of 2.1, which validates its efficacy in real-world scenarios.**

Looking forward, there is a strong intention to expand this analysis by incorporating more dynamic visualizations of taxi movement and fare trends across New York City. Utilizing geospatial data visualization tools will allow us to more vividly represent the flow and profitability of taxi operations across different times and zones. This will not only enhance the interpretability of our findings for stakeholders but also provide more granular insights that could guide strategic decision-making for taxi operators and city planners. The ultimate goal is to blend advanced analytics with intuitive visualizations to forge a deeper understanding of urban transportation dynamics.

# Reference
[1] New York City Taxi Limousine Commission(2024). TLC Trip Record Data. https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page Accessed on April 1st, 2024
