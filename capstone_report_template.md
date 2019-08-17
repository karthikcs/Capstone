# Machine Learning Engineer Nanodegree
## Predicting flight delays
Srividya  
August 17th, 2019

## I. Definition
_(approx. 1-2 pages)_

### Project Overview
The commutes through flights have helped us to overcome the barrier of geography.They have become very
common among the individuals, might be a trip or a business meeting. As the number of people who travel
have increased, the number of airlines and airports have also increased to make a passenger's life much
easier. The sizes of the airports differs from one to another based on the population of the passengers and
number of incoming and outgoing flights in the airports. There are different types of flights, in this problem we
mainly focus on domestic flights. There are nearly 44K aircrafts that lands daily on average in US and 87K
flights that crisscross US everyday on avg. From this we can infer that the Air traffic is huge and there will be
delays in takeoffs and landings. The delay in flight is very painful for passengers and airlines. This could lead
to very serious problem such as missing important business meetings or a sport competition etc which might
lead to huge losses. This mainly creates stressful situations among the passengers. There are cases where
passengers have missed connecting flights and have waited double the time than they expected.
Due to this problem it has become very important that passengers are well informed in advance so that they
can plan their trip accordingly. If we have data on flights arrival and departure timings of its everyday trip we
can capture the pattern of these delays and predict them accordingly based on the data(history) that we have.
This is the main role of our project.

### Problem Statement
Booking flights online have become very common through websites. There are tons of websites that do flight
bookings, they provide details such as arrival, departure time, fare etc. But what if they provide details on the
delays? This will help users or passengers to decide or plan on their trip upfront. The overhead of delays or
surprise delays will be reduced to some extent.
This prediction also helps airlines to avoid delays through planning and thus help both passengers to have a
good journey and airlines to become more efficient. We can predict these delays using regression models.

### Metrics

As our problem is based on prediction its a regression problem and large differences between actual and predicted are punished more in MSE than in MAE. Therefore we have taken MSE as our evaluation metric. As we have tried different models including benchmark model which is linear regression they all should have common evaluation metrics so that we can compare which is the better model. For this purpose we are using MSE as a common metric. MSE is given by:

![Drag Racing](https://raw.githubusercontent.com/srividya129/Capstone/master/MSE2.png)

MSE basically measures average squared error of our predictions. For each point, it calculates square difference between the predictions and the target and then average those values.

The higher this value, the worse the model is. It is never negative, since we’re squaring the individual prediction-wise errors before summing them, but would be zero for a perfect model .

## II. Analysis

### Data Exploration

Each entry of the input file `flights.csv` file corresponds to a flight and we see that more than 5'800'000 flights have been recorded in 2015. These flights are described according to 31 variables. Here are few variables that I explain:
Inputs from the file:
- **YEAR, MONTH, DAY, DAY_OF_WEEK**: dates of the flight <br/>
- **AIRLINE**: An identification number assigned by US DOT to identify a unique airline <br/>
- **ORIGIN_AIRPORT** and **DESTINATION_AIRPORT**: code attributed by IATA to identify the airports <br/>
- **SCHEDULED_DEPARTURE** and **SCHEDULED_ARRIVAL** : scheduled times of take-off and landing <br/> 
- **DEPARTURE_TIME** and **ARRIVAL_TIME**: real times at which take-off and landing took place <br/> 
- **DEPARTURE_DELAY** and **ARRIVAL_DELAY**: difference (in minutes) between planned and real times <br/> 
- **DISTANCE**: distance (in miles)  <br/>
Output:
Y-Predicted delay values
An additional file of this dataset, the `airports.csv` file, gives a more exhaustive description of the airports

For easy computations I will be taking subset of this data which is of one month January 2015.

Next I will clean the data by changing the date time format of time variables for convenience. Also as there are few missing values but very less. I remove them and remove all NANs. 
The statistics of the the delay grouped by each airline is as follows: 

<<<<IMG GET stats1.png>>>

### Exploratory Visualization

<<IMG pie chart>>>>
<<IMG delay dispersion>>

Considering the first pie chart that gives the percentage of flights per airline, we see that there is some disparity between the carriers. For exemple, *Southwest Airlines* accounts for ~20% of the flights which is similar to the number of flights chartered by the 7 tiniest airlines. However, if we have a look at the second pie chart, we see that here, on the contrary, the differences among airlines are less pronounced. Excluding *Hawaiian Airlines* and *Alaska Airlines* that report extremely low mean delays, we obtain that a value of 11+or-7 minutes would correctly represent all mean delays. Note that this value is quite low which mean that the standard for every airline is to respect the schedule !

Finally, the figure at the bottom makes a census of all the delays that were measured in January 2015. This representation gives a feeling on the dispersion of data and put in perspective the relative homogeneity that appeared in the second pie chart. Indeed, we see that while all mean delays are around 10 minutes, this low value is a consequence of the fact that a majority of flights take off on time. However, we see that occasionally, we can face really large delays that can reach a few tens of hours !

The large majority of short delays is visible in the next figure:


<<Img short and large delays>>


This figure gives a count of the delays of less than 5 minutes, those in the range 5 < t < 45 min and finally, the delays greater than 45 minutes. Hence, we wee that independently of the airline, delays greater than 45 minutes only account for a few percents. However, the proportion of delays in these three groups depends on the airline: as an exemple, in the case of 
*SkyWest Airlines*, the delays greater than 45 minutes are only lower by ~30% with respect to delays in the range 5 < t < 45 min. Things are better for *SoutWest Airlines*  since delays greater than 45 minutes are 4 times less frequent than delays in the range 5 < t < 45 min.


Number of airports visited by each airline:
<<IMG number of airports visited by each airline>>

### Algorithms and Techniques
Here we are using the following models:
-Linear regression
-Polynomial regression
-Light GBM 

Here I will use Linear regression as benchmark model and later I move to polynomial regression as it provides of the relationship between the dependent and independent variable.Polynomial basically fits a wide range of curvature. Later we will use Light GBM as I wanted to try out a tree based model.
We split our data to train test and apply the above algorithms and compare the evaluation metrics.
The regularization technique ridge is also implemented for better outcome. We have also used Grid search technique to find best plynomial order n and alpha-coefficient of ridge regression.

### Benchmark 

As any regression problem I have considered linear regression as benchmark model and have tried to beat its MSE value
The below is the value of the MSE for our benchmark model that was used:
Model	MSE
Linear Regression	76.95 (Benchmark)

Liner regression can be mathematically expressed as:
Y = mX+b where:
Y = Target variable or dependent variable, in our case it is delay
X=x1, x2 .. xn = Independent variables (features) which will impact the Target variable. In our case, it can be
source airport, airline, time of departure etc m = set of coefficients of different features b = Bias, it is also Y
interecept

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
We have few data processing steps such as changing the format of Scheduled departure(Date_time), Scheduled arrival(time) and Arrival time(time) and departure time(time) to the formats specified in brackets. Scheduled departure was in float type.
Next we treat missing values by removing them as ~97% of our data is complete.
While doing analysis we do it for the month of january. 
In modelling we split the data into train and test. Also we have considered cross validation and regularization.

### Implementation

Here are the following steps that were followed while solving this problem:

<<IMG Implementation>>

### Refinement

We will preprocess the data by changing the date time formats as required and treat missing values. 
We split the data into train (70%) and test(20%). We include regularization technique and use grid search to find best polynomial order and alpha coefficient of ridge regression.
The grid search result were as below:

<<IMG grid search>>>


## IV. Results


### Model Evaluation and Validation

The below table gives the MSE value obtained for used models
<<MSE Values>>
The Polynomial regression model is better than Linear regression  with MSE ~74. Later we tried it for Light GBM which gave us an MSE value of ~87

### Justification

As the above figure gives the MSE value for each model, we can see that Polynomial regression is performing better than both the benchmark model and LightGBM model. This is due to the use of regulrization with parameter tuning. 
There could be improvements made on the LightGBM as well by tuning their parameters but as we can see the params of LightGBM are very large and would take time for its tuning. This becomes a new project in itself. 


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?