
# Airline Flight Search Ranker with an Unsupervised Learning Model
## CSCA-5632 Introduction to Machine Learning - Unsupervised Learning

The focus of this project is to build machine learning models that can rank and predict which flight options a customer will choose when presented with the flight search results. This is a group-wise ranking problem in that the target of the model is to rank the search results within each unique search session for each customer.

The dataset is provided through a Kaggle competition, hosted by the Aeroclub IT ([FlightRank 2025: Aeroclub RecSysCup](#acknowledgements)). The data is provided as a parquet file containing approximately 18 million rows of flight search results for ~33 thousand customers over a seven month period. Given the size of the dataset, the project will be broken into multiple steps, each with a specific focus.

The goal is to build a machine learning model, based on the historical data, that can rank the results of a flight search and predict based on the ranking of the search results. The model will be evaluated on a holdout set of data, and the final model will be used to rank the flight search results for the test set.

The approach for this project is to perform the final predictions by breaking the problem into two parts:

1. **Customer Segmentation**: For the first part, we will use unsupervised learning to cluster the customers into segments based on profile, search, and behavioral features.
2. **Flight Ranker**: For the second part, we will use a ranking model to predict the rank of the flight search results for each customer search session, with a separate model trained for each customer segment.

The results will be evaluated using the HitRate@3 metric to compare the selected flight to the top 3 ranking search results for each search session.

$HitRate@3 = \frac{1}{|Q|} \sum_{i=1}^{|Q|} 1(rank_i \leq 3)$

Where:
* $|Q|$ is the number of searches within a search session
* $rank_i$ is the rank of the $i^{th}$ search
* $1(rank_i \leq 3)$ is 1 if the correct flight is in the top-3, 0 otherwise


## Table of Contents

- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Data Source](#data-source)
- [Data Description](#data-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Split Data Set for Training and Test](#split-dataset-for-training-and-test)
- [Customer Segmentation - Unsupervised Learning](#customer-segmentation---unsupervised-learning)
- [Flight Ranking - Supervised Learning](#flight-ranking---supervised-learning)
- [Running the Process on Test dataj](#running-the-process-on-test-data)
- [Final Results](#final-results)
- [Next Steps](#next-steps)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [Video Overview](#video-overview)

## Installation

Follow these steps to setup the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/jfroggatt/aeroclub_recsys_2025
cd aeroclub_recsys_2025
```
### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
### 4. Launch Jupyter Notebook
```bash
Jupyter Notebook
```
Then open the `aeroclub_recsys.ipynb` file from the Jupyter interface.

## Requirements
-  Python 3.12
-  Jupyter Notebook
-  See `requirements.txt` for the full list

## Usage
-  Run each cell in sequence
-  Modify parameters or data to explore custom scenarios
-  Refer to comments for explanation of each step
## Project Notebook

The project code and process is maintained and documented in the Jupyter Notebook file `aeroclub_recsys.ipynb`. Given that the data set contain ~18 million rows, the notebook is broken into multiple sections, each with a specific focus, to help manage the memory requirements for running each step.
## Data Source

The dataset used for this project was provided by Aeroclub IT ([FlightRank 2025: Aeroclub RecSys Cup](https://www.kaggle.com/competitions/aeroclub-recsys-2025)) competition on [Kaggle](https://www.kaggle.com). The dataset files are too large to maintain within the GitHub project respository, so the competition project archive is downloaded to a local data directory, and the `train.parquet` file is extracted. No other files are needed from the competition project archive. 
## Data Description

A detailed description of the dataset is provided by the competition host, [Aeroclub IT](https://www.kaggle.com/aeroclubit), under the [data](https://www.kaggle.com/competitions/aeroclub-recsys-2025/data) section of the competition pages.

### Data Details

The `train.parquet` dataset consists of 18,145,372 flight search results, for 105,539 search sessions, for 32,922 customers.

## Exploratory Data Analysis

The following steps were taken in the Exploratory Data Analysis:

1. Import .parquet file, describe the dataset
2. Evaluate the distribution of flights to searches to customers
3. Evaluate missing data for the features
4. Evaluate flight carrier and frequent flyer IATA codes and matches
5. Evaluate airport IATA codes
6. Evaluate flight baggage allowances
7. EDA Conclusion

The output of steps 1 through 6 can be reviewed in the `cc_fraud_detect.ipynb` notebook, under the *Exploratory Data Analysis* section.

### EDA Conclusion

To recap what was identified during the EDA process:

**Data Set Size**: a primary factor for this project is managing a data set that consists of ~18 million rows. To accommodate this, the process is broken into steps where results are saved to .parquet files to support managing the memory utilization footprint for compute intensive steps and running multiple iterations. For the ranking process, the model selected supports running the data set in groups.

**Outliers**: we have some significant outliers related to the number of flights within a search session or the number of search sessions for a customer. The most significant of these outliers are removed from the data set for the training process.

**Missing Values**: There are a large number of missing values for some attributes, specifically for flight segments and legs features when there are limited or no connection or one-way flights. 'Segment3' is 100% missing for all flight legs, as none contain more than two layovers. As such, these features are removed from the data set. The remaining missing values are imputed, as they will still provide search or behavioral related insights and matching.

**Imputed Values**:
* legs[0,1]_segments[1-2]_* (flight segments): NULL values are imputed to 0 or "", depending on the data type, as these will still likely provide behavioral related matching.
* frequent_flyer: the frequent flyer code 'ЮТэйр ЗАО' is imputed to 'UT' (UTair JSC). The remainder, including empty strings are kept as is.
* corporateTariffCode (Corporate tariff code for businesss travel policies): impute to 0 (zero)
* miniRules0_percentage (Percentage penalty for cancellation): impute to 0.0 (zero)
* miniRules1_percentage (Percentage penalty for exchange): impute to 0.0 (zero)

## Feature Engineering

### Customer Features

For customer features, the training data set is aggregated by customer to generate customer profile and flight search and selection behavioral features with a goal of generating well defined customer segments for use in the flight ranking process. After evaluating the customer and flight related data, as well as some industry research, the following features are used or generated:

* Customer profile:
    * company_id, sex, nationality, frequent_flyer, is_vip, by_self (all existing attributes)
    * has_corp_codes, total_flights_searched, total_search_sessions, avg_searches_per_session, unique_routes_searched
* Booking Lead Time:
    * min_booking_lead_days, max_booking_lead_days, avg_booking_lead_days, median_booking_lead_days
* Travel Preferences:
    * most_common_departure_airport, unique_departure_airports, most_common_carrier, unique_carriers_used, round_trip_preference
* Cabin Class Preferences:
    * min_cabin_class, max_cabin_class, avg_cabin_class
* Temporal Preferences:
    * weekday_preference, return_weekday_preference, weekend_travel_rate, return_weekend_travel_rate, time_of_day_variance, night_flight_preference, redeye_flight_preference
* Route Specific Preferences:
    * route_loyalty, hub_preference, short_haul_preference, connection_tolerance, preferred_duration_quartile
* Price Sensitivity:
    * price_to_duration_sensitivity, avg_price_per_minute, price_per_minute_variance, price_position_preference, premium_economy_preference, consistent_price_tier (quartile), preferred_price_tier (quartile)
* Service Preference:
    * baggage_qty_preference, baggage_weight_preference,
    * loyalty_program_utilization
* Derived Metrics:
    * convenience_priority_score, loyalty_vs_price_index, planning_consistency_score, luxury_index, search_intensity_per_route, lead_time_variance, lead_time_skew, carrier_diversity, airport_diversity, cabin_class_range, customer_tier
* Interaction Features (to distiguish vip and corporate customers):
    * vip_search_intensity, vip_carrier_diversity, vip_cabin_preference
    * corp_search_volume, corp_roundtrip_preference, corp_planning_variance

### Flight Features

For flight features, the project generated additional features to generalize some features of the search session, such as is_roundtrip, as well as features that can assist in determining flight selection behaviors, such as is_weekend and carrier/frequent_flyer match. After evaluating the customer and flight related data, as well as some industry research, the following features are used or generated:

* Basic flight features
    * is_access3D, is_roundtrip, route_origin, origin_is_major_hub, destination_is_major_hub
* Flight segment features
    * leg0_num_segments, leg1_num_segments, total_segments, leg0_flight_time_min, leg1_flight_time_min
* Time related features
    * booking_lead_days, leg0_duration_minutes (including layovers), leg1_duration_minutes (including layovers),
    * trip_duration_minutes (including layovers),
    * leg0_departure_hour, leg0_departure_weekday, leg0_arrival_hour, leg0_arrival_weekday,
    * leg1_departure_hour, leg1_departure_weekday, leg1_arrival_hour, leg1_arrival_weekday,
* Flight service features
    * has_cancellation_fee, has_exchange_fee, max_cabin_class, baggage_allowance_qty, baggage_allowance_weight,
    * avg_cabin_class, min_seats_available, total_seats_available
* Carrier features
    * unique_carriers, primary_carrier, marketing_carrier, aircraft_diversity, primary_aircraft
* Derived flight features (based on other engineered features)
    * is_daytime, is_weekend, is_redeye, has_connections
* Windows based features (based on search session)
    * price_percentile, price_rank_pct, price_ratio_to_min, route_popularity, route_popularity_log

### Interactive Features

Interactive customer/flight features are generated to identify how flight characteristics match customer profile or behavioral preferences. The additional interactive Customer/Flight features generated are:
* daytime_alignment, weekend_alignment, departure_weekday_match, return_departure_weekday_match, redeye_alignment,
* price_preference_match, carrier_loyalty_match, carrier_ff_match

## Split dataset for Training and Test
The dataset is split to provide 80% for training and 20% for testing. The split is stratified by groupings for the number of flights per search session. This will help address the high and low extremes that are not removed as outliers, given the distributions identified during the EDA. 

## Customer Segmentation - Unsupervised Learning

This section performs the customer segmentation. The goal is to generate the groups of similar customers based on their profile attributes and search history. We will clean the data and engineer additional features to provide insights around flight search and selection behavior.

The process for peforming the customer segmentation is as follows:
1. Feature Engineering
2. Remove outliers
3. Feature encoding
4. Feature scaling
5. Dimensionality reduction
6. Evaluate alternative clustering methods (Gaussian Mixture Model, DBScan, Agglomerative Clustering)
7. Perform hyperparameter optimization
8. Analyze the cluster results
9. Add cluster labels to the customer data set
10. Save customer features

### Models

For the unsupervised modeling, we evaluated several models to determine which provides the best segmentation. We will evaluate the segmentation performed by each model using the Silhoutte Score metric to determine the best model to use. Once we have selected a model, we will then perform hyperparameter optimization to tune the model parameters. The optimal parameter configuration will again be evaluated using the Silhoutte Score metric. 

The models evaluated for this dataset:
1. Gaussian Mixture Model, 
2. DBScan, 
3. Agglomerative Clustering

### Segment Labels

Following the hyperparameter optimization process, the best model is used to generate the cluster labels for the customer data set. The resulting labels are added as a new 'customer_segment' feature that is used for the flight ranking process.

## Flight Ranking - Supervised Learning

This section performs the flight search results ranking. The goal is to predict the top 3 flights that meet the customer's selection behavior, based on the customer segmentation, flight features, and interactive features between the customer profile and flight data.

The process begins by cleaning the data and engineering additional features to provide insights around flight attributes. Customer features are added to the updated flight feature set and interactive features are generated to provide additional selection insight. Once all feature engineering is completed, the LightGBM model is trained for ranking all flights within their respective search session. A separate model is trained for each customer segment, as well as a model trained on the entire data set. From this, we are looking to determine if generating customer segments around profile and behavioral attributes enhances the predictivec capabilities of the model, or if the combined data set with all customer features is better at identifying the differences.

The process for peforming the ranking is as follows:
1. Feature Engineering of flight features
2. Prepare Training Data
    * combine Customer features with Flight features
    * generate interactive Customer/Flight features
    * generate groups to support efficient training of LightGBM model for such a large data set
        * grouping ensures that search sessions are not split across different groups for ranking.
3. Train LightGBM model for ranking
    * train a model for each customer segment
    * train a model for the entire data set

## Running the Process on Test data

Once the models have been trained, the process for running the models on the test data follows a similar approach to the training process:

1. Customer segmentation
   * Load the trained data encoders, scaler, and reducer for the customer data
   * Cleanse/impute, generate customer features, encode, scale, and reduce dimensions
   * Load the Agglomerative Clustering model centroid result
   * Generate cluster labels for the test data based on the model centroids
     * we use the model centroids, as Agglomerative Clustering does not provide a predict function 
   * Add cluster labels to the test data
2. Flight Ranking
   * Cleanse/impute, generate flight features
   * Combine customer features, flight features, and interactive features
   * Load the LightGBM models for each customer segment and the entire data set
   * Run the models for each customer segment
   * Run the model for the entire data set
3. Evaluate the models on the Test Data
4. Analyze the results

### Evaluation

The flight selection "prediction" is evaluated using the HitRate@3 metric. For this metric, we are evaluating if the flight actually selected by the customer is in the top 3 of the resulting flight ranking predictions.

$HitRate@3 = \frac{1}{|Q|} \sum_{i=1}^{|Q|} 1(rank_i \leq 3)$

Where:
* $|Q|$ is the number of searches within a search session
* $rank_i$ is the rank of the $i^{th}$ search
* $1(rank_i \leq 3)$ is 1 if the correct flight is in the top-3, 0 otherwise

For the final project evaluation, we restrict the HitRate@3 metric to only include search sessions with 10 or more flights. We do this because at 10 flights we have a 30% probability of randomly selecting the correct flight within the top 3 rankings. As the number of flights in the search gets smaller, the probability increases. There are also a number of search sessions in the data set that have only a single flight response, which we would get right 100% of the time regardless.

Note, for 10 flights, there are 120 combinations of picking 3 numbers:

$\binom{10}{3} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = 120$

If the selected flight is in the random sample, then we can determine favorable selections that will include picking any two other numbers from the remaining 9:

$\binom{9}{2} = \frac{9 \times 8}{2} = 36$

Then the probability the selected flight is included:

$P(\text{selected flight}) = \frac{\text{favorable}}{\text{total}} = \frac{36}{120} = 0.3$

## Final Results

Analysis of the final results is broken into two parts. The first part is the overall results around predicting the flight selection based on the HitRate@3 metric. The second part is the analysis on the impact of the customer segmentation on the final results.

### Overall Results

Overall, the final results for the Test data achieved a HitRate@3 of 0.7744. This is a better than expected result, given the process iterations performed and incremental results experienced throughout the project. Evaluating the results achieved by the per segment models in comparison to the complete data set model found that the full model performed better across all segments. As such, it seems that the full model was better able to differentiate between the customer segments, and the per segment models were able to identify the differences between the segments when all segments were included.

### Customer Segmentation Results

The customer segmentation did not achieve the results hope for. The segmentation for the model that provided the highest Silhoutte Score results as follows:

**Train Data**

| Segment | # of Search Sessions | # of Flights |
| :------ | :------------------- | ------------ |
| 0       | 74,673               | 12,313,885   |
| 1       | 9                    | 2,002        |
| 2       | 46                   | 13,441       |
| 3       | 2                    | 978          |


**Test Data**

| Segment | # of Search Sessions | # of Flights |
| :------ | :------------------- | ------------ |
| 0       | 20,843               | 3,506,639    |
| 1       | 36                   | 17,243       |
| 2       | 197                  | 53,189       |
| 3       | 34                   | 50,762       |

The segmentation results for the Test data seemed to perform better than the Train data, but the difference was not significant. The number of search sessions for each segment was similar, but the number of flights per segment was significantly higher for the Test data. 

While the LightGBM model used the cluster segmentation for the flight ranking prediction, since the 'customer_segment' feature was left in the data, the training of segment-based models performed worse than the single general model across the board.

Overall, this is disappointing for this phase of the project outcome. Further evaluation could be done to determine what impact the customer segmentation had by removing the feature. However, this would require training a new LightGBM models as well.

## Next Steps

Given that the customer segmentation did not meet expectations, the next steps are recommended:
1. Remove the 'customer_segment' feature from the training data set and rerun the process to evaluate the impact of the segmentation.
2. Identify additional customer features that would provide additional insights for segmentation
3. Identify additional interactive features between the customer and flight data that would improve behavioral or profile based matching.
4. Determine if additional data exists or could be made available that would augment the existing data set, such as the order flight search results were presented to the customer or any indications whether they viewed the specific flight.
5. Try additional unsupervised models
6. Evaluate other methods for encoding, scaling, and dimensionality reduction of the customer features.

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License. See the `LICENSE` file for details.

## Acknowledgements

 - [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection)
 - [IEEE-CIS Fraud Detection Data Description](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203)
 - [Vesta Corporation](https://www.vesta.io/)
    -  provided the competition project dataset files
 - [NT Project Summary: Part 3](https://www.kaggle.com/code/mariajunior/nt-project-summary-part-3/notebook)
    -  provided some starting parameters for evaluating LightGBM
 - [readme.so Editor](https://readme.so/editor)
    -  a handy online README.md editor by [Katherine Oelsner](https://www.katherineoelsner.com/)

## References
Jetbrains. (2025). *Jetbrains AI* (252.25557.171) [Large Language Model] https://www.jetbrains.com/ai/
 - used with PyCharm IDE for code assist and troubleshooting

Anthropic. (2025). *Claude Sonnet 4* (May 22, 2025) [Large Language Model] https://anthropic.com
 - the LLM used by PyCharm JetBrains AI for code assist and troubleshooting

## Video Overview
A video overview of the project can be accessed on YouTube: [CSCA-5622 Final Project](https://youtu.be/ZtnP777eYj8)
