![image](https://user-images.githubusercontent.com/99521606/219157620-e804abc9-887c-408b-9438-2ae532ccb837.png)

# Getting to the 'Heart' of the Problem


## Project Description


My main goal here, is to, using a heart disease dataset collected for the year 2020, use specific features to predict whether or not someone is at risk for heart disease. This will be a classification project, and I plan on creating clusters to see if they will be of assistance during this. 


## Goals


- Identify some of the main drivers in the cause of heart disease.

- Use those drivers to build clusters, and models to predict if someone is at risk.

- Create and deliver a final report that someone non-technical can read and grasp the key takeaways.


## Initial Questions and Hypotheses


- My initial hypothesis here is that the biggest drivers of heart disease will be:

    - bmi
    
    - physical health
    
    - difficulty walking
    
    - age
    
    - physically active

- My main questions are:

    - How closely correlated are physically active and bmi.
    
    - How the stroke column correlates with heart disease.
    
    - If physical and mental health are related.
    
    - If someone being diabetic and their bmi are related to each other.
        
- I won't be stopping with these, but they will most likely be the focus of my initial exploration.


## Data Dictionary


|Feature|Definition|
|:----------|:-------------|
|heart_disease|**Target Variable**; Whether or not the patient has heart disease|
|bmi|The patients' bmi (body mass index)|
|smoker|Whether or not the patient smokes|
|heavy_drinker|Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week|
|stroke|Has the patients ever had a stroke|
|physical_health|Including physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days)|
|mental_health|For how many days during the past 30 days was your mental health not good? (0-30 days)|
|difficulty_walking|Does the patient have serious difficulty walking or climbing stairs?|
|sex|Patients sex; male or female|
|age_group|Ages grouped up for every few years (18-24, 25-29, 30-34, ect)|
|race|The patients ethnicity (White,  black, hispanic, ect.)|
|diabetic|Ever been diagnosed with diabetes|
|physically_active|Adults who reported doing physical activity or exercise during the past 30 days other than their regular job|
|general_health|Would you say that in general your health is (Fair, good, poor, very good, ect.)|
|sleep_time|On average, how many hours of sleep do you get in a 24-hour period?|
|asthma|Ever been diagnosed with asthma|
|kidney_disease|Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?|
|skin_cancer|Ever been diagnosed with skin cancer|


## My Plan


- Acquire my data (through kaggle)

- Explore data (confirm there are no nulls, check for outliers, make sure data types are how I want to work with them, ect.)

- Create visuals for each feature, and then, after properly splitting data, compare features visually.

- Answer and dig deeper into initial questions and assumptions.

- Develop clusters to assist with the classification models further down the line. 

- Create classificarion models to determine heart diesase risk by:

    - Using the drivers identified by my exploration and questions.
    
    - Use those drivers to build models and evaluate them based upon my train and validate subsets. 
    
    - Identify my top performing models.
    
    - Choose my top model and run it on my test subset.
    
- Draw conclusions based upon my report.


## Steps to Reproduce


1. Clone this Repository

2. Acquire the dataset, through Kaggle.

3. Store the data within the cloned repository.

4. Run the final report notebook.


## Conclusions


- Clustering was not particularly helpful for the end goal of this project, at least on the features I clustered on.

- BMI was by far the most significant factor in heart disease, according to this dataset.

- Age group, sleep time, and health were the most important factors (in that order).

- The kidney disease and drinking columns were the worst two when it comes to helping predict the target.


## Recommendations


- Based on the results of all of my models, my XG Boost model was far above the rest. Therefore I would recommend this model for production use.

- Additionally, I would recommend we send out some sort of communication with each patient predicted to be at risk with the model, so that they can take steps to ensure their health.


## Further Steps


- There are a few things I would like to consider, going forward with this project:

    - I would like to compare more of the features with each other, both to get a better idea of how each relates to the other, but to potentially build clusters that end up being helpful.

    - I would also like to take more time to build more advanced functions, like with the xg boost function I have, to more quickly iterate through more models for the best time to performance efficiency.
