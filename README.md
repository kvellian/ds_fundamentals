<img src="assets/img/headshot_circle_cropped.png" alt="Data Science Portfolio - Ken Vellian" width="200" height="200">

## Dataset

This dataset, penguins_size.csv, was sourced from Kaggle.
- [Kaggle Data Source: penguins_size.csv](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data)

- "Cite: Data are available by CC-0 license in accordance with the Palmer Station LTER Data Policy and the LTER Data Access Policy for Type I data."

- "Gorman KB, Williams TD, Fraser WR (2014) Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis). PLoS ONE 9(3): e90081. doi:10.1371/journal.pone.0090081"

## RMarkdown Code

- [View Data Science Fundamentals Report]()

## Purpose

This project focuses on the fundamentals of data science to explore machine learning algorithms, with R, using this Penguin Sizes data.

Part 1. Data Exploration

Part 2. Data Cleaning

Part 3. Data Preprocessing

Part 4. Clustering

Part 5. Classification 

Part 6. Evaluation


## Part 1. Data Exploration

- Original data: 344 rows, 7 columns

**Categorical Variables**

| Variable    | Description                                           | Unique Values                      |
|-------------|-------------------------------------------------------|------------------------------------|
| Species     | Penguin species                                      | Adelie, Chinstrap, Gentoo         |
| Island      | Island name in the Palmer Archipelago (Antarctica)   | Torgersen, Biscoe, Dream           |
| Sex         | Penguin gender                                       | MALE, FEMALE, NA, “.”              |

**Numerical Variables**
| Measurement           | Mean   | Min    | Max    |
|-----------------------|--------|--------|--------|
| Culmen Length (mm)    | 43.92  | 32.10  | 59.60  |
| Culmen Depth (mm)     | 17.15  | 13.10  | 21.50  |
| Flipper Length (mm)   | 200.90 | 172.00 | 231.00 |
| Body Mass (g)         | 4202   | 2700   | 6300   |

After exploring the data further, there appears to be 2 NA's in each of the columns:
culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g. 


At first glance, the sex column has 10 NAs with 1 entry with the unique chr of ".". This "." appears to be a placeholder for an unknown data point/ missing data. In total, there are 11 NAs, considering the ".".


## Part 2. Data Cleaning

Let's consider the missing NAs and determine how much effect they have on the penguin_size data. First, I'll convert the single "." entry in the sex column to NA.

After calculating the proportion of missing data per column to see the significance of its impact on the data, here are the results:

| Variable            | Missing Percentage |
|---------------------|--------------------|
| species             | 0%                 |
| island              | 0%                 |
| culmen_length_mm    | 0.58%              |
| culmen_depth_mm     | 0.58%              |
| flipper_length_mm   | 0.58%              |
| body_mass_g         | 0.58%              |
| sex                 | 3.20%              |

The sex column has the most significant impact on the data, with 3.20% contributing to missing NAs. As opposed to 0.58% for culmen_length_mm, culmen_depth_mm, flipper_length_mm, and body_mass_g.


Considering, 3.20% is relatively low, I will drop the NAs, since its removal should not have too large of an effect on predictability. After dropping the NAs, 11 rows have been removed, and the dataset now contains 333 rows now.


## Part 3. Data Preprocessing

In this step, let's create 2 new columns:
- culmen_size_mm2: This will be the value of (culmen length x culmen depth), to explore the data more and see if adding this variable can help with prediction accuracy.

  
- body_mass_lbs: Converting the body_mass_g column from grams to pounds with this conversion formula: body_mass_lbs = body_mass_g / 453.59237

Next, I'll convert the categorical variables into factors. I'll also prepare a subset of the penguin_size data without the class labels to perform clustering. Afterward, I'll preprocess the data using the center-scaled method.

      # Creating subset of penguin_size data to perform clustering
      # Removing class labels
      predictors_clustering <- penguin_size %>% select(-c(species, island, sex))


Since, we're going to perform predictions using other classifiers, I'll partition the data after using a 70/30 train-test split and preprocess the data when creating the classifier models.

      # Center scale allows us to standardize the data
      preproc <- preProcess(predictors_clustering, method = c("center", "scale"))
      
      # We have to call predict to fit our data based on preprocessing
      # Using predictors_clustering for Clustering
      predictors_clustering <- predict(preproc, predictors_clustering)
      
      # Partitioning the data to be used in Classification
      index = createDataPartition(y = penguin_size$sex, p = 0.7, list=FALSE)
      
      # Everything in the generated index list
      train_penguin = penguin_size[index,]
      
      # Everything except the generated indices
      test_penguin = penguin_size[-index,]

## Part 4. Clustering

Let's use kmeans to determine the best number for K before clustering. Using the plots for the Within Sum of Squares method and the Silhouette method.
- Using the Within Sum of Squares method, K = 4 represents the last non flat slope, or some argue that we should use K = 5 because that is the last point before the slope goes flat.
- Using the Silhouette method, K = 2.

- 



<img src="assets/img/count_bar_plot.png" alt="count bar plot">
