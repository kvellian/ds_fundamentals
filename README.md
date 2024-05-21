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


## Data Exploration

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





<img src="assets/img/count_bar_plot.png" alt="count bar plot">
