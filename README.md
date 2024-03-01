# Offside

## Introduction, Reference Domain, and Objectives
The Offside project starts from the need of creating a recommendation system based on machine learning which, given a football player, returns a series of players as similar as possible based on a set of technical-tactical data. The reference domain is, obviously, the world of football, and the reference platform for data retrieval is Fbref, a consultation source that includes data relating to players, teams and championships in an easily to consult, -and scrape-, fashion.

The approach that the project intends to adopt is based on a rather simple notion: let's assume we have a two-dimensional Cartesian plane, where each point corresponds to a player based on a pair of technical characteristics of our choice. The players most similar to the original player will be those who have the smallest distance on the Cartesian plane. It is sufficient to raise the plane to n dimensions, one for each of the technical attributes considered, to evaluate the similarity of the players on the hyperplane. The only caveat is that calculating the distance on a n-dimensional plane requires a little bit more effort than doing the same thing for a bi-dimensional plane, but thanks to the use of the Euclidean distance, the problem will be solved quite easily. In the code block that follows, we will begin by importing all the necessary libraries for the model.

## Data Retrieval, Parsing, and Dataframe Construction
This section aims to retrieve data from a Web URL. The specified URL points to a webpage on FBref, a website that, as we mentioned earlier, provides comprehensive football statistics and tables. In particular, the URL variable holds the address of the webpage containing Serie A statistics. The `pd.read_html()` function in the code down below is utilized to parse the HTML content of the webpage and extract tables. However, prior to parsing, the `.replace()` method is employed to remove HTML comment tags ('<!--' and '-->') from the webpage content. This is done to ensure that the HTML can be imported correctly and without hidden elements ruining the formatting, so that it may then be parsed by `read_html()`. This workaround used was found with a quick search on StackOverflow (https://stackoverflow.com/questions/77548912/how-to-extract-table-from-a-webpage-using-its-id)

We then have the `pd.read_html()` function call, inside of which the `requests.get(url).text` retrieves the HTML content of the specified URL, which is then passed to `read_html()` for parsing. The `attrs` parameter is utilized to specify the HTML attribute `id` of the table element that we actually need to be extracted, which in this case is 'stats_standard', a table containing attributes for the players in Serie A. The `[0]` index is appended to the end of the function call to select the first table found on the webpage matching the specified attributes.

## Data Preparation, Cleaning, and Optimization

After scraping the data, we need to prepare it for further analysis and processing. First of all, the column names are re-defined to enhance clarity and comprehension of the dataset. These new column names are structured hierarchically in order to categorize various aspects of player information, performance metrics, and expected outcomes. After the renaming process, the data types of some specific columns are adjusted to ensure consistency and suitability for analysis. Columns containing strings are converted to the string data type, while categorical columns are converted to categorical data type for efficient storage and computation. Additionally, columns representing numerical data are converted to numeric type, with appropriate error handling to manage any inconsistencies or invalid entries. As the model will be based on the computation of the n-Dimensional Euclidean Distance (where n is the amount of attributes considered), we will need several numerical values in order to make a model that is precise and consistent.

Furthermore, unnecessary information is removed from the dataset to streamline its structure and optimize the analytical processes. Columns such as alphabetical rank and nationality are deemed irrelevant for the intended analysis, as they do not contribute significantly to player comparisons or performance evaluations. In the same way, redundancies are removed: we already have the year of birth of a player, so the age attribute is irrelevant and essentially duplicate information. Additionally, a redundant hyperlink column (in particular, the last column, provided by Fbref as a Hyperlink to another page) is eliminated. Finally, the processed dataset is exported to a CSV file named "serieA.csv" to facilitate future analysis and sharing. What follows is a description of the columns of the dataset as provided by the website.

### Dataset Columns Explanation

#### Background Information
1. **(Background Information, Alphabetical Rank)**: represents the alphabetical ranking of the players *[REDACTED]*
2. **(Background Information, Full Name)**: indicates the full name of the player
3. **(Background Information, Nation)**: specifies the nationality of the player *[REDACTED]*
4. **(Background Information, Position)**: specifies the primary position of the player
5. **(Background Information, Squad)**: denotes the squad the player belongs to
6. **(Background Information, Age)**: represents the age of the player *[REDACTED]*
7. **(Background Information, Year of Birth)**: indicates the year of birth of the player

#### Playing Time
8. **(Playing Time, MP)**: stands for Matches Played.
9. **(Playing Time, Starts)**: denotes the number of matches started by the player
10. **(Playing Time, Min)**: represents the total minutes played by the player
11. **(Playing Time, 90s)**: indicates the total minutes played divided by 90

#### Performance
12. **(Performance, Gls)**: represents the number of goals scored by the player
13. **(Performance, Ast)**: denotes the number of assists made by the player
14. **(Performance, G+A)**: indicates the total number of goals and assists combined
15. **(Performance, G-PK)**: represents the number of goals scored excluding penalty kicks
16. **(Performance, PK)**: denotes the number of penalty kicks scored by the player
17. **(Performance, PKatt)**: represents the number of penalty kicks attempted by the player
18. **(Performance, CrdY)**: indicates the number of yellow cards received by the player
19. **(Performance, CrdR)**: denotes the number of red cards received by the player

#### Expected
20. **(Expected, xG)**: represents the expected goals for the player
21. **(Expected, npxG)**: denotes the non-penalty expected goals for the player
22. **(Expected, xAG)**: represents the expected assists for the player
23. **(Expected, npxG+xAG)**: indicates the sum of non-penalty expected goals and expected assists

#### Progression
24. **(Progression, PrgC)**: stands for progressive carries made by the player
25. **(Progression, PrgP)**: denotes progressive passes made by the player
26. **(Progression, PrgR)**: represents progressive carries made by the player

#### Per 90 Minutes
27. **(Per 90 Minutes, Gls)**: represents the average number of goals scored per 90 minutes
28. **(Per 90 Minutes, Ast)**: denotes the average number of assists made per 90 minutes
29. **(Per 90 Minutes, G+A)**: indicates the average number of goals and assists combined per 90 minutes
30. **(Per 90 Minutes, G-PK)**: represents the average number of goals scored excluding penalty kicks per 90 minutes
31. **(Per 90 Minutes, G+A-PK)**: denotes the average number of goals and assists combined excluding penalty kicks per 90 minutes
32. **(Per 90 Minutes, xG)**: represents the average expected goals per 90 minutes
33. **(Per 90 Minutes, xAG)**: denotes the average expected assists per 90 minutes
34. **(Per 90 Minutes, xG+xAG)**: indicates the average sum of expected goals and expected assists per 90 minutes
35. **(Per 90 Minutes, npxG)**: represents the average non-penalty expected goals per 90 minutes
36. **(Per 90 Minutes, npxG+xAG)**: denotes the average sum of non-penalty expected goals and expected assists per 90 minutes
37. **(Hyperlink Column)**: shows a hyperlink to another page of the website *[REDACTED]*

These columns collectively provide a comprehensive overview of player performance, background, and progression metrics, facilitating in-depth analysis and insights as the further progression of the model will demonstrate.

## Parametric Configuration, Algorithmic Definition, and Data Modelling

The following block of code is dedicated to the configuration of the model, which was designed to be as modular as possible in order to be reused according to the individual needs of sports data analysis professionals. In particular, the code first allows us to specify the name and surname of the player we intend to take as a reference. Simply pass a string to the code containing the name of the player contained in the Column `(Background Information, Full Name)`, for example `reference_player_name = "Francesco Acerbi"`. Obviously, the player must be present in the Table, otherwise an error will be thrown. We then have a role filter, called `role_filter`. If this variable is set to `None`, then no filter will be used and the search will take place taking the players' statistical differences as a guideline. If, however, we have a specific position in mind for which to scout a player, for example the goalkeeper, just write `role_filter = "GK"`, and all the players returned will be players with `"GK"` as the value of their `(Background Information, Position)` column. This can be particularly useful when the starting point of the search is a goalkeeper with good feet (i.e. Mike Maignan), who due to his high ability to generate assists, without the filter activated could be compared with defenders or midfielders

We then have the variable `number_of_players_returned`, which indicates the quantity of players that will be returned by the algorithm. By default, this parameter is set to 5, but you can change it to have a longer list of similarities. It goes without saying that the longer the list, the less similar the values at the bottom of the list are to the original player. Finally we have the `relevant_cols` Array, which contains a list chosen by the analyst of the attributes to be included in the Euclidean comparison of the players. The columns to be inserted must have the same name as the corresponding columns on the dataframe, paying careful attention to Case Sensitivity: uppercase and lowercase are treated differently, and this could cause errors. By default, the vast majority of the columns present have been selected, as this allows for an all-round analysis of the players and their respective comparisons between them. However, it is possible to remove or add values (simply by commenting the respective line) based on scouting needs

After the data preparation and the configuration phase, we can finally enter the modelling phase. The data pre-processing begins with the removal of any rows containing missing values (`NaN`) from the DataFrame containing the football player data, as this will avoid potential errors down the line. Relevant columns (`relevant_cols`) are then selected in order to describe various aspects of the player such as background information, playing time, performance metrics, expected goals, progression, and statistics per 90 minutes. In order to facilitate the normalization process, only the numeric columns are retained from the array of containing the relevant columns. The data is then normalized for each column, in order to create a fairer distribution and result in the evaluation phase.

The real insight behind the construction of the Model is the idea of mapping football players in an n-dimensional plane, based on their statistics and technical characteristics. To do this, we use the Euclidean distance formula, which is applicable to vectors of any size. Instead of implementing it manually, we will use the SciPy `euclidean` Function (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html), which we have already imported previously with `from scipy .spatial.distance import euclidean`. This formula can be used to calculate the distance between two points in n-dimensional space, and in our example it fits perfectly to the size of the vectors involved. So, whether the vectors are two-dimensional, three-dimensional or higher dimensional, the formula remains the same, and this allows us to reason at an n-dimensional level as we would have done at a two-dimensional level: neighboring points in space equal players with similar characteristics. For completeness, here is the Euclidean Formula used, where `u` and `v` are two vectors of dimension `n`

$$\text{euclidean}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$$

The code will compute the Euclidean distances between the provided reference player and all other players in the dataset, filtering by role if specified. The distances are stored in a dictionary where the keys represent player indices and the values represent their respective distances from the reference player. These distances are then sorted in ascending order. The players most similar to the reference player are extracted based on their ascending Euclidean distances, defined as `Delta` in the printed summary, to represent a numerical indicator of distance between the reference player and another player (the lower the delta value, the better the similarity). Finally, the results are printed, formatted for having good readability and insights. If a role filter is applied, a note is provided regarding its influence on the returned results, to avoid potential oversights. It is also good to mention the speed of execution. The Euclidean Distance using the following code can be calculated in `O(n)` time, where `n` is the dimension of the input vectors. This is because the calculation involves only sums, multiplications, and square roots, which can be performed in linear time with respect to the size of the vectors. Therefore, we can also decide to have a huge list of players returned to us, and everything will be calculated in a matter of milliseconds.

## Data Visualisation and Plotting


To better understand the results obtained, we will exploit two different plotting techniques, the first will be a classic Horizontal Bar Plot, while the second will be a Radar Plot. Starting with the Bar Plot contained in the following code, this rather simple and intuitive to understand graph shows the results of the distance of the `n` players most similar to the player taken as reference. The x-axis of the graph shows the Delta, i.e. the distance coefficient between the reference player and the players most similar to him. The lower the value, the better the similarity between the two. The y-axis shows the name of the players, ordered from most to least similar to the player who gives the title to the table (in other words, the players are presented in ascending order based on the value of Delta). The color of the bars corresponding to the players displayed in the graph will be useful to visualize at a glance the team to which they belong, as shown in the legend. The primary purpose of this graph is to give a quick and shareable visualization of the result obtained by the algorithm, useful as a clarifying example for non-technical personnel who may need to use the information produced. However, this is not the most in-depth way of analyzing the differences, which is why it will not be the only view adopted.

The real complication arising from plotting data beyond three dimensions comes from finding a compromise between making the data as comprehensible as possible, without at the same time giving up the depth and insights they offer. After an analysis of the graphs that can be created with Python (an excellent reference material in this regard is the Python Graph Gallery, that can be found at this URL: https://python-graph-gallery.com), it was clear that the best choice for the problem at hand was the Radar Plot, also known as Spider Plot or Polar Chart. The Plot contained in the following code block was created using MatPlotLib (the documentation used to build the next code block can be found on https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html, and scattered around StackOverflow Threads and Online Forums), through the creation of the `radar_plot` function. This function takes as input several parameters `player_names`, `player_data`, `reference_data`, `attribute_labels`, and `reference_player_name`. Using the values contained in them, it constructs an informative display, complete with legend. This is by far the most useful graph, as it allows you to observe multivariate data in a clear and efficient way, allowing easy comparison of players based on their attributes. 

The way the Calculation Model is structured generates an interesting graphical phenomenon. As long as you keep the list of suggested players within responsible limits, the plotting creates a clear silhouette of their areas of excellence, -or lack-. Since the model calculates the delta via Euclidean distance, the resulting plots are denoted by a similar shape and distribution within the radar space. Furthermore, such a visualization allows you to focus at a glance on attributes that may be more interesting than others. For example, you can focus on the number of assists you expect, find the player who has the most playing time, or even find the player most capable of scoring goals; and in the list of proposed players find the one who is considered best in that aspect. In this way, the graph not only highlights the actual similarity of the players compared to the reference player, but also proves to be a valuable tool in moments when it is necessary to make an informed decision based on the available data.

## In Conclusion

The Offside project was born with the idea of developing a solid, modular and efficient recommendation system, trying to offer personalized suggestions on football players based on the analysis and modeling of a constantly updated set of technical-tactical data. By exploiting a multidimensional approach based on the use of the expansion of a Cartesian plane and Euclidean distances, the project aimed to identify football players with the best similarity to a player taken as a reference.

Central to the project was the idea of building an algorithm from scratch that was capable of exploiting Euclidean distance calculations to quantify player similarities in an n-dimensional space. The resulting model was able to compute the required similarities, doing so with high computational efficiency, and minimizing the time investments required to use the model in practice. Not only this, but the model also presents great modularity: a professional in the analytical sector can easily modify its parameters to obtain a personalized and satisfactory result based on his needs.

Everything is completed by the visualization aspect, which provides intuitive representations that can be used not only by an analyst, but also by non-technical personnel, who can thus easily understand the insights deriving from the model. By mixing data-driven methodologies, a little creativity and machine learning, the project succeeded in its aim of building a framework capable of facilitating informed decision-making, optimizing player selection strategies in the dynamic world of football.