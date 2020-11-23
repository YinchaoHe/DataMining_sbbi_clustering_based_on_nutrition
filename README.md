# sbbi_clustering
This repository is used to group the similar recipes and find the patterns within calories, TotalFat, SaturatedFat, Cholesterol, Sodium, Potassium, TotalCarbohydrates, Protein, Sugars, VitaminA. 

## Description
- preprocess.py: This module is applied to extract nutrients information from recipes csv files. 
- clustering.py: In this module, three clustering methods, such as K-Means, Hierarchical, and DBSCAN, are employed to find the similarity between each recipe.
- data_statistics.py: This module is used to analyze and visualize the relationship between each nutrient. For instance, there are linear regression, Spearman correlation coefficience, mean, and standard deviation. Headmap, density curve, and 3D scatter plots are employed to show their result.

## Usage
- Extract the text files for one country from cuisine folders with images and text files into one folder whose name is given by users. (Warrning: You can only extract one country each time).
	* `python preprocess -c [country name]`
- Extract the nutrients information from cuisine text files into one csv file.
	* `python preprocess -c [country name] -n True`
- Group similar recipes using 3-Means method.
	* `python clustering.py`
