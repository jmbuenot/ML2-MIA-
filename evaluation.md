Correction Guidelines for the Online Learning Project  
Note1: The scoring description is based on a 10-point practice, The final grade will be weighted at 4
points.  
Note2: the following main scored sections described a non-exhaustive list of possible items to be
evaluated.

1. Problem description (max. 1 point)  
   a. Informal problem description  
   i. The goals of the problem are stated in non-ML terms  
   b. Description of the problem characteristics from the ML viewpoint. Each of the following questions should be addressed:  
   i. What is the problem?  
   ii. Is this problem one of clustering, regression, or classification?  
   iii. Is the dataset of the problem imbalanced?  
   iv. Could the dataset of the problem be influenced by drift?  
   v. Have metrics to evaluate the model been described and are they appropriate?  
   vi. Are there any relevant assumptions for this addressing the problem?
2. Dataset Selection (max 0.5 points).  
   a. Maximum points shall be given to a dataset that is suitable for stream learning. A brief justification and explanation of suitability is provided.  
   b. The dataset is part of the River library or it is an external one. A River dataset is allowed but may not receive maximum points.  
   c. The dataset is already prepared, or it has required some preparation. A prepared dataset is allowed but may not receive maximum points.
3. Data preparation (max. 1.5 points)  
   a. A brief description is provided about how the data was studied in order to perform the data type conversions required by River.  
   b. If required, has the data been normalized or standardized? If so, has the motivation and procedure been shown and described?  
   c. If the dataset contains nominal features or the problem is a multiclass problem, has one-hot encoding been performed? The encoding scheme should be briefly described.  
   d. Is the definition of new features required? If so, a brief description should be provided.  
   e. Is the categorization of any features required?  
   f. Specific adaptations to the selected problem.
4. Concept drifts (max 1 point)  
   a. Has the project implemented at least the two required detectors? Which ones?  
   b. A brief description of why these detectors were selected should be provided.
5. Batch Learning (max 1 point)  
   a. Is the split correctly made, i.e., if required that data is stratified or grouped? Tip: Batch learning can be done by defining the pipelines in River and using the built-in wrapper to perform the remaining operations.  
   b. Have any model hyperparameter been tuned?  
   c. Have different models been compared? Have the models been correctly adjusted/compared? No data of the test is used in the training/validation phase  
   d. Is a cross-validation mechanism used?
6. Stream Learning (max 2 points)  
   a. Does the notebook contain at least 3 stream learning pipelines with their corresponding models  
   b. Are pipelines used correctly in the solution?  
   c. Is one of the models a Hoeffding Tree?  
   d. Are the metrics selected suitable to evaluate the performance of the models?
7. Notebook: Presentation (max 0.5 point)  
   a. The notebook has plots to support the provided arguments.  
   b. Are Notebooks informative and well written?  
   c. If the dataset is affected by concept drifts, are the drifts exemplify on those plots?  
   d. Is there a plot (or plots) that compare batch learning results with that from the stream learning approaches?
8. Notebook: Results and conclusions (max 0.5 point)  
   a. Are the conclusions supported by the results in the notebook?  
   b. Do the results and conclusions offer some open questions and future work?
9. Oral Presentation (max 2 points)  
   a. Is it organized?  
   b. Are student’s arguments clear?  
   c. Did the student correctly answer questions posed by the professors?
