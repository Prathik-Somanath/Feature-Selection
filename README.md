# CS 205: Artificial Intelligence - Project 2

## Project Title: Feature Selection with Nearest Neighbor

### Author
Prathik Somanath  
SID: 862467832  
Email: [psoma005@ucr.edu](mailto:psoma005@ucr.edu)  
Date: June 12th, 2024

### Introduction
This project focuses on the implementation of feature selection techniques using the nearest neighbor algorithm for classification. The goal is to increase the accuracy of the nearest neighbor classifier by selecting the most relevant features from a given dataset.

### Feature Selection Methods
1. **Forward Selection**: Starts with no features and iteratively adds the most relevant feature at each step to improve the model's accuracy.
2. **Backward Elimination**: Starts with all features and iteratively removes the least relevant feature at each step to improve the model's accuracy.

### Datasets
Two datasets were used in this project:
1. **CS205_small_Data__10**: Contains 12 features and 500 instances.
2. **CS205_large_Data__26**: Contains 50 features and 5000 instances.

### Results Summary
#### Small Dataset (CS205_small_Data__10)
- **Forward Selection**: Achieved the highest accuracy of 94.6% with features {2, 8}.
- **Backward Elimination**: Also achieved the highest accuracy of 94.6% with features {2, 8}.

#### Large Dataset (CS205_large_Data__26)
- **Forward Selection**: Achieved the highest accuracy of 97.5% with features {8, 23, 47}.
- **Backward Elimination**: Achieved the highest accuracy of 83.8% with features {28, 47}.

### Conclusion
- For the small dataset, both forward selection and backward elimination identified features {2, 8} as the most relevant, achieving an accuracy of 94.6%.
- For the large dataset, forward selection outperformed backward elimination with an accuracy of 97.5% using features {8, 23, 47}.
- The project demonstrates the effectiveness of feature selection methods in improving the performance of nearest neighbor classifiers.

### Computational Effort
- **Small Dataset**: Forward Selection took 0.3 seconds, and Backward Elimination took 0.5 seconds.
- **Large Dataset**: Forward Selection took 45 minutes, and Backward Elimination took 86 minutes.

### Code
The complete code for this project is available on [GitHub](https://github.com/Prathik-Somanath/Feature-Selection).

### Acknowledgements
In completing this project, I consulted:
- Professor Dr. Eamonn Keogh for verifying the results.
- Pseudo code from the Project 2 Briefing slide.

### Contact
For any inquiries or questions, please contact Prathik Somanath at [psoma005@ucr.edu](mailto:psoma005@ucr.edu).

---
