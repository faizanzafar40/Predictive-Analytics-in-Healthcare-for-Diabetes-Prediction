# Predictive Analytics in Healthcare for Diabetes Prediction

[Faizan Zafar](https://faizanzafar40.github.io/), Saad Raza, Muhammad Umair Khalid, Muhammad Ali Tahir 

This repository includes the code for the paper:

'**Predictive Analytics in Healthcare for Diabetes Prediction**', In Proceedings of the 2019 9th International Conference on Biomedical Engineering and Technology (ICBET' 19). ACM, New York, NY, USA, 253-259. 2019. [[DOI]](https://doi.org/10.1145/3326172.3326213)

The code is written in Python 3.7.0 and requires the following dependencies to run:
```
numpy 1.15.1
matplotlib 3.0.0
pytorch 0.4.1
seaborn 0.9.0
scikit-learn 0.21.3
tensorboard 1.13.1
tensorboardx 1.7
keras 2.2.4
pickle 3.0
tensorflow 1.13.1
pandas 0.24.2
scipy 1.1.0
virtualenv 16.7.3
```

## Abstract

Diabetes mellitus type 2 is a chronic disease which poses a serious challenge to human health worldwide. Globally, about 8.3% of the population is diagnosed with the disease. The applications of predictive analytics in diagnosis of diabetes are gaining significant momentum in medical research. The aim of this research paper is to aid medical professionals in the early detection and efficient diagnosis of Type 2 diabetes. We utilize bioinformatics theory and supervised machine learning techniques for improving the accuracy in predicting diabetes, based on 8 clinical measurements existing in the widely used PIMA dataset. We outline our methodology and highlight the implementation steps, while reviewing prominent past work in the field. Moreover, this paper fully exploits known machine learning algorithms and provides a detailed comparison of the results obtained from each method. The gradient boosting algorithm with parameter tuning proves to be the most successful, having an F1 Score of 0.853 and out of sample accuracy of 89.94%. Our prediction model focuses on computing the probability of the onset of diabetes in an individual based on their clinical data. The most crucial results of using this research within the healthcare sector are its cost-effectiveness and yielding of instant diagnosis. With this work, we intend to improve the process of diagnosing Type 2 diabetes and inspire other researchers to use machine learning based techniques for further inquiry into diabetes prediction.

 
## Dataset

Create a directory called `input` inside the parent directory containing the project files. Download the dataset from this 
[link](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and place it in the `input` directory.

## Running the code

After scrolling to the directory containing the project, open the command line and create a virtual environment:

```
virtualenv venv
```

Then, run the main project file:

```
python main_code.py
```

This runs the main_code.py file with the default settings for all parameters.

## Citation

If you found our code and paper useful, we humbly request you to cite our work:
```tex
@inproceedings{Zafar:2019:PAH:3326172.3326213,
 author = {Zafar, Faizan and Raza, Saad and Khalid, Muhammad Umair and Tahir, Muhammad Ali},
 title = {Predictive Analytics in Healthcare for Diabetes Prediction},
 booktitle = {Proceedings of the 2019 9th International Conference on Biomedical Engineering and Technology},
 series = {ICBET' 19},
 year = {2019},
 isbn = {978-1-4503-6130-9},
 location = {Tokyo, Japan},
 pages = {253--259},
 numpages = {7},
 url = {http://doi.acm.org/10.1145/3326172.3326213},
 doi = {10.1145/3326172.3326213},
 acmid = {3326213},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Bioinformatics, Diabetes Prediction, Gradient Boosting, Machine Learning},
} 

```

The paper is available at: [https://doi.org/10.1145/3326172.3326213](https://doi.org/10.1145/3326172.3326213)

## Questions?

If you have any questions regarding the code or dataset, you can contact me at (14besefzafar@seecs.edu.pk), or even better, open an issue in this repo and we will do our best to help.
