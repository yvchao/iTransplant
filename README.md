# iTransplant

Source code for the paper "Closing the loop in medical decision support by understanding clinical decision-making: A case study on organ transplantation" submitted to NeurIPS 2021.

The prepared organ offer data used for our experiments are provided in the **data** folder.
The codes used for processing the liver transplantation data from OPTN are provided under the **data_preparing** directory.

To generate organ offer data from scratch, please put the transplant data, waiting list history and organ allocation data (match data) into the corresponding subfolders under the path **data_preparing/Raw Data**, and execute ***data_proprocessing.sh*** under the **data_preparing** directory.
The extracted organ offer history in selected transplant centers will be copied to the **data** folder.

The main results in our paper are illustrated via the following notebooks:
1. hyperparameter_selection.ipynb
2. benchmark.ipynb
3. investigative experiments.ipynb

The configurations used for hyperparameter selection can be found in ***hyperparameter_selection_configures.py***.
The configurations used for benchmark are given in ***hyperparameter_selection_configures.py***.


To reprocude our results, delete all files inside the **report** directory and rerun all three notebooks mentioned above.
