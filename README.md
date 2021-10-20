# iTransplant

Source code for the paper "Closing the loop in medical decision support by understanding clinical decision-making: A case study on organ transplantation" submitted to NeurIPS 2021.

## Experiment environment set up

The experiments are conducted under the Windows Subsystem for Linux ([**WSL 2**](https://docs.microsoft.com/en-us/windows/wsl/)).
The installed linux distribution is **Debian GNU/Linux 10 (buster)** with the following kernel:
```bash
> uname -a
Linux XXPro-13 4.19.104-microsoft-standard #1 SMP Wed Feb 19 06:37:35 UTC 2020 x86_64 GNU/Linux
```

The python environment is created with **pyenv** version 1.2.23 via 
```bash
> pyenv install 3.8.7
```

Please install necessary dependencies with the provided **requirements.txt** via
```bash
> pip install -r requirements.txt
```

## Data preparation

To prepare liver transplantation data for experiments, please put the transplantation data from OPTN into the **data_preparing/Raw Data** folder according to the following structure:

- Transplant Data:
    - LIVER_DATA.DAT -- Liver transplatation data
    - LIVER_DATA.htm -- Column definitions

- Waiting List History
    - LIVER_WLHISTORY_DATA.DAT -- Waitlist history data
    - LIVER_WLHISTORY_DATA.htm -- Column definitions

- Match Data
    - PTR.DAT -- Organ allocation and associated decisions history (Match Run data)
    - PTR.htm -- Column definitions
    - MATCH_CLASSIFICATIONS.DAT -- Classification of organ allocation types
    - MATCH_CLASSIFICATIONS.htm -- Column definitions

The codes used for processing the liver transplantation data from OPTN are provided under the **data_preparing** directory.
By execute the following script ***under the root directory*** of this project, the extracted decision history in considered transplant centers will be copied to the **data** folder.
```bash
> bash data_processing.sh
```
All of our experiments are based on the processed data in the **data** folder.


## Reproduce the main results

The main results in our paper can be obtained by executing the following notebooks:

1. hyperparameter_selection.ipynb

2. benchmark.ipynb

3. investigative experiments.ipynb

To reprocude our results, delete all files inside the **report** directory and rerun all three notebooks mentioned above.

Configurations used for hyperparameter selection and benchmark can be found in **hyperparameter_selection_configures.py**,
and
**hyperparameter_selection_configures.py**,
respectively.

## Note on reproducibility
The experiments in our paper are conducted on a custom limited dataset (covering organ offers from January 1, 2003 to December 4, 2020) from OPTN.

There could be minor variations in the experiment results due to differences in:
- Operating systems
- Python versions
- Versions of third-party dependencies. 
- Liver transplant data from OPTN
