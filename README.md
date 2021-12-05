# iTransplant: individualized TRANSparent Policy Learning for orgAN Transplantation (NeurIPS2021)

Source code for the iTransplant framework proposed in paper "Closing the loop in medical decision support by understanding clinical decision-making: A case study on organ transplantation".
ITransplant is a novel data-driven framework to learn interpretable organ offer acceptance policies directly from clinical data. 
It learns a patient-wise parametrization of the expert clinician policy that accounts for the differences between patients, a crucial but often overlooked factor in organ transplantation. 
We conducted several investigative experiments with real-world liver transplantation data from the Organ Procurement and Transplantation Network (OPTN), covering 190,525 organ offers. 
The results show that iTransplant can be used to probe clinical decision-making practices in a number of ways.
More specifically, our investigations allow us to: 
1. identify important match criteria for organ offer acceptance;
2. discover patient-wise organ preferences of clinicians via automatic patient stratification in a latent representation space;
3. examine the transplantation practice variations across transplant centers.

## Installation & Environment Setup

To run the experiment locally, directly clone this repository via the following command.
```bash
> git clone git@github.com:yvchao/iTransplant.git
```

We recommand creating the Python environment with **pyenv** (version 1.2.23) as follows.
```bash
> pyenv install 3.8.7
```
Python of version 3.8.7 is preferred for best compatibility and reproducibility.

Make sure to install necessary dependencies with the provided **requirements.txt** before conducting any experiment with iTransplant.
```bash
> pip install -r requirements.txt
```

## Data Preparation
The liver transplantation data from OPTN is used for experiments in the paper. Please refer to the README.md file under the **OPTN_data** directory for instructions on how to obtain the dataset.
The considered feature variables are listed in the *selected_features.py* file under the same path.
Note that the liver transplantation data, waiting list history data and match run data (organ cllocation and associated decisions on organ offers) are necessary.
The processed data need to be stored in HDF5 format with speficied keys. Please refer to [src/data_loading/data_utils.py](https://github.com/yvchao/iTransplant/blob/4fe0ce6962e109b020c440827d116fdca5b7617f/src/data_loading/data_utils.py#L7) for details.
Please put all processed data in the **data** folder to run experiments.

## Main Results

The main results in the paper can be obtained by executing the following notebooks:

- hyperparameter_selection.ipynb
- benchmark.ipynb
- investigative experiments.ipynb

Configurations used for hyperparameter selection and benchmark can be found in **hyperparameter_selection_configures.py**
and
**benchmark_configures.py**,
respectively.

Additional benchmark results in the Appendix of the paper can be obtained by executing the following notebooks:

- additional_hyperparameter_selection.ipynb
- additional_benchmark.ipynb

## Citation
If you find the software useful, please consider citing the following [paper](https://proceedings.neurips.cc/paper/2021/hash/c344336196d5ec19bd54fd14befdde87-Abstract.html):
```
@inproceedings{iTransplant2021,
  title={Closing the loop in medical decision support by understanding clinical decision-making: A case study on organ transplantation},
  author={Qin, Yuchao and Imrie, Fergus and Hüyük, Alihan and Jarrett, Daniel and Edward Gimson, Alexander and van der Schaar, Mihaela},
  booktitle={Advances in neural information processing systems},
  year={2021}
}
```

## Additional Notes
The experiments in the paper are conducted on a custom limited dataset (covering organ offers from January 1, 2003 to December 4, 2020) from [OPTN](https://optn.transplant.hrsa.gov/).

There could be minor variations in the experiment results due to differences in:
- Python versions
- Versions of third-party dependencies.
- Liver transplant data from OPTN

The iTransplant framework currently does **not** support execution on CUDA decives.

## License
Copyright 2021, Yuchao Qin.

This software is released under the GPLv3 license.
