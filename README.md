# NMCB

This code gives the implementation  of the paper "Data augmentation with consistency regularization for multi-labeled complementary label learning". The paper and appendix can be found at [here](https://gaoyi439.github.io/papers/).

 Requirements
- Python >=3.6
- PyTorch >=1.10
- CUDA 11.3

---
## Run:
**main.py**
- This is main function. After running, you will see a .csv file with the results saved in the directory.
The results will have seven columns: epoch number, training loss, hamming loss of test data, one error of test data,
coverage of test data, ranking loss of test data and average precision of test data.

## Specify the dataset argument:
- bookmark15: bookmark dataset with 15 labels
- ml_tmc: tmc2007 dataset

**Please note: unzip the dataset before running the code.**

