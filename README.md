# Battery Management System Thesis

This repository contains the code and data used in the research thesis on Battery Management Systems (BMS) for battery state monitoring. The purpose of this thesis was to compare the performance of Equivalent Circuit Models (ECM) and Neural Network (NN) models, specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM), for terminal voltage estimation.

## Requirements

- Python 3.9

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r Requirements.txt
   ```

## Data

To run the Jupyter notebooks, you need to download and extract the following data archive:

- [Battery Data Archive](https://example.com/data_archive.zip)

Place the extracted data files in the appropriate directories as specified within the Jupyter notebooks.

## Code Structure

The code in this repository is organized as follows:

### Jupyter Notebooks

1. [1_ecm_comparison.ipynb](1_ecm_comparison.ipynb)
2. [2_generate_ecm_training_data.ipynb](2_generate_ecm_training_data.ipynb)
3. [3_cnn_trainer.ipynb](3_cnn_trainer.ipynb)
4. [4_lstm_trainer.ipynb](4_lstm_trainer.ipynb)
5. [5_ecm_cnn_lstm_comparison.ipynb](5_ecm_cnn_lstm_comparison.ipynb)
6. [6_cnn_fine_tuning.ipynb](6_cnn_fine_tuning.ipynb)

### Python Files

- [generate_training_data.py](generate_training_data.py)
- [Nth_Order_ECM.py](Nth_Order_ECM.py)
- [tools.py](tools.py)

## Summary

Battery management systems are crucial for monitoring the state of batteries and preventing hazards by ensuring they do not exceed critical values. This thesis focuses on comparing two categories of battery models: Equivalent Circuit Models (ECM) and Neural Network (NN) models, specifically CNN and LSTM. The comparison is based on runtime performance and prediction accuracy for terminal voltage estimation.

To generate the training data, a 2nd order ECM was utilized. The generated data were then used to train two NN models. Finally, the ECM and NN models were compared based on the specified criteria. The results showed that NN models can achieve ECM-similar performance for terminal voltage estimation with significantly faster execution time.

Additionally, an investigation was conducted to determine if the accuracy of the NN models could be improved using non-ECM-generated data. It was found that this was not the case.

## Conclusion

This thesis involved selecting an ECM, generating data, and evaluating the performance of NN models for battery monitoring, specifically for terminal voltage estimation. The simulation software was developed using Python and the PyTorch framework. The software provides insights into the strengths and limitations of the models for the given task.

The ECM class was implemented, and two ECM models of 1st and 2nd order were instantiated and compared in terms of prediction and runtime performance. The 1st order ECM showed better runtime performance, while the 2nd order ECM had slightly better prediction performance.

Subsequently, training data was generated using the 2nd order ECM for the NN models. The CNN and LSTM models were found to be suitable for the task, with the CNN outperforming the LSTM in terms of prediction accuracy, comparable to the ECM model. However, the CNN's runtime was higher than the LSTM and ECM models.

Attempts were made to improve the CNN's prediction performance by fine-tuning the already trained model. The last layer of the CNN was replaced with a fully connected layer and a new output layer. The previously trained layers were frozen, and the new layers were trained using non-ECM-generated data from the BMW i3 driving dataset, scaled down to match the battery specification of this thesis. These enhancements resulted in a slight improvement in prediction performance, but it did not surpass that of the ECM model.

The results demonstrate that NN models, particularly CNN, are viable alternatives to ECM for battery modeling. However, NN models require a significant amount of training data, which can be time-consuming if collected directly from batteries. The ECM provides a valuable solution by generating data rapidly for feeding into the NN models.

Future improvements to this work include implementing more error messages, further investigating the LSTM model, addressing the sensitivity of NN models to input feature order, and expanding battery state estimation beyond terminal voltage to include temperature, aging, etc.

For more details, please refer to the complete thesis document.
