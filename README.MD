# LSTM-FCN SageMaker Algorithm

This algorithm performs time series classification with the Long Short-Term Memory Fully Convolutional Network (LSTM-FCN) model. 
It implements both training and inference and supports both CPU and GPU instances.
The training and inference Docker images were built by extending the PyTorch 2.0 Python 3.10 SageMaker containers.

<img src=https://fg-research-assets.s3.eu-west-1.amazonaws.com/lstm-fcn-diagram.png style="width:80%;margin-top:30px;margin-bottom:20px"/>

*LSTM-FCN architecture (source: https://ieeexplore.ieee.org/abstract/document/8141873)*

## Training
The training algorithm has two data channels: `"training"` and `"testing"`. 
The `"training"` channel is mandatory, while the `"testing"` channel is optional.
The data should be provided in a CSV file containing the time series and their class labels.
The class labels should be stored in the first column, while the time series values should be stored in the subsequent columns.
The class label values should be integers between `0` and `n_classes - 1`, where `n_classes` is the number of classes.
The CSV file should not contain any index column or column headers. 
See the examples `train.csv` and `valid.csv` in the `sample_data` folder.

### Distributed Training
The algorithm supports multi-GPU training on a single instance, which is implemented through [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
The algorithm does not support multi-node (or distributed) training across multiple instances. 

## Inference
The inference algorithm takes as input a CSV file containing the time series.
The CSV file should not contain any index column or column headers. 
See the example `test_data.csv` in the `sample_data` folder.
The inference algorithm outputs the predicted class labels in the same format as they were provided for training, that is as integers between `0` and `n_classes - 1`, where `n_classes` is the number of classes. The predicted class labels are returned in CSV format.
See the example `test_labels.csv` in the `sample_data` folder.

## Hyperparameters
The training algorithm takes as input the following hyperparameters:
- `units`: `int`. The number of units of the LSTM layer (default = 8).
- `dropout`: `float`. The rate of the dropout layer after the LSTM layer (default = 0.8).
- `filters-1`: `int`. The number of filters of the first convolutional layer (default = 128).
- `kernel-size-1`: `int`. The size of the kernel of the first convolutional layer (default = 8).
- `filters-2`: `int`. The number of filters of the second convolutional layer (default = 256).
- `kernel-size-2`: `int`. The size of the kernel of the second convolutional layer (default = 5).
- `filters-3`: `int`. The number of filters of the third convolutional layer (default = 128).
- `kernel-size-3`: `int`. The size of the kernel of the third convolutional layer (default = 3).
- `lr`: `float`. The learning rate used for training (default = 0.001).
- `batch-size`: `int`. The batch size used for training (default = 128).
- `epochs`: `int`. The number of training epochs (default = 2000).

## Metrics
The training algorithm logs the following metrics:
- `train:loss`: Training set cross-entropy loss.
- `train:acc`: Training set accuracy.

If the `"testing"` data channel is provided, the training algorithm also logs the following additional metrics:
- `test:loss`: Test set cross-entropy loss.
- `test:acc`: Test set accuracy.

## References
Karim, F., Majumdar, S., Darabi, H. and Chen, S., 2017. LSTM fully convolutional networks for time series classification. *IEEE access*, 6, pp.1662-1669.








