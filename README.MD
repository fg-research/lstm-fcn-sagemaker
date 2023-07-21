# LSTM-FCN SageMaker Algorithm

This algorithm performs time series classification with the Long Short-Term Memory Fully Convolutional Network (LSTM-FCN) model. 
It implements both training and inference and supports both CPU and GPU instances.
The training and inference Docker images were built by extending the PyTorch 2.0 Python 3.10 SageMaker containers.

## Model Description
The LSTM-FCN model includes two blocks: a recurrent block and a convolutional block.
The recurrent block consists of a single LSTM layer followed by a dropout layer.
The convolutional block consists of three one-dimensional convolutional layers, each followed by batch normalization and ReLU activation, and of a global average pooling layer.
The input time series are passed to both blocks. 
The convolutional block processes each time series as a single feature observed across multiple timesteps, while the recurrent block processes each time series as multiple features observed at a single timestep.
The outputs of the two blocks are concatenated and passed to a multi-class classification layer with softmax activation. 

<img src=https://fg-research-assets.s3.eu-west-1.amazonaws.com/lstm-fcn-diagram.png style="width:80%;margin-top:30px;margin-bottom:20px"/>

*LSTM-FCN architecture (source: https://ieeexplore.ieee.org/abstract/document/8141873)*

## SageMaker Algorithm Description
### Training
The training algorithm has two input data channels: `"training"` and `"testing"`. 
The `"training"` channel is mandatory, while the `"testing"` channel is optional.
The data should be provided in a CSV file containing the time series and their class labels.
The class labels should be stored in the first column, while the time series values should be stored in the subsequent columns.
The class label values should be integers between `0` and `n_classes - 1`, where `n_classes` is the number of classes.
The CSV file should not contain any index column or column headers. 
See the examples `train.csv` and `valid.csv` in the `sample_data` folder.

#### Distributed Training
The algorithm supports multi-GPU training on a single instance, which is implemented through [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
The algorithm does not support multi-node (or distributed) training across multiple instances. 

### Inference
The inference algorithm takes as input a CSV file containing the time series.
The CSV file should not contain any index column or column headers.
See the example `test_data.csv` in the `sample_data` folder.
The inference algorithm outputs the predicted class labels in the same format as they were provided for training, that is as integers between `0` and `n_classes - 1`, where `n_classes` is the number of classes. 
The predicted class labels are returned in CSV format.
See the example `test_labels.csv` in the `sample_data` folder.

#### Endpoints
The algorithm supports only real-time inference endpoints. The inference image is too large to be uploaded to a serverless inference endpoint.

### Hyperparameters
The training algorithm takes as input the following hyperparameters:
- `units`: `int`. The number of units of the LSTM layer (default = 8).
- `dropout`: `float`. The rate of the dropout layer (default = 0.8).
- `filters-1`: `int`. The number of filters of the first convolutional layer (default = 128).
- `kernel-size-1`: `int`. The size of the kernel of the first convolutional layer (default = 8).
- `filters-2`: `int`. The number of filters of the second convolutional layer (default = 256).
- `kernel-size-2`: `int`. The size of the kernel of the second convolutional layer (default = 5).
- `filters-3`: `int`. The number of filters of the third convolutional layer (default = 128).
- `kernel-size-3`: `int`. The size of the kernel of the third convolutional layer (default = 3).
- `lr`: `float`. The learning rate used for training (default = 0.001).
- `batch-size`: `int`. The batch size used for training (default = 128).
- `epochs`: `int`. The number of training epochs (default = 2000).

### Metrics
The training algorithm logs the following metrics:
- `train:loss`: `float`. Training set loss.
- `train:acc`: `float`. Training set accuracy.

If the `"testing"` data channel is provided, the training algorithm also logs the following additional metrics:
- `test:loss`: `float`. Test set loss.
- `test:acc`: `float`. Test set accuracy.

## References
F. Karim, S. Majumdar, H. Darabi and S. Chen, "LSTM Fully Convolutional Networks for Time Series Classification," in IEEE Access, vol. 6, pp. 1662-1669, 2018, doi: 10.1109/ACCESS.2017.2779939.

