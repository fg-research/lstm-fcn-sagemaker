# LSTM-FCN SageMaker Algorithm
The [Time Series Classification (LSTM-FCN) Algorithm from AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-vzxmyw25oqtx6) performs time series classification with the Long Short-Term Memory Fully Convolutional Network (LSTM-FCN). 
It implements both training and inference from CSV data and supports both CPU and GPU instances.
The training and inference Docker images were built by extending the PyTorch 2.0 Python 3.10 SageMaker containers.

## Model Description
### Model Architecture
The LSTM-FCN model includes two blocks: a recurrent block and a convolutional block.
The recurrent block consists of a single LSTM layer (either general or with attention) followed by a dropout layer.
The convolutional block consists of three convolutional layers, each followed by batch normalization and ReLU activation, and of a global average pooling layer.

The input time series are passed to both blocks.
The convolutional block processes each time series as a single feature observed across multiple time steps, while the recurrent block processes each time series as multiple features observed at a single time step (referred to as dimension shuffling).
The outputs of the two blocks are concatenated and passed to a final output layer with softmax activation.
The model parameters are learned by minimizing the cross-entropy loss.

<img src=https://fg-research-assets.s3.eu-west-1.amazonaws.com/lstm-fcn-diagram.png style="width:80%;margin-top:30px;margin-bottom:20px"/>

*LSTM-FCN architecture (source: [doi: 10.1109/ACCESS.2017.2779939](https://doi.org/10.1109/ACCESS.2017.2779939))*

**Model Resources:** [Paper](https://doi.org/10.1109/ACCESS.2017.2779939) [Code](https://github.com/titu1994/LSTM-FCN).

## SageMaker Algorithm Description
The algorithm implements the model as described above with no changes. 
However, the algorithm implements only the general LSTM layer, the attention LSTM layer is not implemented.
Furthermore, the algorithm allows for multiple LSTM layers, instead of only a single LSTM layer.

### Training
The training algorithm has two input data channels: `training` and `validation`. 
The `training` channel is mandatory, while the `validation` channel is optional.

The data should be provided in a CSV file containing the time series and their class labels.
The CSV file should not contain any index column or column headers. 
Each row of the CSV file represents a time series, while each column represents a time step.
The class labels should be stored in the first column, while the time series should be stored in the subsequent columns.
All the time series should have the same length and should not contain missing values.
The time series are scaled internally by the algorithm, there is no need to scale the time series beforehand.

See the sample input files `train.csv` and `valid.csv` in the `data/training/` folder.
See `notebook.ipynb` for an example of how to launch a training job.

#### Distributed Training
The algorithm supports multi-GPU training on a single instance, which is implemented through [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html).
The algorithm does not support multi-node (or distributed) training across multiple instances. 

#### Hyperparameters
The training algorithm takes as input the following hyperparameters:
- `num-layers`: `int`. The number of LSTM layers (default = 1).
- `hidden-size`: `int`. The number of hidden units of each LSTM layer (default = 8).
- `dropout`: `float`. The dropout rate applied after each LSTM layer (default = 0.8).
- `filters-1`: `int`. The number of filters of the first convolutional layer (default = 128).
- `filters-2`: `int`. The number of filters of the second convolutional layer (default = 256).
- `filters-3`: `int`. The number of filters of the third convolutional layer (default = 128).
- `kernel-size-1`: `int`. The size of the kernel of the first convolutional layer (default = 8).
- `kernel-size-2`: `int`. The size of the kernel of the second convolutional layer (default = 5).
- `kernel-size-3`: `int`. The size of the kernel of the third convolutional layer (default = 3).
- `lr`: `float`. The learning rate used for training (default = 0.001).
- `batch-size`: `int`. The batch size used for training (default = 128).
- `epochs`: `int`. The number of training epochs (default = 2000).

#### Metrics
The training algorithm logs the following metrics:
- `train_loss`: `float`. Training loss.
- `train_accuracy`: `float`. Training accuracy.

If the `validation` channel is provided, the training algorithm also logs the following additional metrics:
- `valid_loss`: `float`. Validation loss.
- `valid_accuracy`: `float`. Validation accuracy.

See `notebook.ipynb` for an example of how to launch a hyperparameter tuning job.

### Inference
The inference algorithm takes as input a CSV file containing the time series.
The CSV file should not contain any index column or column headers.
Each row of the CSV file represents a time series, while each column represents a time step.
All the time series should have the same length and should not contain missing values.
The time series are scaled internally by the algorithm, there is no need to scale the time series beforehand.
See the sample input file `test_data.csv` in the `data/inference/input` folder.

The inference algorithm outputs the predicted class labels, which are returned in CSV format.
See the sample output file `batch_predictions.csv` in the `data/inference/output/batch` folder.

See `notebook.ipynb` for an example of how to launch a batch transform job.

#### Endpoints
The algorithm supports only real-time inference endpoints. The inference image is too large to be uploaded to a serverless inference endpoint.

See `notebook.ipynb` for an example of how to deploy the model to an endpoint, invoke the endpoint and process the response.
See the sample output file `real_time_predictions.csv` in the `data/inference/output/real-time` folder.

## References
- F. Karim, S. Majumdar, H. Darabi and S. Chen, "LSTM Fully Convolutional Networks for Time Series Classification," in *IEEE Access*, vol. 6, pp. 1662-1669, 2018, [doi: 10.1109/ACCESS.2017.2779939](https://doi.org/10.1109/ACCESS.2017.2779939).
- F. Karim, S. Majumdar and H. Darabi, "Insights Into LSTM Fully Convolutional Networks for Time Series Classification," in *IEEE Access*, vol. 7, pp. 67718-67725, 2019, [doi: 10.1109/ACCESS.2019.2916828](https://doi.org/10.1109/ACCESS.2019.2916828).

