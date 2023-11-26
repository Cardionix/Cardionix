# Cardio Sonix Training Pipeline ⚡️️
**A regular machine learning pipeline including an iterative 
ETL-pipeline for processing and other data transformation, 
datasets for classification and segmentation of heartbeats, 
a data module for management for partitioning and managing data, 
as well as models of the Cardio Sonix line. 
The pipeline is covered with documentation and logging for any experiment. 
Launch configurations are placed in a separate configs package 
and contain data models for individual modules. 
This makes it easy to log and validate multiple input arguments**
****

## Requirements
```
# Run this command from the project directory after cloning the source code
pip install -r requirements.txt
```

## Start-up
1. **Define** `callbacks.py`. **You can edit and add your own callbacks (which will be used for early stopping etc.) to your callbacks.py**
```
# You can see callbacks by default
callbacks = [
    ModelCheckpoint(
        dirpath="./checkpoints",
        filename="epoch={epoch}"
                 "-precision={val/macro avg/precision:.2f}"
                 "-recall={val/macro avg/recall:.2f}"
                 "-f1-score={val/macro avg/f1-score:.2f}",
        monitor="val/macro avg/f1-score",
        mode="max",
        save_top_k=10,
        auto_insert_metric_name=False,
        save_weights_only=True
    ),

    EarlyStopping(
        monitor="val/macro avg/f1-score",
        mode="max",
        patience=8,
        min_delta=1e-6
    )
]
```

2. **Define** `config.py`.
**This file should contain configurations for the Cardio Sonix Pipeline.
You also can edite this configs as you want and
import them into main.py to start training or
define configurations currently into main.py**
```
dataset_params = ClassifyDatasetParams(
    extra_filepath="Your/Filepath/To/Extra/Dataset",  # file .csv (label encoded CDC survey 2020)
    audio_dirpath="Your/Dirpath/To/Audio/Of/Heartbeat/Sounds",  # dirpath (DHD/audio dir)
    labels_filepath="Your/Filepath/To/File/With/Classes/Labels",  # file .csv (DHD/labels.csv file)
    split_ratio=[0.80, 0.20],  # split ratio of the dataset: 80% - train and 20% - validation
    # Which classes is needed for merging into one class
    merge_classes={
        "artifact": ["artifact"],
        "healthy": ["normal"],
        "abnormal": ["murmur", "extrahls", "extrastole"]
    }
)

etl_pipeline_params = ETLPipelineParams(
    scaler="Normalizer",  # scaler for normalization or scaling numerical features in tabular data
    encoder="OneHotEncoder",  # encoder for encoding categorical features in tabular data
    sample_rate=16000,  # sample rate for the resampling all audio data
    duration=15,  # duration to pad or clip all audio data
    mono=True,  # one or two audio channels load
    extractor="MFCC",  # extractor name for extracting features from audio data
    # extractor parameters for extracting features from audio data
    extractor_kwargs={
        "n_fft": 2048,
        "win_length": 2048,
        "hop_length": 1024,
        "n_mels": 52,  # number of mel filters
        "n_mfcc": 52,  # number of mfcc`s
        "average_by": None  # average time or frequency axis (non-average if None)
    }
)

datamodule_params = DataModuleParams(batch_size=20, num_workers=12)  # Data uploading parameters

lightmodule_params = LightningModuleParams(
    optimizer=optim.Adam,  # Optimizer to use for training model
    # optimizer parameters
    optimizer_kwargs={
        "lr": 1e-4
    },
    lr_scheduler=lr_scheduler.ReduceLROnPlateau,  # learning rate Scheduler
    # and his parameters
    lr_scheduler_kwargs={
        "patience": 3
    },
    lr_scheduler_dict_kwargs={
        "monitor": "val/loss",
        "interval": "epoch"
    },
    criterion=nn.CrossEntropyLoss,  # Loss function to use
    # and its parameters
    criterion_kwargs={
        "weight": torch.tensor([1.0, 1.0, 1.0])
    }
)
```

3. **Run** `main.py` 
```
# Run this file for start training
python main.py
```

## Models
### <i>BaselineRNN</i>
**This neural network is the first in the Cardio Sonix line of models.** 
**The architecture consists:**
   - Encoder 
      - (0): Conv1d
      - (1): ReLU
      - (2): MaxPool1d
      - (3): BatchNorm1d
   - LSTMModule
     - LSTM
   - Decoder
     - Linear
     - ReLU
     - Linear
     - Softmax

**The encoder, during the training process, 
learns to extract features with a time domain, 
which then analyzes the LSTM, 
and the decoder expands the network output in logits or probabilities.**

![](https://i.ibb.co/Zh7Fsf2/attached.png)

### <i>CardioNetV2</i>
**The latest multi-modal model in the Cardio Sonix line. 
Built on the basis of models whose architectures were originally 
intended for computer vision tasks 
(like a modified ResNet) or for NLP (like LSTM). 
The model works with audio signal and tabular data. 
The model works with the input audio signal as with tokens: 
a mel-kesprogram with time samples is extracted from the audio, 
where each time sample has N-mel-cepstral coefficients. 
At the very beginning, the LSTM takes a mel-cepstrogram as input 
and produces an output tensor that goes into ResNet (Residual Neural Network). 
ResNet is a modified audio signal processing model from the family of residual networks. 
In this implementation, residual blocks with pre-activation were used.
The data then goes to the DenseMixer input. 
This model performs inference separately for audio and tabular features, 
then concatenates the outputs into a dense feature vector and performs inference on it, 
after which we get a prediction based on audio and tabular data**

![](https://i.ibb.co/gW14Dh2/attached.png)
****

## Classes diagram
**In the image below you can see the class diagram of the entire pipeline.**

![](https://i.ibb.co/pxc2hgf/k.jpg)


## Classes description
**And here is a brief description of all the classes involved in the pipeline architecture. 
Classes are grouped depending on the global tasks they perform.**

1. Processing, data augmentation:
   - AudioAugment – augmentation of audio data.
   - MFCCExtractor – extraction of mel-cepstral coefficients.
   - AudioPreprocessor – preprocessing of audio data, including standardization of sampling frequency and duration, as well as loading an audio file
   - TabularPreprocessor – processing of tabular data
   - ETLPiepeline – organization of a pipeline for sequential processing, augmentation and feature extraction from tabular and audio data.
   
2. Construction, partitioning and iteration of data:
   - HealthChecker – checks the correctness of database partitioning.
   - DatasetParthioner – partitioning and portioned data output.
   - Builder – loading, building and providing data through property properties.
   - CardioAnomalyDataset – iterative output of features and predictor.

3. Organization of stage cycles and data loaders:
   - DatasetParthioner – manipulation of sections and data loaders.
   - CardioLightningModule – organization of training and test cycles.
   - CardioTrainer – logging configurations, composition of all parts of the pipeline, launching training and testing
   
4. Storing model residuals, calculating metrics:
   - MetricsStorage – storage and averaging of external step-by-step calculated metrics.
   - ProbaStorage – storing step-by-step model outputs in the form of probabilities, calculating default classification metrics (roc-auc, precision, recall, etc.)
