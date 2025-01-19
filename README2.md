## How to use watchmal code for DANN
### 0. Code
Code is currently hosted here [https://github.com/takumihoriba/t2k_ml_training/tree/dann_2_datasets](https://github.com/takumihoriba/t2k_ml_training/tree/dann_2_datasets). Please see branch `dann_2_datasets`.

### 1. Configuration
The configuration for DANN training is managed using YAML files. Below is an example configuration file (`t2k_dann_train_classifier.yaml`) that specifies the settings for DANN training.

```yaml
data:
  # specify source dataset (no dead PMTs)
  source:
    split_path: '/data/thoriba/t2k/indices/apr3_eMuPiPlus_1500MeV_small_1/train_val_test_gt200Hits_FCTEST_nFolds10_fold0.npz'
    dataset:
      h5file: '/data/thoriba/t2k/datasets/apr3_eMuPiPlus_1500MeV_small_1/multi_combine.hy'
      _target_: watchmal.dataset.cnn.cnn_dataset.CNNDataset
      pmt_positions_file: '/fast_scratch/WatChMaL/data/T2K/image_files/skdetsim_imagefile.npy'
      channel_scaling:
        time: [400, 1000]
  # specify target dataset (dead PMTs)
  target:
    split_path: '/data/thoriba/t2k/indices/apr3_eMuPiPlus_1500MeV_small_1/train_val_test_gt200Hits_FCTEST_nFolds10_fold0.npz'
    dataset:
      h5file: '/data/thoriba/t2k/datasets/apr3_eMuPiPlus_1500MeV_small_1/multi_combine.hy'
      _target_: watchmal.dataset.cnn.cnn_dataset.CNNDatasetDeadPMT
      pmt_positions_file: '/fast_scratch/WatChMaL/data/T2K/image_files/skdetsim_imagefile.npy'
      channel_scaling:
        time: [400, 1000]
      dead_pmt_rate: 0.05
      dead_pmt_seed: 42  

model:
  _target_: watchmal.model.dann.DANNModel
  # main network
  label_predictor:
    _target_: watchmal.model.dann.LabelPredictor
    # produces hidden representation
    feature_extractor:
      _target_: watchmal.model.resnet.resnet34
      num_input_channels: 2
      num_output_channels: 3 
      stride: 1
      kernelSize: 1
      dropout_p: 0.
    # makes predictions based on representation produced by feature extractor
    label_pred:
      _target_: watchmal.model.dann.FlexibleNNClassifier
      input_dim: 512  # Should match feature_extractor output
      hidden_dims: [64]  
      output_dim: 3    
      dropout_p: 0.2
  domain_classifier:
    _target_: watchmal.model.dann.FlexibleNNClassifier
    input_dim: 512  # Should match feature_extractor output
    hidden_dims: [128, 64, 32]  
    output_dim: 1
    dropout_p: 0.2   

# specify DANN classifier engine
engine:
  _target_: watchmal.engine.classification_dann.DANNClassifierEngine
  truth_key: 'labels'
  label_set: [0,1,2]
  # specify model already trained 
  pretrained_model_path: /data/thoriba/t2k/models/14042024-00062_jun17/ClassifierEngine_ResNet_BEST.pth
  # epochs to train adversary before 2-step min-max training
  domain_pre_train_epochs: 0
  # how many updates on adversary for one parameter update on main network
  domain_in_train_itrs: 1
  # max value of lambda in total loss formula
  max_lammy: 1.5

tasks:
  train:
    epochs: 50
    val_interval: 25
    num_val_batches: 16
    checkpointing: false
    loss:
      # main loss function
      classification:
        _target_: torch.nn.CrossEntropyLoss
      # domain loss function
      domain:
        _target_: torch.nn.BCEWithLogitsLoss
    data_loaders:
      # data loaders should be specified as <type of dataset>_<train or validation>
      source_train:
        split_key: train_idxs
        batch_size: 16
        num_workers: 1
        sampler:
          _target_: torch.utils.data.sampler.SubsetRandomSampler
      target_train:
        split_key: train_idxs
        batch_size: 16
        num_workers: 1
        sampler:
          _target_: torch.utils.data.sampler.SubsetRandomSampler
      source_validation:
        split_key: val_idxs
        batch_size: 8
        num_workers: 1
        sampler:
          _target_: torch.utils.data.sampler.SubsetRandomSampler
      target_validation:
        split_key: val_idxs
        batch_size: 8
        num_workers: 1
        sampler:
          _target_: torch.utils.data.sampler.SubsetRandomSampler
    # separate optimizers can be specified here
    optimizers:
      label_predictor:
        _target_: torch.optim.Adam
        lr: 0.01
        weight_decay: 0
      domain_classifier:
        _target_: torch.optim.Adam
        lr: 0.005
        weight_decay: 0
```

### 2. Running the Training
To run the DANN training, use the main.py script with the appropriate configuration file. 

### 3. Explanation of Configuration
- `data`: Specifies the source and target datasets, including paths to the data files and any preprocessing steps.
- `model`: Defines the DANN model, including the label predictor and domain classifier.
- `engine`: Specifies the engine to be used for training, in this case, `DANNClassifierEngine`.
- `tasks`: Defines the training task, including the number of epochs, validation intervals, loss functions, data loaders, optimizers, and scheduler.
### 4. Key Components
- Label Predictor: The part of the model responsible for predicting the labels of the input data.
- Domain Classifier: The part of the model responsible for distinguishing between the source and target domains.
- Gradient Reversal Layer (GRL): A layer that reverses the gradients during backpropagation to encourage the feature extractor to learn domain-invariant features.
### 5. Customizing the Training
You can customize various aspects of the training by modifying the YAML configuration file. For example:
- Change the learning rates or optimizers for the label predictor and domain classifier.
- Adjust the batch sizes and number of workers for the data loaders.
- Modify the architecture of the label predictor and domain classifier.


## Development Notes
In this notes, I try to outline how the code evolved from base model and engine. 
### Engine
🆕 denotes new methods or ideas introduced for domain adaptation and ✏️ denotes methods that were modified.
1. `DANNEngine` is based on `ReconstructionEngine`. The class is stored in `domain_adaptation.py`. Here are some changes.
    - 🆕 It takes additional parameters such as `pretrained_model_path=None, domain_pre_train_epochs=2, domain_in_train_itrs=2, max_lammy=1.0`. 
    - ✏️ Method `configure_data_loaders` is modified such that it can deal with two datasets.
    - 🆕 `load_pretrained_model` to use pre-trained weights
    - ⚠️ `forward` method is implemented but ` DANNClassifierEngine`'s `forward` method is used.
    - ✏️ `train` is the core method of this class. It has several functionalities related to training of DANN
        - Loads pre-trained model by calling the helper method
        - Trains adversary only before anything if `domain_pre_train_epochs > 0`
        - Trains adversary and main network in turn. In a loop of training iteration, it makes calls to `train_adversary`, a helper method to train adversary network.
    - 🆕 `set_requires_grads_for_models(self, is_train_f: bool, is_train_r: bool)`: used for setting need of gradients for main network `f` and adversary `r`.
    - 🆕 `train_adversary(...)` to train specified iterations of adversary network.
    - ✏️ `validate` is modified to deal with multiple datasets.

2. `DANNClassifierEngine` is in `classification_dann.py`, and it is an engine specific to classification task. This extends original `DANNEngine` described above.
    - ✏️ `configure_data_loaders` adds configuration of labels not covered by parent class
    - ✏️ `forward` computes class accuracy and domain accuracy in addition to losses.

### Model
The model is stored in `dann.py`. Model components created for domain adaptation are described below.

1. `GradientReversalLayer`
    - A custom autograd function that implements the gradient reversal layer.
    - Methods:
        - `forward(ctx, x, alpha)`: Stores the alpha parameter and returns the input tensor as-is.
        - `backward(ctx, grad_output)`: Reverses the gradients by multiplying with -alpha.

1. `GradientReversalLayerModule`
    - A PyTorch module that wraps the GradientReversalLayer autograd function.
    - Methods:
        - `__init__(self, lambda_grad=1.0)`: Initializes the module with a gradient reversal factor `lambda_grad`.
        - `forward(self, x)`: Applies the gradient reversal layer to the input tensor x.
1. `DANNModel`
    - The main model class for the Domain-Adversarial Neural Network.
    - Attributes:
        - `label_predictor`: A module that predicts the labels for the primary task.
        - `domain_classifier`: A module that classifies the domain (source or target).
        - `grl`: An instance of the GradientReversalLayerModule.
    - Methods:
        - `__init__(self, label_predictor, domain_classifier)`: Initializes the model with a label predictor and a domain classifier.
        - `forward(self, x, alpha=0, apply_grl=True)`: Performs the forward pass. Applies the GRL to the features if `apply_grl` is `True`, and returns the label and domain outputs.

1. `LabelPredictor`
    - A module that predicts the labels for the primary task.
    - Attributes:
        - `feature_extractor`: A module that extracts features from the input data.
        - `label_predictor`: A module that predicts the labels based on the extracted features.
    - Methods:
        - `__init__(self, feature_extractor, label_pred)`: Initializes the module with a feature extractor and a label predictor.
        - `forward(self, x)`: Extracts features from the input data and predicts the labels.
