# Robst equivariant learning framework for CT tasks
## Model Training
To train models, change the value of the following variables to the desired value, then run `Python train.py`. 

- `n_views` takes an integer that is the number of uniform views to take within the angle range defined in the constructor of `CT.py`
- `lb` and `ub` takes an integer between 0 and 180 with $ub >= lb$ that specifies the angle range for projection taking during the measurement phase.
- `epochs` takes an integer that trains the model for this many epochs
- `ckp_interval` takes an integer that determines the 
- `schedule` takes a list of integers determining at which epoch should the learning rate be adjusted
- `batch_size` takes an integer that determines the number of batches to use for training
- `lr` takes a dictionary with keys `G` and `WD` whose values of type float specify the learning rate and weight decay for the optimiser respectively.
- `alpha` takes a dictionary with keys `req`, `sure`, `mc`, `eq` whose values of type float specify the weighting on the corresponding loss function in the framework.
- `noise_model` takes a dictionary that specifies the type of additive noise for the measurement phase. Currently only Gaussian noises are  available.
- `image_width` takes an integer that determines the size of the images used for training. The setting supposes square images therefore `img_height = img_width`.

## Model Evaluation
To evaluate trained models, specify the values for the variables below, then run `Python test.py`. 
- `path` specifies the path containing the csv file with the loss and reconstruction metrics during training.
- `net_ckp_ct` specifies the path to the trained model.