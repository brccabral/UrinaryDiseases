# import libraries
import os
import numpy as np
import torch
from six import BytesIO

# import model from model.py, by name
from model import LinearModelDisease

# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'


# Provided model load function
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearModelDisease(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


# Provided input data loading
def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


# Provided predict function
def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model.

    min_t = 35.5
    max_t = 41.5
    
    temp_pred = (input_data[:,0]-min_t)/(max_t-min_t)
    temp_pred = temp_pred.reshape(-1, 1)
    x_input = np.concatenate((temp_pred, input_data[:,1:]), axis=1)
    
    data = torch.from_numpy(x_input.astype('float32'))
    data = data.to(device)

    # Put the model into evaluation mode
    model.eval()
    
    log_ps = model.forward(data)
    log_ps = log_ps.cpu().detach().numpy()
    ps = torch.exp(torch.tensor(log_ps))

    return ps