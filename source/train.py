import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn

from model import LinearModelDisease

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
    model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model


def _get_train_data_loader(training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0,1,2,3]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0,1,2,3], axis=1).values).float()

    return train_y, train_x


def train(model, train_y, train_x, epochs, loss_fn, optimizer, device, output_dim):
    print("Training.")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in zip(train_x, train_y):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            log_ps = model.forward(x)
            ps = torch.exp(log_ps)
            loss = loss_fn(ps, torch.tensor([y.argmax().item()]).to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch}',
        f' Train Loss: {total_loss/train_y.shape[0]:.2f}')
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # training
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # model
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--input_features', type=int, default=4, metavar='IN',
                        help='input dimension for training (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=10, metavar='H',
                        help='hidden dimension for training (default: 10)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='output dimension for training (default: 1)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    train_y, train_x = _get_train_data_loader(args.data_dir)
    
    model = LinearModelDisease(args.input_features, args.hidden_dim, args.output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    train(model, train_y, train_x, args.epochs, loss_fn, optimizer, device, args.output_dim)
    
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)
    
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)