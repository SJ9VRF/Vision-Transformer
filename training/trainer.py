# Contains training loop and optimization logic.

import torch
from torch import optim
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy

def train(model, train_loader, config):
    # Set the model to training mode
    model.train()
    
    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Setup loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(config['num_epochs']):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

def evaluate(model, val_loader):
    # Set the model to evaluation mode
    model.eval()

    correct = 0
    total = 0
    # Initialize the accuracy metric
    accuracy = Accuracy()
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(config['device']), labels.to(config['device'])
            
            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update accuracy
            accuracy.update(predicted, labels)

    # Calculate final accuracy
    final_accuracy = accuracy.compute()
    print(f'Accuracy of the network on the validation images: {100 * final_accuracy:.2f}%')
    return final_accuracy

# Example configuration
config = {
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
