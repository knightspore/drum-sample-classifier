# Create Training Loop
import time
import torch
import torch.nn as nn

def training(model, train_dl, device, num_epochs, logger=None):
    model.train()
    start_time = time.time()
    num_batches = len(train_dl)
    
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)), epochs=num_epochs, anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get inputs and labels on device
            inputs = data[0].to(device) 
            labels = data[1].to(device)

            # Normalize Inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predictions
            score, prediction = torch.max(outputs.data, 1)

            # Count of predcitions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            if logger is not None:
                avg_loss = running_loss / num_batches
                acc = correct_prediction / total_prediction
                logger.log({'Train Loss': avg_loss, 'Train Accuracy': acc, 'Epoch': epoch, 'Correct Predictions': correct_prediction})

        # Print stats
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        epoch_time = time.time() - epoch_start
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, Time: {epoch_time:.2f}s')

    end_time = time.time()
    print('Finished Training')
    print(f'Total Training Time: {end_time - start_time:.2f}s')

# Create Inference Loop

def inference (model, val_dl, device, logger=None):
    correct_prediction = 0
    total_prediction = 0
    start_time = time.time()
    model.eval()

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:

            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction/total_prediction
    end_time = time.time()

    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    print(f'Total Inference Time: {end_time - start_time:.4f}s')
    if logger is not None:
            logger.log({'Inference Accuracy': acc, 'Correct Inf. Predictions': correct_prediction})