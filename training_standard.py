# Create Training Loop
import time
import torch
import torch.nn as nn
import tqdm
from tqdm.auto import trange

def training(model, train_dl, device, num_epochs, lr, max_lr, logger=None):
    model.train()
    start_time = time.time()
    num_batches = len(train_dl)
    summary = []
    
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=int(len(train_dl)), epochs=num_epochs, anneal_strategy='linear')

    # Repeat for each epoch
    with trange(num_epochs, desc="Train Epoch", unit="epoch") as tepoch:
        for epoch in tepoch:
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
                avg_loss = running_loss / num_batches
                acc = correct_prediction / total_prediction

                tepoch.set_postfix(loss=avg_loss, acc=acc, batch=f"{i}/{len(train_dl)}")
                if logger is not None:
                    logger.log({'Train Loss': avg_loss, 'Train Accuracy': acc, 'Epoch': epoch, 'Correct Predictions': correct_prediction})

            # Print stats
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            epoch_time = time.time() - epoch_start
            summary.append(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}, Time: {epoch_time:.2f}s')

    end_time = time.time()
    print('Finished Training\n')
    for l in summary:
        print(l)
    print("\n")
    print(f'Total Training Time: {end_time - start_time:.2f}s')

# Create Inference Loop

def inference (model, val_dl, device, logger=None):
    print(f"Running Validation...")
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
