import torch

def train_epoch(model, device, dataloader, loss_fn, optimizer, scheduler):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    model.train()    

    for data in dataloader:
        
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
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()*inputs.size(0)

        # Get the predictions
        score, prediction = torch.max(outputs.data, 1)

        # Count of predcitions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
        
        # wandb.log({'Training Accuracy': (correct_prediction/total_prediction), 'Training Loss': loss})

        
    return running_loss, correct_prediction

def valid_epoch(model, device, dataloader, loss_fn):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    model.eval()
    
    for data in dataloader:
        
        # Get inputs and labels on device
        inputs = data[0].to(device)
        labels = data[1].to(device)

        # Normalize Inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        
        # Run Inputs Through Model
        outputs = model(inputs)
        loss=loss_fn(outputs,labels)
        running_loss += loss.item()*inputs.size(0)
        
        # Get the predictions
        score, prediction = torch.max(outputs.data, 1)
        
        # Count of predcitions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]
        
        # wandb.log({'Validation Accuracy': (correct_prediction/total_prediction), 'Validation Loss': loss})
        
    return running_loss, correct_prediction