import torch
import config # Import config settings
import time

def train_model(model, train_loader, criterion, optimizer):
    #Trains the model just for one epoch
    device = config.DEVICE
    #setting the model to training mode
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    batch_count = 0
    #again to keep track of the time spent
    epoch_start_time = time.time()
    batch_times = []



    for i, (inputs, labels) in enumerate(train_loader):
        batch_start_time = time.time()
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) # Use non_blocking for potential overlap

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        batch_count += 1
        batch_times.append(time.time() - batch_start_time)



    epoch_duration = time.time() - epoch_start_time
    #for possibility of dataloader being empty
    num_samples = total_predictions if total_predictions > 0 else 1
    epoch_loss = running_loss / num_samples
    epoch_acc = correct_predictions / num_samples

    print(f"  Epoch completed in {epoch_duration:.2f}s. Avg Batch Time: {sum(batch_times)/len(batch_times):.3f}s" if batch_times else "")

    return epoch_loss, epoch_acc