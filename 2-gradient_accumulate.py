# using accumulation for larger batch size in limited device
# some codes are refering to https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3

import model, evaluate_model, loss_function
import training_set
import torch

# solution 1
model.zero_grad()                                   # Reset gradients tensors
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     # Forward pass
    loss = loss_function(predictions, labels)       # Compute loss function
    loss = loss / accumulation_steps                # Normalize our loss (if averaged)
    loss.backward()                                 # Backward pass
    if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step
        model.zero_grad()                           # Reset gradients tensors
        if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
            evaluate_model()                        # ...have no gradients accumulated


## Solution 2

total_loss=0
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     # Forward pass
    loss = loss_function(predictions, labels)       # Compute loss function

    if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
        total_loss = (total_loss+loss )/ accumulation_steps                # Normalize our loss (if averaged)
        total_loss.backward()                                 # Backward pass
        optimizer.step()                            # Now we can do an optimizer step
        model.zero_grad()                           # Reset gradients tensors
        total_loss=0
        if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
            evaluate_model()
   else :
        total_loss=loss+total_loss


## Our solution: more implementation for in-batch negative sampling

iter_dataset = iter(training_set)
training_batch_number = int(len(training_set) // (batch_size * accumulation_steps))

for batch_idx in range(training_batch_number):
    total_loss=0
    all_prediction = []
    all_labels = []
    model.zero_grad()

    for batch_idx in range(accumulation_steps):
        inputs, labels = next(iter_dataset)
        all_labels.append(labels)

        predictions = model(inputs)
        all_prediction.append(predictions)

    all_labels = torch.cat(all_labels, dim=0)
    all_prediction = torch.cat(all_prediction, dim=0)  # concat all the result together for in-batch sampling

    total_loss = loss_function(all_prediction, all_labels)
    total_loss.backward()
    optimizer.step()


## solution 4: use amp

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

model = model().cuda()
optimizer = optim.SGD(model.parameters(),...)

enable_amp = True if 'cuda' in device.type else False

scaler = GradScaler(enabled=enable_amp)

total_loss=0
for i, (inputs, labels) in enumerate(training_set):
    model.zero_grad()

    with autocast(enabled=enable_amp):                  # only forward pass in autocast
        predictions = model(inputs)                     # Forward pass
        loss = loss_function(predictions, labels)       # Compute loss function

    # 1. scales the loss, in case the gradient is disappear
    scaler.scale(loss).backward()

    # 2. if gradient is not infs or nans, then using optimizer.step() to update
    #  else ignore the step, in case the weights are not damaged
    scaler.step(optimizer)

    # 3. whether or not to increase the scaler
    scaler.update()
