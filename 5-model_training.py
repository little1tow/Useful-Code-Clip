import torch

# adjust the learning rate with the traning epochs
def adjust_learning_rate(learning_rate, learning_rate_decay, optimizer, epoch):
    """Sets the learning rate to the initial LR multiplied by learning_rate_decay(set 0.98, usually) every epoch"""
    learning_rate = learning_rate * (learning_rate_decay ** epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    return learning_rate


# print the gradient to verify whether there are gradients to be returned.
for name, parms in model.named_parameters():
	print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
