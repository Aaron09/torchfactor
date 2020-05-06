import matplotlib.pyplot as plt
import torch

class Experiment:
    """General experiment wrapper.
    
    For running an experiment with the usual setup of 
    data as input to the network, with ground truths (labels), and calculating accuracies,
    the constructor would be something like:
    
    experiment = Experiment(net=MY_NETWORK, loss=MY_LOSS, optimizer=MY_OPTIMIZER,
        train_dataloader=MY_TRAIN_DATALOADER, validation_dataloader=MY_VALIDATION_DATALOADER,
        cuda=(True/False), accuracy_func=MY_ACCURACY_FUNC
    )
    
    train_losses, val_losses, val_accs = experiment.run(
        train_epochs=NUM_EPOCHS, train_validation_interval=VAL_INTERVAL
    )
    """
    
    REQUIRED_PARAMS = ["net", "loss", "optimizer",
                       "train_dataloader", "validation_dataloader"]
    
    def __init__(self, **kwargs):
        """Params
            net (required): PyTorch nn module to be trained and evaluated
            loss (required): loss function used in training
            optimizer (required): optimizer function used in training
            train_dataloader (required): PyTorch dataloader object used for training
            validation_dataloader (required): PyTorch dataloader object used for validation
        
            cuda (optional): use cuda, defaults to False
            use_eye_as_net_input (optional): use identity matrix as network input, defaults to False
            inputs_are_ground_truth (optional): use the input data as the ground truth
                in loss and accuracy functions
            accuracy_func (optional): functional for computing output accuracy as compared to ground truth
        """
        self._validate_required_params(kwargs)
        
        # required params
        self.net                     = kwargs["net"]
        self.loss_func               = kwargs["loss"]
        self.optimizer               = kwargs["optimizer"]
        self.train_dataloader        = kwargs["train_dataloader"]
        self.validation_dataloader   = kwargs["validation_dataloader"]
        
        # optional params
        self.cuda                    = kwargs["cuda"] if "cuda" in kwargs else False
        self.use_eye_as_net_input    = kwargs["use_eye_as_net_input"] if "use_eye_as_net_input" in kwargs else False
        self.inputs_are_ground_truth = kwargs["inputs_are_ground_truth"] if "inputs_are_ground_truth" in kwargs else False
        self.accuracy_func           = kwargs["accuracy_func"] if "accuracy_func" in kwargs else None
        self.has_labels              = kwargs["has_labels"] if "has_labels" in kwargs else True
        self.print_all_epochs        = kwargs["print_all_epochs"] if "print_all_epochs" in kwargs else False
        
        
    def run(self, train_epochs, train_validation_interval):
        """Params
            train_epochs: number of epochs to train for
            train_validation_interval: the epoch frequency which to calculate validation statistics
        
        Returns
            train_loss_over_epochs (list of floats): train loss for every epoch
            val_loss_over_epochs (list of floats): validation loss for each epoch at interval
            val_acc_over_epochs (None or list of floats): None if no accuracy function provided,
                accuracies otherwise
        """
        if train_epochs <= 0:
            raise Exception(f"Error: train_epochs must be greater than zero. {train_epochs} provided is invalid.")
            
        train_loss_over_epochs, val_loss_over_epochs, val_acc_over_epochs = self._train(
            train_epochs, train_validation_interval
        )
        
        return train_loss_over_epochs, val_loss_over_epochs, val_acc_over_epochs
        
        
    def plot_result(self, xs, ys, xlabel, ylabel, title):
        plt.plot(xs, ys)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(xs)
        plt.show()
        
        
    def _train(self, num_epochs, validation_interval):    
        train_loss_over_epochs = []
        val_loss_over_epochs = []
        val_acc_over_epochs = [] if self.accuracy_func else None
        
        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
  
            for i, data  in enumerate(self.train_dataloader, 0):  
                self.optimizer.zero_grad()
            
                loss, _ = self._run_network(data, calc_accuracy=False)
                    
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
            train_loss_over_epochs.append(running_loss)
            if self.print_all_epochs or epoch % validation_interval == 0:
                print(f"epoch {epoch}: total loss is {running_loss}, avg loss is {running_loss / len(self.train_dataloader)}")
                
            if epoch % validation_interval == 0:
                validation_loss, validation_acc = self._validation()
                val_loss_over_epochs.append(validation_loss)
                
                outstr = f"epoch {epoch}: val avg loss is {validation_loss}"
                if self.accuracy_func:
                    val_acc_over_epochs.append(validation_acc)
                    outstr += f". val avg acc is {validation_acc}"
                    
                print(outstr)
                
                
        return train_loss_over_epochs, val_loss_over_epochs, val_acc_over_epochs
        
    
    def _validation(self):
        self.net.eval()
        
        running_loss = 0.0
        running_acc = 0.0 if self.accuracy_func else None
        
        for i, data  in enumerate(self.validation_dataloader, 0):
            loss, acc = self._run_network(data, calc_accuracy=True)
            running_loss += loss.item()
            
            if acc:
                running_acc += acc
                   
        avg_loss = running_loss / len(self.validation_dataloader)
        
        if running_acc:
            avg_acc = running_acc / len(self.validation_dataloader)
        else:
            avg_acc = None
            
        self.net.train()
        return avg_loss, avg_acc
    
    
    def _run_network(self, data, calc_accuracy=False):
        if self.has_labels:
            inputs, gts = data
        else:
            inputs = data
                                    
        if self.cuda:
            inputs = inputs.cuda()
                    
        if self.use_eye_as_net_input:
            # dim 0 is batch size, dim 1 is number of channels (e.g., 3 for color image)
            outputs = self.net(torch.eye(inputs.shape[2]))
        else:
            outputs = self.net(inputs)

        if self.inputs_are_ground_truth:
            loss = self.loss_func(outputs, inputs)
        else:
            loss = self.loss_func(outputs, gts)
        
        if calc_accuracy and self.accuracy_func is not None:
            if self.inputs_are_ground_truth:
                acc = self.accuracy_func(outputs, inputs)
            else:
                acc = self.accuracy_func(outputs, gts)
        else:
            acc = None
        
        return loss, acc
            
        
    def _validate_required_params(self, params):
        for p in self.REQUIRED_PARAMS:
            if p not in params:
                raise Exception(f"Error: required param {p} not found in Experiment args")