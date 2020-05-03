import matplotlib.pyplot as plt

class Experiment:
    
    REQUIRED_PARAMS = ["net", "loss", "optimizer", "comparison_metric",
                       "train_dataloader", "validation_dataloader"]
    
    def __init__(self, **kwargs):
        """Params
        net (required): PyTorch nn module to be trained and evaluated
        loss (required): loss function used in training
        optimizer (required): optimizer function used in training
        comparison_metric (required): function used for comparing a network output and ground truth,
            needs to accept a network ouput and ground truth and return accuracy
        train_dataloader (required): PyTorch dataloader object used for training
        validation_dataloader (required): PyTorch dataloader object used for validation
        """
        self._validate_required_params(kwargs)
        
        # required params
        self.net                   = kwargs["net"]
        self.loss                  = kwargs["loss"]
        self.optimizer             = kwargs["optimizer"]
        self.comparison_metric     = kwargs["comparison_metric"]
        self.train_dataloader      = kwargs["train_dataloader"]
        self.validation_dataloader = kwargs["validation_dataloader"]
        
        # optional params
        self.cuda                  = kwargs["cuda"] if "cuda" in kwargs else False
        
        
    def run(self, train_epochs, train_validation_interval):
        if train_epochs <= 0:
            raise(f"Error: train_epochs must be greater than zero. {train_epochs} provided is invalid.")
            
        train_loss_over_epochs, val_acc_over_epochs = self._train(train_epochs, train_validation_interval)
        return train_loss_over_epochs, val_acc_over_epochs
        
        
    def plot_result(self, xs, ys, xlabel, ylabel, title):
        plt.plot(xs, ys)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xticks(xs)
        plt.show()
        
        
    def _train(self, num_epochs, validation_interval):    
        train_loss_over_epochs = []
        val_acc_over_epochs = []
        
        for epoch in range(num_epochs):
            
            running_loss = 0.0
            for i, data  in enumerate(self.train_dataloader, 0):
                inputs, gts = data
                
                if self.cuda:
                    inputs = inputs.cuda()
                    gts = gts.cuda()
                    
                self.optimizer.zero_grad()

                outputs = self.net(inputs)

                loss = self.loss(outputs, gts)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
            train_loss_over_epochs.append(running_loss)
                
            if epoch % validation_interval == 0:
                validation_acc = self._validation()
                val_acc_over_epochs.append(validation_acc)
                
        return train_loss_over_epochs, val_acc_over_epochs
        
    
    def _validation(self):
        self.net.eval()
        
        running_avg_acc = 0.0
        for i, data  in enumerate(self.validation_dataloader, 0):
            inputs, gts = data
                
            if self.cuda:
                inputs = inputs.cuda()
                gts = gts.cuda()
                    
            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            running_avg_acc += self.comparison_metric(outputs, gts)
           
        self.net.train()
        return running_avg_acc / len(self.validation_dataloader)
    
        
    def _validate_required_params(self, params):
        for p in self.REQUIRED_PARAMS:
            if p not in params:
                raise(f"Error: required param {p} not found in Experiment args")