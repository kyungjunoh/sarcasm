from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class BaseTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.loss_fn = CrossEntropyLoss()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def compute_loss(self, model, inputs, num_items_in_batch = None, return_outputs=False):
        image = inputs['frames'].to(self.device)
        label = inputs['labels'].to(self.device)
        # Forward pass
        output = model(image)
        logits = output["logits"] if isinstance(output, dict) else output
        # Compute loss
        loss = self.loss_fn(logits, label)

        if return_outputs:
            return loss, output, label
        return loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override the prediction_step to handle the custom loss function.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs, labels = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        return (loss, logits.detach(), labels.detach())