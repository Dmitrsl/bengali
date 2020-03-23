from catalyst.core.callback import Callback, CallbackOrder
import torch
import torch.nn as nn
import torch.nn.functional as F

class BengaliRecall(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """
    def __init__(
        self,
        prefix: str,
        #metric_fn: Callable,
        input_key: str = "targets",
        output_key: str = "logits",
        average = "macro",
        **metric_params
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix = prefix
        self.metric_fn = recall_score#(y_true_subset, y_pred_subset, average='macro')

        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params
        self.prefix_dict = {'m_grapheme': 0, 'm_vowel': 1, 'm_consonant': 2 }
        self.n_classes = 0
        self.average = average
        self.cm = None
        self.eps = 1e-6


    def on_batch_end(self, state):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]
        
        metrics = []
        for pref in self.prefix_dict:

            preds = outputs[self.prefix_dict[pref]].argmax(-1).cpu()
            targs = targets[:, self.prefix_dict[pref]].cpu()
            #print(preds, targs)
            co_matrix = confusion_matrix(targs, preds)

            FN = (co_matrix.sum(axis=1) - np.diag(co_matrix)).sum()
            TP = (np.diag(co_matrix)).sum()

            recall_ = TP/(TP+FN-self.eps)
            metrics.append(recall_)
            
        metric = 0.5 * metrics[0] + 0.25 * metrics[1] + 0.25 * metrics[2]
        state.batch_metrics[self.prefix] = metric

class Loss_combine(nn.Module):
    def __init__(self, loss, koef):
        super().__init__()
        self.loss = loss
        self.koef = koef
    def forward(self, input, target,reduction='mean'):
        x1,x2,x3 = input
        x1,x2,x3 = x1.float(),x2.float(),x3.float()
        y = target.long()
        return self.koef[0] * self.loss(x1,y[:,0],reduction=reduction) + \
               self.koef[1] * self.loss(x2,y[:,1],reduction=reduction) + \
               self.koef[2] * self.loss(x3,y[:,2],reduction=reduction)