import torch
import random
import torch.nn as nn

class DisLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y1, y2):
        # add noise trick, 5% precent to exchange the label of sample
        if random.randint(1,100) <= 5:
            y1, y2 = y2, y1
        # soft label trick
        soft_label1 = random.uniform(0.9, 1)
        soft_label0 = random.uniform(0, 0.1)
        return torch.pow((y1 - soft_label1), 2).sum() + torch.pow((y2 - soft_label0), 2).sum()
    
class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y2):
        return torch.pow(y2, 2).sum()


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
            
def symmetric_cross_entropy(alpha, beta):
    """
    Symmetric Cross Entropy: 
    ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    https://arxiv.org/abs/1908.06112
    """
    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return alpha*tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis = -1)) + beta*tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis = -1))
    return loss

class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross Entropy: 
    ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels" 
    https://arxiv.org/abs/1908.06112
    """
    def __init__(self, alpha=0.1, beta=1):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
