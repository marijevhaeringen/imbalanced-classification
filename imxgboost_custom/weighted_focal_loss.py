import numpy as np


class Weighted_Focal_Binary_Loss:
    '''The class of weighted focal loss, allows the users to change the alpha and gamma parameter'''

    def __init__(self, imbalance_alpha, gamma_indct):
        '''
        :param imbalance_alpha: The parameter to specify the alpha weight
        :param gamma_indct: The parameter to specify the gamma indicator
        '''
        self.gamma_indct = gamma_indct
        self.imbalance_alpha = imbalance_alpha

    def robust_pow(self, num_base, num_pow):
        # allows power of negative number

        return np.sign(num_base) * (np.abs(num_base)) ** num_pow

    def weighted_focal_binary_object(self, pred, dtrain):
        gamma_indct = self.gamma_indct
        imbalance_alpha = self.imbalance_alpha
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient parts
        a = (imbalance_alpha ** label)
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred
        # combine the gradient
        grad = a * (gamma_indct * g3 * self.robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + ((-1) ** label) * self.robust_pow(g5, (gamma_indct + 1)))
        # combine the gradient parts to get hessian components
        hess_1 = self.robust_pow(g2, gamma_indct) + gamma_indct * ((-1) ** label) * g3 * self.robust_pow(g2, (gamma_indct - 1))
        hess_2 = ((-1) ** label) * g3 * self.robust_pow(g2, gamma_indct) / g4
        # get the final 2nd order derivative
        hess = a * (((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct + (gamma_indct + 1) * self.robust_pow(g5, gamma_indct)) * g1)

        return grad, hess
