import torch
import torch.nn as nn
from torch_scatter import scatter


class cluster_loss_helper(nn.Module):
    def __init__(self):
        super(cluster_loss_helper, self).__init__()

    def forward(self, prediction, correct_label, delta_v, delta_d):
        """

        :param prediction: [N, 4, 256, 512]
        :param correct_label: [N, 256, 512]
        :param delta_v:
        :param delta_d:
        :return:
        """
        prediction_reshape = prediction.view(prediction.shape[0], prediction.shape[1],
                                             prediction.shape[2] * prediction.shape[3])  # [N, 4, 131072]
        correct_label_reshape = correct_label.view(correct_label.shape[0], 1,
                                                   correct_label.shape[1] * correct_label.shape[
                                                       2])  # [N, 1, 131072]

        output, inverse_indices, counts = torch.unique(correct_label_reshape, return_inverse=True,
                                                       return_counts=True)
        counts = counts.float()
        num_instances = len(output)

        # mu_sum = scatter(prediction_reshape, inverse_indices, dim=2, reduce="sum") # [N, 4, 5]
        # muc = mu_sum/counts # [N, 4, 5]
        muc = scatter(prediction_reshape, inverse_indices, dim=2, reduce="mean")  # [N, 4, 5]

        dis = torch.index_select(muc, 2, inverse_indices.view(inverse_indices.shape[-1]),
                                 out=None)  # [N, 4, 131072]
        dis = dis - prediction_reshape
        dis = torch.norm(dis, dim=1, keepdim=False, out=None, dtype=None)  # [N, 131072]
        dis = dis - delta_v
        dis = torch.clamp(dis, min=0.)  # [N, 131072]
        dis = torch.pow(dis, 2, out=None)

        L_var = scatter(dis, inverse_indices.view(inverse_indices.shape[-1]), dim=1, reduce="mean")  # [N, 3]
        L_var = torch.sum(L_var) / num_instances

        L_dist = torch.tensor(0, dtype=torch.float)
        for A in range(num_instances):
            for B in range(num_instances):
                if A != B:
                    dis = muc[:, :, A] - muc[:, :, B]
                    dis = torch.norm(dis, dim=1, keepdim=False, out=None, dtype=None)
                    dis = delta_d - dis
                    dis = torch.clamp(dis, min=0.)
                    dis = torch.pow(dis, 2, out=None)
                    L_dist = L_dist + dis
        L_dist = L_dist / (num_instances * (num_instances - 1))
        L_dist = L_dist.view([])
        total_loss = L_var + L_dist
        return total_loss


class cluster_loss(nn.Module):
    def __init__(self):
        super(cluster_loss, self).__init__()

    def forward(self, binary_logits, binary_labels,
                instance_logits, instance_labels, delta_v=0.5, delta_d=3):
        # Binary Loss
        # Since the two classes (lane/background) are highly unbalanced, we apply bounded inverse class weighting
        output, counts = torch.unique(binary_labels, return_inverse=False, return_counts=True)
        counts = counts.float()
        inverse_weights = torch.div(1.0, torch.log(
            torch.add(torch.div(counts, torch.sum(counts)), torch.tensor(1.02, dtype=torch.float))))

        binary_loss = torch.nn.CrossEntropyLoss(weight=inverse_weights)
        binary_segmenatation_loss = binary_loss(binary_logits, binary_labels)

        batch_size = instance_logits.shape[0]
        loss_set = []
        for dimen in range(batch_size):
            loss_set.append(cluster_loss_helper())

        instance_segmenatation_loss = torch.tensor(0.)#.cuda()

        for dimen in range(batch_size):
            instance_loss = loss_set[dimen]
            # prediction = instance_logits[dimen].view(1, instance_logits.shape[1], instance_logits.shape[2],
            #                                          instance_logits.shape[3])
            # correct_label = instance_labels[dimen].view(1, instance_labels.shape[1], instance_labels.shape[2])
            # instance_segmenatation_loss += instance_loss(prediction, correct_label, delta_v, delta_d)
            prediction = torch.unsqueeze(instance_logits[dimen], 0) # .cuda()
            correct_label = torch.unsqueeze(instance_labels[dimen], 0)# .cuda()
            instance_segmenatation_loss += instance_loss(prediction, correct_label, delta_v, delta_d)

        instance_segmenatation_loss = instance_segmenatation_loss / batch_size
        return binary_segmenatation_loss, instance_segmenatation_loss