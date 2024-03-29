import torch
import torch.nn as nn
import random

# code from https://github.com/alfonmedela/triplet-loss-pytorch/blob/master/loss_functions/triplet_loss.py
# original code from https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py
def pairwise_distance_torch(embeddings, device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)

    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)

    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    c12 = c1 + c2
    pairwise_distances_squared = c12 - 2.0 * c3

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.tensor([0.]).to(device))
    # Get the mask where the zero distances are at.
    error_mask = pairwise_distances_squared.clone()
    error_mask[error_mask > 0.0] = 1.
    error_mask[error_mask <= 0.0] = 0.

    pairwise_distances = torch.mul(pairwise_distances_squared, error_mask)

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    return pairwise_distances

def TripletSemiHardLoss(y_true, y_pred, device, margin=0.2, sampling_num = None):
    """Computes the triplet loss_functions with semi-hard negative mining.
       The loss_functions encourages the positive distances (between a pair of embeddings
       with the same labels) to be smaller than the minimum negative distance
       among which are at least greater than the positive distance plus the
       margin constant (called semi-hard negative) in the mini-batch.
       If no such negative exists, uses the largest negative distance instead.
       See: https://arxiv.org/abs/1503.03832.
       We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
       [batch_size] of multi-class integer labels. And embeddings `y_pred` must be
       2-D float `Tensor` of l2 normalized embedding vectors.
       Args:
         margin: Float, margin term in the loss_functions definition. Default value is 1.0.
         name: Optional name for the op.
       """

    labels, embeddings = y_true, y_pred
    # Reshape label tensor to [batch_size, 1].
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])

    pdist_matrix = pairwise_distance_torch(embeddings, device)

    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))
    # print(adjacency.shape)
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]

    # Compute the mask.
    pdist_matrix_tile = pdist_matrix.repeat(batch_size, 1)
    adjacency_not_tile = adjacency_not.repeat(batch_size, 1)
    # print(pdist_matrix.shape)
    transpose_reshape = pdist_matrix.transpose(0, 1).reshape(-1, 1)
    # print(transpose_reshape.shape)
    greater = pdist_matrix_tile > transpose_reshape
    # print(greater.shape)
    mask = adjacency_not_tile & greater
    # print(mask.shape)
    # final mask
    mask_step = mask.to(dtype=torch.float32)
    mask_step = mask_step.sum(axis=1)
    mask_step = mask_step > 0.0
    mask_final = mask_step.reshape(batch_size, batch_size)
    mask_final = mask_final.transpose(0, 1)

    adjacency_not = adjacency_not.to(dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    axis_maximums = torch.max(pdist_matrix_tile, dim=1, keepdim=True)
    masked_minimums = torch.min(torch.mul(pdist_matrix_tile - axis_maximums[0], mask), dim=1, keepdim=True)[0] + axis_maximums[0]
    negatives_outside = masked_minimums.reshape([batch_size, batch_size])
    negatives_outside = negatives_outside.transpose(0, 1)

    # negatives_inside: largest D_an.
    axis_minimums = torch.min(pdist_matrix, dim=1, keepdim=True)
    masked_maximums = torch.max(torch.mul(pdist_matrix - axis_minimums[0], adjacency_not), dim=1, keepdim=True)[0] + axis_minimums[0]
    negatives_inside = masked_maximums.repeat(1, batch_size)

    semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + pdist_matrix - semi_hard_negatives

        
    mask_positives = adjacency.to(dtype=torch.float32) - torch.diag(torch.ones(batch_size)).to(device)
    if sampling_num is not None:
        mask_positives = select_positives(mask_positives, sampling_num)
    num_positives = mask_positives.sum()
    # print(num_positives)
    # print(num_positives)
    # if sampling_num is not None:
    #     loss_mat = random_sample_rows(loss_mat, sampling_num)
        
    triplet_loss = (torch.max(torch.mul(loss_mat, mask_positives), torch.tensor([0.]).to(device))).sum() / num_positives
    triplet_loss = triplet_loss.to(dtype=embeddings.dtype)
    return triplet_loss

def select_positives(input, n):
    
    positive_indices = torch.nonzero(input == 1)
    num_positives = positive_indices.size(0)
    if num_positives <= n:
        return input
    select_indices = torch.randperm(num_positives)[:n]
    mask = torch.zeros_like(input)
    mask[positive_indices[select_indices][:, 0], positive_indices[select_indices][:, 1]] = 1
    return mask

class TripletLoss(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.margin = margin
    def forward(self, input, target, sampling_num = None):
        return TripletSemiHardLoss(target, input, self.device, self.margin, sampling_num)

class TripletLossHard(nn.Module):
    def __init__(self, margin=1):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.margin = margin
    def forward(self, input, target, **kwargs):
        return TripletHardLoss(target, input, self.device, self.margin)
      

def get_triplets(pdist_matrix, labels):

    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])
    # Build pairwise binary adjacency matrix.
    adjacency = torch.eq(labels, labels.transpose(0, 1))

    # print(adjacency.shape)
    # Invert so we can select negatives only.
    adjacency_not = adjacency.logical_not()

    batch_size = labels.shape[0]
    adjacency = torch.eq(labels, labels.t())
    adjacency_not = adjacency.logical_not()

    # For each anchor, find the hardest positive and hardest negative
    max_pos_dist, _ = pdist_matrix.masked_fill(adjacency_not, float('-inf')).max(1)
    min_neg_dist, _ = pdist_matrix.masked_fill(adjacency, float('inf')).min(1)
    return max_pos_dist, min_neg_dist

def TripletHardLoss(y_true, y_pred, device, margin=0.2):
  
    labels, embeddings = y_true, y_pred
    lshape = labels.shape
    labels = torch.reshape(labels, [lshape[0], 1])
    
    pdist_matrix = pairwise_distance_torch(embeddings, device)
    max_pos_dist, min_neg_dist = get_triplets(pdist_matrix, labels)
    # print(max_pos_dist)
    # print(min_neg_dist)
    losses = torch.relu(max_pos_dist - min_neg_dist + margin)
    # print(losses)
    # print('~~~~~~~~~~~')
    triplet_loss = losses.mean()
    return triplet_loss.to(dtype=embeddings.dtype)
