import torch
from pytorch_metric_learning.distances.base_distance import BaseDistance
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

class NormMinusLpDistance(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_inverted = True
    def compute_mat(self, query_emb, ref_emb):
        dtype, device = query_emb.dtype, query_emb.device
        if ref_emb is None:
            ref_emb = query_emb
        if dtype == torch.float16:  # cdist doesn't work for float16
            rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)
            output = torch.zeros(rows.size(), dtype=dtype, device=device)
            rows, cols = rows.flatten(), cols.flatten()
            distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
            output[rows, cols] = distances
            return output
        else:
            res = torch.cdist(query_emb, ref_emb, p=self.p)
            min_res = torch.min(torch.abs(res))
            max_res = torch.max(torch.abs(res))
            
            res = (res-min_res)/(max_res-min_res)
            # res = res/torch.max(torch.abs(res))
            
            # print(torch.max(res))
            # print(torch.min(res))
            
            res = res*2 - 1
            res = -res
            # print(res)
            return res
    def pairwise_distance(self, query_emb, ref_emb):
        res = torch.cdist(query_emb, ref_emb, p=self.p)
        min_res = torch.min(torch.abs(res))
        max_res = torch.max(torch.abs(res))
        
        res = (res-min_res)/(max_res-min_res)
        # res = res/torch.max(torch.abs(res))
        
        # print(torch.max(res))
        # print(torch.min(res))
        
        res = res*2 - 1
        res = -res
        # print(res)
        return res


# class NormMinusLpDistance(BaseDistance):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.is_inverted = True
#     def compute_mat(self, query_emb, ref_emb):
#         dtype, device = query_emb.dtype, query_emb.device
#         if ref_emb is None:
#             ref_emb = query_emb
#         if dtype == torch.float16:  # cdist doesn't work for float16
#             rows, cols = lmu.meshgrid_from_sizes(query_emb, ref_emb, dim=0)
#             output = torch.zeros(rows.size(), dtype=dtype, device=device)
#             rows, cols = rows.flatten(), cols.flatten()
#             distances = self.pairwise_distance(query_emb[rows], ref_emb[cols])
#             output[rows, cols] = distances
#             return output
#         else:
#             res = -torch.cdist(query_emb, ref_emb, p=self.p)
#             mean_res = torch.mean(res)
#             std_res = torch.std(res)
#             res = (res - mean_res) / std_res
#             # print(torch.max(res))
#             # print(torch.min(res))
#             # print(res)
#             return res
#     def pairwise_distance(self, query_emb, ref_emb):
#         res = -torch.cdist(query_emb, ref_emb, p=self.p)
#         mean_res = torch.mean(res)
#         std_res = torch.std(res)
#         res = (res - mean_res) / std_res
#         # print(res)
#         return res
