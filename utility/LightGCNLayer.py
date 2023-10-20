import torch
import torch.nn as nn
import dgl.function as fn

class LightGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, h, etype, norm_2 = -1):
        with graph.local_scope():
            src, _, dst = etype
            feat_src = h[src]
            feat_dst = h[dst]
            # aggregate_fn = fn.copy_src('h', 'm')
            aggregate_fn = fn.copy_u('h', 'm')

            out_degrees = graph.out_degrees(etype=etype).float().clamp(min=1)
            norm_src = torch.pow(out_degrees, norm_2)

            shp_src = norm_src.shape + (1,) * (feat_src.dim() - 1)
            norm_src = torch.reshape(norm_src, shp_src)
            feat_src = feat_src * norm_src

            graph.nodes[src].data['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'), etype=etype)

            rst = graph.nodes[dst].data['h']
            in_degrees = graph.in_degrees(etype=etype).float().clamp(min=1)
            norm_dst = torch.pow(in_degrees, norm_2)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm_dst, shp_dst)
            rst = rst * norm

            return rst
