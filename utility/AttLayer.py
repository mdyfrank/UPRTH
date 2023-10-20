import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, merge='sum'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            # print(in_dim, out_dim)
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, graph, h, etype, etype2 =('item', 'iu', 'user'),norm = -1):
        with graph.local_scope():
            head_outs = [attn_head(graph, h, etype) for attn_head in self.heads]
            if self.merge == 'cat':
                # concat on the output feature dimension (dim=1)
                # print(torch.cat(head_outs, dim=1).shape)
                # sys.exit(0)
                return torch.cat(head_outs, dim=1)
            if self.merge == 'sum':
                # concat on the output feature dimension (dim=1)
                # print(torch.cat(head_outs, dim=1).shape)
                # sys.exit(0)
                return torch.sum(torch.stack(head_outs), dim=0)
            else:
                # merge using average
                # print(torch.stack(head_outs).shape)
                # print(torch.mean(torch.stack(head_outs),dim=0).shape)
                # sys.exit(0)
                return torch.nn.functional.elu(torch.mean(torch.stack(head_outs), dim=0))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # equation (1)
        # print(in_dim, out_dim)
        self.fc_src = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_dst = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, h, etype, prompt = 0):
        with graph.local_scope():
            src, _, dst = etype
            # equation (1)
            # print(h)
            if prompt == 1:
                if src == 'item':
                    h_src = src +'_'+ dst
                    h_dst = dst
                elif dst == 'item':
                    h_src = src
                    h_dst = dst +'_'+ src
            else:
                h_src = src
                h_dst = dst
            z_src = self.fc_src(h[h_src])
            z_dst = self.fc_dst(h[h_dst])
            graph.nodes[src].data['z'] = z_src
            graph.nodes[dst].data['z'] = z_dst
            # equation (2)
            graph.apply_edges(self.edge_attention, etype=etype)
            # equation (3) & (4)
            graph.update_all(self.message_func, self.reduce_func, etype=etype)
            # print(etype)
            rst = graph.nodes[dst].data['h']
            return rst

class MotivationAttLayer2(nn.Module):
    def __init__(self, in_dim, out_dim, linear_transform):
        super(MotivationAttLayer2, self).__init__()
        # equation (1)
        self.fc_src = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_dst = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, h, etype, item_user=False):
        with graph.local_scope():
            src, _, dst = etype
            # equation (1)
            if item_user:
                z_src = F.softplus(self.fc_src(h[src]))
            else:
                z_src = self.fc_src(h[src])
            z_dst = self.fc_dst(h[dst])
            graph.nodes[src].data['z'] = z_src
            graph.nodes[dst].data['z'] = z_dst
            # equation (2)
            graph.apply_edges(self.edge_attention, etype=etype)
            # equation (3) & (4)
            graph.update_all(self.message_func, self.reduce_func, etype=etype)
            rst = graph.nodes[dst].data['h']
            return rst


class MotivationAttLayer(nn.Module):
    def __init__(self, in_dim, out_dim, linear_transform):
        super(MotivationAttLayer, self).__init__()
        self.linear_transform = linear_transform
        self.fc_K = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_V = nn.Linear(in_dim, out_dim, bias=False)
        self.fc_Q = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_K.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_Q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_V.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        print(type(edges.src['k']))
        print(edges.src['k'].shape)
        print(edges.dst['q'].shape)
        z1 = torch.matmul(edges.src['k'], edges.dst['q'].t())
        # z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # a = self.attn_fc(z2)
        return {'e': F.leaky_relu(z1)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'v': edges.src['v'], 'k': edges.src['k'], 'q': edges.dst['q']}

    def reduce_func(self, nodes):
        # print(nodes.mailbox['k'].shape)
        # print(nodes.mailbox['k'].unsqueeze(-1).shape)
        # print(nodes.mailbox['q'].shape)
        # print(nodes.mailbox['q'].unsqueeze(-2).shape)
        # norm = nodes.mailbox['q'].shape[-1] ** 0.5
        att = torch.matmul(nodes.mailbox['k'].unsqueeze(-2), nodes.mailbox['q'].unsqueeze(-1)).squeeze(-1)

        # print(att.shape)
        # sys.exit(0)
        z1 = F.leaky_relu(att)
        alpha = F.softmax(z1, dim=1)
        h = torch.sum(alpha * nodes.mailbox['v'], dim=1)
        return {'h': h}

    def forward(self, graph, h, etype):
        with graph.local_scope():
            src, _, dst = etype
            if self.linear_transform:
                v = self.fc_V(h[src])
                k = self.fc_K(h[src])
                q = self.fc_Q(h[dst])
            else:
                v = h[src]
                k = h[src]
                q = h[dst]

            graph.nodes[src].data['v'] = v
            graph.nodes[src].data['k'] = k
            graph.nodes[dst].data['q'] = q
            graph.update_all(self.message_func, self.reduce_func, etype=etype)
            rst = graph.nodes[dst].data['h']
            return rst