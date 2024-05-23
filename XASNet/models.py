from __future__ import annotations
from typing import List, Optional, Any, Dict

import torch
import torch.nn.functional as F
import torch_geometric.nn as geomnn
from torch.nn import Linear, LSTM
from torch_geometric.nn import MessagePassing, global_mean_pool, GATConv, GATv2Conv, NNConv, PNAConv
from torch.nn import ModuleList, Dropout
from torch_geometric.nn import global_add_pool

from .utils.weight_init import kaiming_orthogonal_init

from .XASNet_GAT import *
from .XASNet_GraphNet.modules import GraphNetwork

gnn_layers = {
    'gat': geomnn.GATConv,
    'gcn': geomnn.GCNConv,
    'gatv2': geomnn.GATv2Conv,
    'graphConv': geomnn.GraphConv
    }

class XASNet_GNN(torch.nn.Module):
    """
    General implementation of XASNet. The class provides multi-layer GNN 
    and supports different GNN types e.g. GCN, GATv2, GAT, GraphConv.

    Args:
        gnn_name: The type of GNN to train including gcn, gatv2, gat, graphconv.  
        num_layers: Number of GNN layers. 
        in_channels: List of input channels (same size of layers).
        out_channels: List of output channels (same size of layers).
        num_targets: Number of target data points in energy axis 
                of XAS spectrum. Defaults to 100.
        heads: Number of heads in gat and gatv2.
        gat_dp: The rate of dropout in case of gat and gatv2.
    """
    def __init__(
        self, 
        gnn_name: str, 
        num_layers: int,
        in_channels: List[int],
        out_channels: List[int],
        num_targets: int,
        heads: Optional[int] = None,
        gat_dp: float = 0,
        ) -> None:
        super().__init__()
        assert gnn_name in gnn_layers
        assert num_layers > 0
        assert len(in_channels) == num_layers and \
        len(out_channels) == num_layers

        self.gnn_name = gnn_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_targets = num_targets
        self.num_layers = num_layers
        self.heads = heads

        gnn_layer = gnn_layers[gnn_name]
        int_layers = []

        self.batch_norms = torch.nn.ModuleList()
      
        for i, in_c, out_c in zip(range(num_layers - 1), 
                                    in_channels[:-1], out_channels[:-1]):
                  
                if i == 0 and heads: 
                    int_layers.append(gnn_layer(in_c, out_c, heads=heads))
                elif i==0:
                    int_layers.append(gnn_layer(in_c, out_c))
                elif i > 0 and heads:
                    int_layers.append(gnn_layer(in_c*heads, out_c, heads=heads)) 
                elif i > 0:
                    int_layers.append(gnn_layer(in_c, out_c))
                #int_layers.append(torch.nn.BatchNorm1d(out_c))
                #int_layers.append(torch.nn.ReLU(inplace=True))

        for i, out_c in zip(range(num_layers), out_channels):
            self.batch_norms.append(torch.nn.BatchNorm1d(out_c))
        
        if heads:
            int_layers.append(gnn_layer(in_channels[-1]*heads, 
                                        out_channels[-1], heads=1, 
                                        dropout=gat_dp))
        else:
            int_layers.append(gnn_layer(in_channels[-1], 
                                        out_channels[-1]))
        
        self.interaction_layers = torch.nn.ModuleList(int_layers)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.out = torch.nn.Linear(out_channels[-1], num_targets)      
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.interaction_layers:
            if isinstance(m, geomnn.GATConv):
                layers = [m.lin_src, m.lin_dst]
            elif isinstance(m, geomnn.GCNConv):
                layers = [m.lin]
            elif isinstance(m, geomnn.GATv2Conv):
                layers = [m.lin_r, m.lin_l]
            elif isinstance(m, geomnn.GraphConv):
                layers = [m.lin_rel, m.lin_root]
                
            for layer in layers:
                kaiming_orthogonal_init(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
        
        kaiming_orthogonal_init(self.out.weight.data)
        self.out.bias.data.fill_(0.0)   

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch_seg: torch.Tensor) -> torch.Tensor:   
    
        for layer in enumerate(self.interaction_layers[:-1]):
            x = layer[1](x, edge_index)
            x = self.batch_norms[layer[0]](x)
            x = self.dropout(F.relu(x))
            #if isinstance(layer, geomnn.MessagePassing):
                #print('EDGE_INEX')
                #x = layer(x, edge_index)
            #else:
             #   print('NOPE')
              #  x = layer(x)
                #x = self.batch_norms[layer](x)
               # x = self.dropout(x)
         
        x = self.interaction_layers[-1](x, edge_index)    
        x = self.batch_norms[self.num_layers - 1](x)
        x = self.dropout(x)
        x = geomnn.global_mean_pool(x, batch_seg)
        #x = self.dropout(x)
        p = torch.nn.LeakyReLU(0.1)
        out = p(self.out(x))
        return out



class XASNet_GAT(torch.nn.Module):
    """
    More detailed and custom implementation of GAT with different types of GAT layers.
    The model can get deeper using prelayers and residual layers. Moreover, jumping knowledge mechanism 
    as an additional layer is applied to focus on important parts of the node's environment.

    Args:
        node_features_dim: The dimension of node features.
        num_layers: Number of GNN layers. 
        in_channels: List of input channels (same size of layers).
        out_channels: List of output channels (same size of layers).
        n_heads: Number of heads.
        targets: Number of target data points in energy axis 
                of XAS spectrum. Defaults to 100.
        gat_type: Type of the gat layer.
        use_residuals: If true, residual layers is used.
        use_jk: If true, jumping knowledge mechanism is applied.
    """
    def __init__(
        self, 
        node_features_dim: int,
        n_layers: int,
        in_channels: List[int],
        out_channels: List[int],
        n_heads: int,
        targets: int,
        gat_type: str = 'gat_custom',
        use_residuals: bool = False,
        use_jk: bool = False
        ):
        super().__init__()

        self.node_features_dim = node_features_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.targets = targets
        self.gat_type = gat_type
        self.use_residuals = use_residuals
        self.use_jk = use_jk

        assert len(in_channels) == n_layers
        gat_types = {
            'gat_custom' : GATLayerCus,
            'gatv2_custom' : GATv2LayerCus,
            'gat' : GATConv,
            'gatv2' : GATv2Conv
        }
        assert gat_type in gat_types

        if use_residuals:
            self.pre_layer = LinearLayer(node_features_dim, in_channels[0], activation='relu')
            self.res_block = Residual_block(in_channels[0], 4, activation='relu')

        gat_layers = []

        for i, c_in, c_out in zip(range(n_layers-1), \
            in_channels[:-1], out_channels[:-1]):
            if i == 0:
                gat_layers.append(gat_types[gat_type](c_in, c_out, 
                heads=n_heads))
            elif i > 0:
                gat_layers.append(gat_types[gat_type](c_in*n_heads, c_out, 
                heads=n_heads))
            gat_layers.append(torch.nn.ReLU(inplace=True))
    
        gat_layers.append(gat_types[gat_type](in_channels[-1]*n_heads, out_channels[-1]))
        self.gat_layers = torch.nn.ModuleList(gat_layers)

        #jumping knowledge layers
        self.lstm = LSTM(out_channels[-2]*n_heads, out_channels[-2], 
        num_layers=3, bidirectional=True, batch_first=True)
        self.attn = Linear(2*out_channels[-2], 1)

        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear_out = LinearLayer(out_channels[-1], targets)

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch_seg: torch.Tensor) -> torch.Tensor:

        x = self.pre_layer(x)
        x = self.res_block(x)

        xs = []
        for layer in self.gat_layers[:-1]:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)
            else: 
                x = layer(x)
                x = self.dropout(x)
                xs.append(x.flatten(1).unsqueeze(-1))
        
        xs = torch.cat(xs, dim=-1).transpose(1, 2)
        alpha, _ = self.lstm(xs)
        alpha = self.attn(alpha).squeeze(-1)
        alpha = torch.softmax(alpha, dim=-1)
        x = (xs * alpha.unsqueeze(-1)).sum(1)

        x = self.gat_layers[-1](x, edge_index)
        x = global_mean_pool(x, batch_seg)
        x = self.dropout(x)
        out = self.linear_out(x)
        return out

class XASNet_GraphNet(torch.nn.Module):
    """
    GraphNet implementation of XASNet. The global, node and edge states 
    are used in the messsage function.
    """
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_channels: int,
                 out_channels: int,
                 gat_hidd: int,
                 gat_out: int,
                 n_layers: int = 3,
                 n_targets: int = 100):
        """
        Args:
            node_dim (int): Dimension of the nodes' attribute in the graph data.
            edge_dim (int): Dimension of the edges' attribute in the graph data.
            hidden_channels (int): Hidden channels in GraphNet layers.
            out_channels (int): Output channels in GraphNet layers.
            gat_hidd (int): Hidden channels for GAT layer used to obtain 
                the global state of each input graph.
            gat_out (int): Output channels for GAT layer used to obtain 
                the global state of each input graph.
            n_layers (int, optional): Number of layers in GraphNet. Defaults to 3.
            n_targets (int, optional): Number of target data points in energy axis 
                of XAS spectrum. Defaults to 100.
        """
        super().__init__()
        assert n_layers > 0

        #preparing the parameters for global, node and edge models 
        feat_in_node = node_dim + 2*edge_dim + gat_out
        feat_in_edge = 2*out_channels + edge_dim + gat_out
        feat_in_glob = 2*out_channels + gat_out
        node_model_params0 = {"feat_in": feat_in_node, 
                              "feat_hidd": hidden_channels, 
                              "feat_out": out_channels}  
        edge_model_params0 = {"feat_in": feat_in_edge, 
                              "feat_hidd": hidden_channels, 
                              "feat_out": out_channels} 
        global_model_params0 = {"feat_in": feat_in_glob, 
                                "feat_hidd": hidden_channels, 
                                "feat_out": out_channels}

        all_params = {"graphnet0": {"node_model_params": node_model_params0,
            "edge_model_params": edge_model_params0,
            "global_model_params": global_model_params0,
            "gat_in": node_dim,
            "gat_hidd": gat_hidd,
            "gat_out": gat_out}}
        
        for i in range(1, n_layers):
            all_params[f"graphnet{i}"] = {
                "node_model_params": {"feat_in": 4*out_channels, 
                                      "feat_hidd": hidden_channels, 
                                      "feat_out": out_channels},
                "edge_model_params": {"feat_in": 4*out_channels, 
                                      "feat_hidd": hidden_channels, 
                                      "feat_out": out_channels},
                "global_model_params": {"feat_in": 3*out_channels, 
                                      "feat_hidd": hidden_channels, 
                                      "feat_out": out_channels},
                "gat_in": node_dim,
                "gat_hidd": gat_hidd,
                "gat_out": gat_out
                }
            
        graphnets = []
        for v in all_params.values():
            graphnets.append(GraphNetwork(**v))

        self.graphnets = ModuleList(graphnets)

        self.dropout = Dropout(p=0.3)
        self.output_dense = Linear(out_channels, n_targets)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_orthogonal_init(self.output_dense.weight.data)

    def forward(self, graph: Any) -> torch.Tensor:
        for graphnet in self.graphnets:
            graph = graphnet(graph)

        x = global_add_pool(graph.x, graph.batch)
        
        x = self.dropout(x)
        out = self.output_dense(x)
        return out
    
class XASNet_NNconv(torch.nn.Module):
    """
    Generaln implementation of NN_Conv model
    """
    def __init__(
            self,
            gnn_name: str,
            num_layers: int,
            in_channels: List[int],
            out_channels: List[int],
            num_targets: int,
    ) -> None:
        super().__init__()
        assert num_layers > 0
        assert len(in_channels) == num_layers and \
        len(out_channels) == num_layers

        self.gnn_name = gnn_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_targets = num_targets
        self.num_layers = num_layers

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 5 * 64)
        )

        self.conv1 = NNConv(5, 64, nn1, aggr='mean')
        self.batchnorm1 = torch.nn.BatchNorm1d(64)

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64 * 128)
        )

        self.conv2 = NNConv(64, 128, nn2, aggr='mean')
        self.batchnorm2 = torch.nn.BatchNorm1d(128)

        nn3 = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128 * 256)
        )

        self.conv3 = NNConv(128, 256, nn3, aggr='mean')
        self.batchnorm3 = torch.nn.BatchNorm1d(256)

        self.mlp = torch.nn.Linear(256, num_targets)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch_seg: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.batchnorm1(x)

        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)

        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batchnorm3(x)

        x = F.relu(x)

        x = geomnn.global_mean_pool(x, batch_seg)

        p = torch.nn.LeakyReLU(0.1)
        out = p(self.mlp(x))

        return out
    
class XASNet_PNA(torch.nn.Module):
    """
    """
    def __init__(
            self,
            num_targets: int,
            deg: torch.Tensor
    ) -> None:
        super().__init__()

        self.num_targets = num_targets
        self.deg = deg

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.conv1 = PNAConv(in_channels=5, out_channels=64, aggregators=aggregators, scalers=scalers,
                             deg=self.deg, edge_dim=3)
        
        self.batchnorm1 = torch.nn.BatchNorm1d(64)

        self.conv2 = PNAConv(in_channels=64, out_channels=128, aggregators=aggregators, scalers=scalers,
                             deg=deg, edge_dim=3)
        self.batchnorm2 = torch.nn.BatchNorm1d(128)

        self.conv3 = PNAConv(in_channels=128, out_channels=256, aggregators=aggregators, scalers=scalers,
                             deg=deg, edge_dim=3)
        self.batchnorm3 = torch.nn.BatchNorm1d(256)

        self.mlp = torch.nn.Linear(256, num_targets)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                batch_seg: torch.Tensor) -> torch.Tensor:
        print('x', x.shape)
        print('idx', edge_index.shape)
        print('attr', edge_attr.shape)
        x = self.conv1(x, edge_index, edge_attr)
        x= self.batchnorm1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batchnorm3(x)
        x = F.relu(x)

        x = geomnn.global_mean_pool(x, batch_seg)

        p = torch.nn.LeakyReLU(0.1)
        out = p(self.mlp(x))

        return out 