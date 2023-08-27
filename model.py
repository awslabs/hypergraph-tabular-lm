import torch
from layers import *




class Embedding(nn.Module):
  def __init__(self, config):
    super(Embedding, self).__init__()
    self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)  # token embedding
    self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)

  def forward(self, x_s, x_t):
    embedding_s, embedding_t = self.tok_embed(x_s), self.tok_embed(x_t)
    embedding_s, embedding_t = torch.div(torch.sum(embedding_s, dim=1), torch.count_nonzero(x_s, dim=1).unsqueeze(-1)), torch.div(torch.sum(embedding_t, dim=1), torch.count_nonzero(x_t, dim=1).unsqueeze(-1))
    return self.dropout(self.norm(embedding_s)), self.dropout(self.norm(embedding_t))




class EncoderLayer(nn.Module):
  """SetTransformer Encoder Layer"""
  def __init__(self, config):
    super().__init__()
    self.dropout = config.hidden_dropout_prob
    self.V2E = AllSetTrans(config = config)
    self.fuse = nn.Linear(config.hidden_size*2, config.hidden_size)
    self.E2V = AllSetTrans(config = config)


  def forward(self, embedding_s, embedding_t, edge_index):

    # reverse the index
    reversed_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    # from nodes to hyper-edges
    embedding_t_tem = F.relu(self.V2E(embedding_s, edge_index))

    # from hyper-edges to nodes
    embedding_t = torch.cat([embedding_t, embedding_t_tem], dim=-1)
    # fuse the output t_embeds with original t_embeds, or the t_embeds will not have the original info
    embedding_t = F.dropout(self.fuse(embedding_t), p=self.dropout, training=self.training)
    embedding_s = F.relu(self.E2V(embedding_t, reversed_edge_index))
    embedding_s = F.dropout(embedding_s, p=self.dropout, training=self.training)

    return embedding_s, embedding_t



class Encoder(nn.Module):
  def __init__(self, config):
    super(Encoder, self).__init__()
    self.config = config
    self.embed_layer = Embedding(config)
    self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, data):
    embedding_s, embedding_t = self.embed_layer(data.x_s, data.x_t)
    embedding_t = torch.cat([embedding_t, embedding_s], dim=0)

    # Add self-loop
    num_nodes, num_hyper_edges = data.x_s.size(0), data.x_t.size(0)
    self_edge_index = torch.tensor([[i, num_hyper_edges+i] for i in range(num_nodes)]).T
    if ('edge_neg_view' in self.config.to_dict() and self.config.edge_neg_view == 1):
        edge_index = torch.cat([data.edge_index_corr1, self_edge_index.to(data.edge_index_corr1.device)], dim=-1)
    elif ('edge_neg_view' in self.config.to_dict() and self.config.edge_neg_view == 2):
        edge_index = torch.cat([data.edge_index_corr2, self_edge_index.to(data.edge_index_corr2.device)], dim=-1)
    else:
        edge_index = torch.cat([data.edge_index, self_edge_index.to(data.edge_index.device)], dim=-1)


    for i, layer_module in enumerate(self.layer):
      embedding_s, embedding_t  = layer_module(embedding_s, embedding_t, edge_index)
    outputs = (embedding_s, embedding_t[:num_hyper_edges])

    return outputs



class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   """
   def __init__(self, temperature=0.5):
       super().__init__()
  
       self.temperature = temperature
       self.loss_fct = nn.CrossEntropyLoss()


   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       
       z_i = F.normalize(proj_1, p=2, dim=1)
       z_j = F.normalize(proj_2, p=2, dim=1)
       
       cos_sim = torch.einsum('id,jd->ij', z_i, z_j)/self.temperature
       labels = torch.arange(cos_sim.size(0)).long().to(proj_1.device)       
       loss = self.loss_fct(cos_sim, labels)

       return loss
     
     
     
