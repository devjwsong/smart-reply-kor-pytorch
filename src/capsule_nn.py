from kobert_transformers import get_distilkobert_model

import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import random


class CapsuleNetwork(nn.Module):
    def __init__(self, config, is_train=True):
        super(CapsuleNetwork, self).__init__()

        # Seed fixing
        if is_train:
            np.random.seed(777)
            torch.manual_seed(777)
            torch.cuda.manual_seed_all(777)
            random.seed(777)

        self.config = config

        # The embedding layer and encoder can be different according to the model type.
        self.encoder = get_distilkobert_model()
        bert_config = self.encoder.config
        hidden_size = bert_config.dim

        # BERT's embedding layer might be frozen in some cases.
        if self.config['bert_embedding_frozen']:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False

        self.drop = nn.Dropout(self.config['dropout'])

        # Parameters for self attention.
        self.ws1 = nn.Linear(hidden_size, self.config['d_a'], bias=False)
        self.ws2 = nn.Linear(self.config['d_a'], self.config['r'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # Parameters for linear transformation before DetectCaps.
        self.capsule_weights = nn.Parameter(torch.zeros((self.config['r'], hidden_size,
                                                         self.config['num_classes'] * self.config['caps_prop'])))

        # Initialize parameters.
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.ws1.weight)
        nn.init.xavier_uniform_(self.ws2.weight)
        nn.init.xavier_uniform_(self.capsule_weights)

        self.ws1.weight.requires_grad_(True)
        self.ws2.weight.requires_grad_(True)
        self.capsule_weights.requires_grad_(True)

    def forward(self, input_x, is_train=True):
        input_x = input_x.to(self.config['device']) # (B, L)

        # SemanticCaps
        # When using BERT, make attention mask and conduct encoder process.
        attention_mask = (input_x != self.config['pad_id']).float()
        output = self.encoder(input_x, attention_mask=attention_mask)[0]  # (B, L, D_H)

        size = output.size()
        compressed_embeddings = output.view(-1, size[2])  # (B * L, D_H)
        pre_attention = self.tanh(self.ws1(self.drop(compressed_embeddings))) # (B * L, D_A)

        attention = self.ws2(pre_attention).view(size[0], size[1], -1)  # (B, L, R)
        attention = torch.transpose(attention, 1, 2).contiguous() # (B, R, L)
        attention = F.softmax(attention, dim=-1)

        semantic_vecs = torch.bmm(attention, output) # (B, R, D_H) -> This is the final output of SemanticCaps

        # DetectionCaps
        semantic_vecs = self.drop(semantic_vecs)

        semantic_vecs_tiled = torch.unsqueeze(semantic_vecs, -1).repeat(1, 1, 1, self.config['num_classes'] * self.config['caps_prop']) # (B, R, D_H, 1) => (B, R, D_H, K * num_properties)
        prediction_vecs = torch.sum(semantic_vecs_tiled * self.capsule_weights, dim=2) # (B, R, D_H, K * num_properties) => (B, R, K * num_properties)
        prediction_vecs_reshaped = torch.reshape(prediction_vecs, [-1, self.config['r'], self.config['num_classes'], self.config['caps_prop']]) # (B, R, K, num_properties)

        semantic_vecs_shape = semantic_vecs.shape
        logits_shape = np.stack([semantic_vecs_shape[0], self.config['r'], self.config['num_classes']])

        # v: (B, K, num_properties), b: (B, R, K), c: (B, R, K)
        v, b, c = self.routing(prediction_vecs_reshaped, logits_shape, num_dims=4, is_train=is_train)

        logits = self.get_logits(v) # (B, K)
        prediction = prediction_vecs_reshaped # (B, R, K, num_properties)

        return attention, logits, prediction, c

    def get_logits(self, activation):
        logits = torch.norm(activation, dim=-1)
        return logits

    def routing(self, prediction_vecs_reshaped, logits_shape, num_dims, is_train):
        prediction_shape = [3, 0, 1, 2]
        for i in range(num_dims - 4):
            prediction_shape += [i + 4]
        r_t_shape = [1, 2, 3, 0]
        for i in range(num_dims - 4):
            r_t_shape += [i + 4]

        prediction_vecs_trans = prediction_vecs_reshaped.permute(prediction_shape) # (num_properties, B, R, K)
        logits = torch.zeros(logits_shape[0], logits_shape[1], logits_shape[2]).to(self.config['device'])  # (B, R, K) This is bkr in the paper
        if is_train:
            logits = nn.Parameter(logits) # (B, R, K) This is bkr in the paper
        activations = []
        routes = None

        # Iterative routing
        for i in range(self.config['num_iters']):
            routes = F.softmax(logits, dim=2).to(self.config['device']) # (B, R, K) This is cr in the paper
            vote_vecs_unrolled = routes * prediction_vecs_trans # (num_properties, B, R, K)
            vote_vecs = vote_vecs_unrolled.permute(r_t_shape) # (B, R, K, num_properties)

            preactivate = torch.sum(vote_vecs, dim=1) # (B, K, num_properties) This is sk in the paper
            activation = self.squash(preactivate) # (B, K, num_properties) This is vk in the paper
            activations.append(activation)

            act_extended = activation.unsqueeze(1) # (B, 1, K, num_properties)
            tile_shape = np.ones(num_dims, dtype=np.int32).tolist() # [1, 1, 1, 1]
            tile_shape[1] = self.config['r'] # [1, R, 1, 1]
            act_replicated = act_extended.repeat(tile_shape) # (B, R, K, num_properties)
            distances = torch.sum(prediction_vecs_reshaped * act_replicated, dim=3) # (B, R, K, num_properties) => (B, R, K)
            logits = logits + distances # (B, R, K)

        return activations[self.config['num_iters']-1], logits, routes

    def squash(self, input_tensor):
        # Execute Squash function.
        norm = torch.norm(input_tensor, dim=2, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (0.5 + norm_squared))

    def get_margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        logits = raw_logits - 0.5
        positive_cost = labels * (logits < margin).float() * ((logits - margin) ** 2)
        negative_cost = (1 - labels) * (logits > -margin).float() * ((logits + margin) ** 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    def get_loss(self, label, logits, attention):
        label = label.to(self.config['device'])
        loss_value = self.get_margin_loss(label.float(), logits)
        loss_value = torch.mean(loss_value)

        self_atten_mul = torch.matmul(attention, attention.permute([0, 2, 1])).float()
        sample_num, att_matrix_size, _ = self_atten_mul.shape
        self_atten_loss = (torch.norm(self_atten_mul - torch.from_numpy(np.identity(att_matrix_size)).float().to(self.config['device'])).float()) ** 2

        return 1000 * loss_value + self.config['alpha'] * torch.mean(self_atten_loss)