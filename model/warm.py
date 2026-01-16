import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.autograd as autograd
import os
import pickle as pkl
from .rectified_flow import RectifiedFlowWrapper

from .diffusion import GaussianDiffusion, get_named_beta_schedule
from .unet import Unet
import math
from .flow_model import UniModMLP, ExFM, ResNetMLP


class DropoutNet(nn.Module):
    def __init__(self, model: nn.Module, device, item_id_name='item_id'):
        super(DropoutNet, self).__init__()
        self.model = model
        self.item_id_name = item_id_name
        item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                            .repeat(item_emb.num_embeddings, 1)
        return

    def foward_without_itemid(self, xdict):
        bsz = xdict[self.item_id_name].shape[0]
        target = self.model.forward_with_item_id_emb(self.mean_item_emb.repeat([bsz, 1]), xdict)
        return target

    def foward(self, xdict):
        item_id_emb = xdict[self.item_id_name]
        target = self.model.forward_with_item_id_emb(item_id_emb, xdict)
        return target

class MetaE(nn.Module):
    
    def __init__(self, 
                 model: nn.Module,
                 warm_features: list,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(MetaE, self).__init__()
        self.build(model, warm_features, device, item_id_name, emb_dim)
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model 
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        output_embedding_size = 0
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            type = self.model.description[item_f][1]
            if type == 'spr' or type == 'seq':
                output_embedding_size += emb_dim
            elif type == 'ctn':
                output_embedding_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 

        self.itemid_generator = nn.Sequential(
            nn.Linear(output_embedding_size, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim),
        )
        return

    def init_metaE(self):
        for name, param in self.named_parameters():
            if 'itemid_generator' in name:
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_metaE(self):
        for name, param in self.named_parameters():
            if 'itemid_generator' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        # sideinfo_emb = torch.mean(torch.stack(item_embs, dim=1), dim=1)
        sideinfo_emb = torch.concat(item_embs, dim=1)
        pred = self.itemid_generator(sideinfo_emb)
        return pred

    def forward(self, features_a, label_a, features_b, criterion=torch.nn.BCELoss(), lr=0.001):
        new_item_id_emb = self.warm_item_id(features_a)
        target_a = self.model.forward_with_item_id_emb(new_item_id_emb, features_a)
        loss_a = criterion(target_a, label_a.float())
        grad = autograd.grad(loss_a, new_item_id_emb, create_graph=True)[0]
        new_item_id_emb_update = new_item_id_emb - lr * grad
        target_b = self.model.forward_with_item_id_emb(new_item_id_emb_update, features_b)
        return loss_a, target_b

class MWUF(nn.Module):

    def __init__(self, 
                 model: nn.Module,
                 item_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(MWUF, self).__init__()
        self.build(model, item_features, train_loader, device, item_id_name, emb_dim)
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):

        self.model = model 
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            type = self.model.description[item_f][1]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 
        item_emb = self.model.emb_layer[self.item_id_name]
        new_item_emb = torch.mean(item_emb.weight.data, dim=0, keepdims=True) \
                            .repeat(item_emb.num_embeddings, 1)
        self.new_item_emb = nn.Embedding.from_pretrained(new_item_emb, \
                                                                freeze=False)
        # self.new_item_emb = nn.Embedding(item_emb.num_embeddings, item_emb.embedding_dim)
        self.meta_shift = nn.Sequential(
            nn.Linear(emb_dim, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim)
        )
        self.meta_scale = nn.Sequential(
            nn.Linear(self.output_emb_size, 16),
            nn.ReLU(),
            nn.Linear(16, emb_dim)
        )
        self.get_item_avg_users_emb(train_loader, device)
        return

    def init_meta(self):
        for name, param in self.named_parameters():
            if ('meta_scale') in name or ('meta_shift' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_meta(self):
        for name, param in self.named_parameters():
            if ('meta_shift' in name) or ('meta_scale' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return
    
    def optimize_new_item_emb(self):
        for name, param in self.named_parameters():
            if 'new_item_emb' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def cold_forward(self, x_dict):
        item_id_emb = self.new_item_emb(x_dict[self.item_id_name])
        target = self.model.forward_with_item_id_emb(item_id_emb, x_dict)
        return target

    def warm_item_id(self, x_dict):
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.new_item_emb(item_ids).detach().squeeze()
        user_emb = self.avg_users_emb(item_ids).detach().squeeze()
        if user_emb.sum() == 0:
            user_emb = self.model.emb_layer['user_id'](x_dict['user_id']).squeeze()
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.model.emb_layer[item_f](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.model.emb_layer[item_f](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        item_emb = torch.concat(item_embs, dim=1).detach()
        # warm
        scale = self.meta_scale(item_emb)
        shift = self.meta_shift(user_emb)
        warm_item_id_emb = (scale * item_id_emb + shift)
        return warm_item_id_emb

    def forward(self, x_dict):
        warm_item_id_emb = self.warm_item_id(x_dict).unsqueeze(1)
        target = self.model.forward_with_item_id_emb(warm_item_id_emb, x_dict)
        return target

    def get_item_avg_users_emb(self, data_loaders, device):
        dataset_name = data_loaders.dataset.dataset_name
        path = "./datahub/item2users/{}_item2users.pkl".format(dataset_name)
        if os.path.exists(path):
            with open(path, 'rb+') as f:
                item2users = pkl.load(f)
        else:
            item2users = {}
            for features, _ in data_loaders:
                u_ids = features['user_id'].squeeze().tolist()
                i_ids = features['item_id'].squeeze().tolist()
                for i in range(len(i_ids)):
                    iid, uid = u_ids[i], i_ids[i]
                    if iid not in item2users.keys():
                        item2users[iid] = []
                    item2users[iid].append(uid)
            with open(path, 'wb+') as f:
                pkl.dump(item2users, f)
        avg_users_emb = []
        emb_dim = self.model.emb_layer[self.item_id_name].embedding_dim
        for item in range(self.model.emb_layer[self.item_id_name].num_embeddings):
            if item in item2users.keys():
                users = torch.Tensor(item2users[item]).long().to(device)
                avg_users_emb.append(self.model.emb_layer['user_id'](users).mean(dim=0))
            else:
                avg_users_emb.append(torch.zeros(emb_dim).to(device))
        avg_users_emb = torch.stack(avg_users_emb, dim=0) 
        self.avg_users_emb = nn.Embedding.from_pretrained(avg_users_emb, \
                                                                freeze=True)
        return

class CVAR(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16):
        super(CVAR, self).__init__()
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
        return

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()

        transformer_depth = 2
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Linear(1, emb_dim)
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f)
        self.vf_fn = UniModMLP(num_layers=transformer_depth, d_token=emb_dim, d_in=emb_dim).to(device)
        self.flow_model = ExFM(
            vf_fn=self.vf_fn,
            device=device,
        )
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_encoder = nn.Linear(emb_dim, 16)
        self.decoder = nn.Linear(17, 16)
        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_cvar(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features:
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                if x.dim() == 1: x = x.unsqueeze(1)
                emb = self.warmup_emb_layer[name](x)
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x) \
                        .sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)
        condition = torch.stack(item_embs, dim=1)
        mean = self.mean_encoder(item_id_emb)
        reg_term = self.flow_model.train_loss(condition, mean)

        z_p = self.flow_model.sample(condition)
        recon_term = torch.zeros([1]).to(self.device)
        freq = x_dict['count']


        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        return pred_p, reg_term, recon_term


    def forward(self, x_dict):
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term


# class CVAR(nn.Module):
#     def __init__(self,
#                  model: nn.Module,
#                  warm_features: list,
#                  train_loader,
#                  device,
#                  item_id_name = 'item_id',
#                  emb_dim = 16):
#         super(CVAR, self).__init__()
#         self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
#         return
#
#     def build(self,
#               model: nn.Module,
#               item_features: list,
#               train_loader,
#               device,
#               item_id_name = 'item_id',
#               emb_dim = 16):
#         self.model = model
#         self.device = device
#         assert item_id_name in model.item_id_name, \
#                         "illegal item id name: {}".format(item_id_name)
#         self.item_id_name = item_id_name
#         self.item_features = []
#         self.output_emb_size = 0
#         self.warmup_emb_layer = nn.ModuleDict()
#         for item_f in item_features:
#             assert item_f in model.features, "unkown feature: {}".format(item_f)
#             size, type = self.model.description[item_f]
#             if type == 'spr' or type == 'seq':
#                 self.output_emb_size += emb_dim
#                 self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
#             elif type == 'ctn':
#                 self.output_emb_size += 1
#             else:
#                 raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
#             self.item_features.append(item_f)
#         self.vf_fn = ResNetMLP(emb_dim, self.output_emb_size)
#         self.flow_model = ExFM(self.vf_fn, device)
#         self.origin_item_emb = self.model.emb_layer[self.item_id_name]
#         self.mean_encoder = nn.Linear(emb_dim, 16)
#         self.decoder = nn.Linear(17, 16)
#         return
#
#     def wasserstein(self, mean1, log_v1, mean2, log_v2):
#         p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
#         p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
#         return torch.sum(p1 + p2)
#
#     def init_all(self):
#         for name, param in self.named_parameters():
#             torch.nn.init.uniform_(param, -0.01, 0.01)
#
#     def optimize_all(self):
#         for name, param in self.named_parameters():
#             param.requires_grad_(True)
#         return
#
#     def init_cvar(self):
#         for name, param in self.named_parameters():
#             if ('encoder') in name or ('decoder' in name):
#                 torch.nn.init.uniform_(param, -0.01, 0.01)
#
#     def optimize_cvar(self):
#         for name, param in self.named_parameters():
#             if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
#                 param.requires_grad_(True)
#             else:
#                 param.requires_grad_(False)
#         return
#
#     def warm_item_id(self, x_dict):
#         # get original item id embeddings
#         item_ids = x_dict[self.item_id_name]
#         item_id_emb = self.origin_item_emb(item_ids).squeeze()
#         # get embedding of item features
#         item_embs = []
#         for item_f in self.item_features:
#             type = self.model.description[item_f][1]
#             name = "warmup_{}".format(item_f)
#             x = x_dict[item_f]
#             if type == 'spr':
#                 emb = self.warmup_emb_layer[name](x).squeeze()
#             elif type == 'ctn':
#                 emb = x
#             elif type == 'seq':
#                 emb = self.warmup_emb_layer[name](x) \
#                         .sum(dim=1, keepdim=True).squeeze()
#             else:
#                 raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
#             item_embs.append(emb)
#         condition = torch.cat(item_embs, dim=1)
#         mean = self.mean_encoder(item_id_emb)
#         reg_term = self.flow_model.train_loss(condition, mean)
#
#         z_p = self.flow_model.sample(condition)
#         recon_term = torch.zeros([1]).to(self.device)
#         freq = x_dict['count']
#
#
#         pred_p = self.decoder(torch.concat([z_p, freq], 1))
#
#         return pred_p, reg_term, recon_term
#
#
#     def forward(self, x_dict):
#         warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
#         target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
#         return target, recon_term, reg_term



class DIFF(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name = 'item_id',
                 emb_dim = 16,
                 T = None,
                 w = None,
                 v = None,
                 noise_scale = None,
                 noise_min = None,
                 noise_max = None,
                 eta=None,
                 timesteps=None
                 ):
        super(DIFF, self).__init__()
        self.T = T
        self.w = w
        self.v = v
        self.noise_scale=noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.eta = eta
        self.timesteps = timesteps
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)

        betas = get_named_beta_schedule(num_diffusion_timesteps = self.T,
                                        noise_scale=self.noise_scale,
                                        noise_min=self.noise_min,
                                        noise_max=self.noise_max
                                        )
        self.unet = Unet(in_dims=[emb_dim, emb_dim, emb_dim],
                     out_dims=[emb_dim, emb_dim, emb_dim],
                     emb_size=2*emb_dim)
        
        self.diffusion = GaussianDiffusion(
            dtype=torch.float32,
            model=self.unet,
            betas = betas,
            w = self.w,
            v = self.v,
            noise_scale=self.noise_scale,
            noise_min = self.noise_min,
            noise_max = self.noise_max,
            eta = self.eta,
            timesteps=self.timesteps,
            device=device)

        # print('self.T:', self.T)
        # print('self.w:', self.w)
        # print('self.v:', self.v)
        # print('self.nosie_scale:', self.noise_scale)
        # print('self.noise_min:', self.noise_min)
        # print('self.noise_max:', self.noise_max)
        # print('betas: ', betas)
        # print('eta: ', eta)
        # print('timesteps: ', timesteps)
        
        return 

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name = 'item_id',
              emb_dim = 16):
        self.model = model 
        self.device = device
        assert item_id_name in model.item_id_name, \
                        "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f) 
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]
        self.mean_encoder = nn.Linear(emb_dim, emb_dim)
        #self.log_v_encoder = nn.Linear(emb_dim, 16)
        #self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
        self.mean_encoder_p = nn.Linear(self.output_emb_size,  emb_dim)
        #self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
        #self.decoder = nn.Linear(17, 16)
        self.decoder = nn.Linear(emb_dim + 1, emb_dim)

        #self.decoder = nn.Linear(emb_dim, emb_dim)
        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_diff(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)


    def optimize_diff(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    def warm_item_id(self, x_dict):
        # get original item id embeddings
        item_ids = x_dict[self.item_id_name]
        item_id_emb = self.origin_item_emb(item_ids).squeeze()
        # get embedding of item features
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x).sum(dim=1, keepdim=True).squeeze()
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            item_embs.append(emb)


        mean = self.mean_encoder(item_id_emb)
            
        sideinfo_emb = torch.concat(item_embs, dim=1)

        # bsz = sideinfo_emb.shape[0]
        # t = torch.rand(bsz).to(self.device)
        # t = t[:, None]
        # mean_t = mean
        # if mean.shape[1] > 0:
        #     noise = torch.randn_like(mean)
        #     mean_t = t * mean + (1 - t) * noise  # + noise * sigma_num

        #fake_time = torch.ones((bsz, 16), requires_grad=False).to(self.device)
        
        mean_p = self.mean_encoder_p(sideinfo_emb)
        # pred = self.vf_fn(mean_t, mean_p, t.squeeze())

        # pred_p = self.vf_fn(torch.cat([mean_t, mean_p], dim=1), t.squeeze())

        reg_term = self.diffusion.trainloss(mean, x_T=mean_p)
        #reg_term = self.diffusion.trainloss(mean_p, x_T = mean)

        #z_p =  self.diffusion.rec_backward(x_t=mean_p)
        z_p = self.diffusion.rec_backward_ddim(x_t = mean_p)
        
        #log_v = self.log_v_encoder(item_id_emb)
        #mean_p = self.mean_encoder_p(sideinfo_emb)
        #log_v_p = self.log_v_encoder_p(sideinfo_emb)
        #reg_term = self.wasserstein(mean, log_v, mean_p, log_v_p)
        #reg_term = torch.square(mean - mean_p).sum()
        #reg_term = torch.zeros([1]).to(self.device)
        recon_term = torch.zeros([1]).to(self.device)
        #z = mean
        #z_p = mean_p
        #z = mean + 1e-4 * torch.exp(log_v * 0.5) * torch.randn(mean.size()).to(self.device)
        #z_p = mean_p + 1e-4 * torch.exp(log_v_p * 0.5) * torch.randn(mean_p.size()).to(self.device)
        freq = x_dict['count']

        #pred = self.decoder(z)
        #pred = self.decoder(torch.concat([z, freq], 1))
        #return pred, reg_term, recon_term
        pred_p = self.decoder(torch.concat([z_p, freq], 1))
        #recon_term = torch.square(pred - item_id_emb).sum(-1).mean()
        return pred_p, reg_term, recon_term

    def forward(self, x_dict):
        warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
        target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
        return target, recon_term, reg_term


# class FLOW(nn.Module):
#     def __init__(self,
#                  model: nn.Module,
#                  warm_features: list,
#                  train_loader,
#                  device,
#                  item_id_name='item_id',
#                  emb_dim=16,
#                  timesteps=None,
#                  inference_steps=10,):
#         super(FLOW, self).__init__()
#         self.timesteps = timesteps
#         self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
#         self.inference_steps = inference_steps
#         self.unet = Unet(in_dims=[emb_dim, emb_dim, emb_dim],
#                          out_dims=[emb_dim, emb_dim, emb_dim],
#                          emb_size=2 * emb_dim)

#         self.flow_model = RectifiedFlowWrapper(velocity_model=self.unet, device=device)

#         return

#     def build(self,
#               model: nn.Module,
#               item_features: list,
#               train_loader,
#               device,
#               item_id_name='item_id',
#               emb_dim=16,
#               ):
#         self.model = model
#         self.device = device
#         assert item_id_name in model.item_id_name, \
#             "illegal item id name: {}".format(item_id_name)
#         self.item_id_name = item_id_name
#         self.item_features = []
#         self.output_emb_size = 0
#         self.warmup_emb_layer = nn.ModuleDict()
#         for item_f in item_features:
#             assert item_f in model.features, "unkown feature: {}".format(item_f)
#             size, type = self.model.description[item_f]
#             if type == 'spr' or type == 'seq':
#                 self.output_emb_size += emb_dim
#                 self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
#             elif type == 'ctn':
#                 self.output_emb_size += 1
#             else:
#                 raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
#             self.item_features.append(item_f)
#         self.origin_item_emb = self.model.emb_layer[self.item_id_name]
#         self.mean_encoder = nn.Linear(emb_dim, emb_dim)
#         # self.log_v_encoder = nn.Linear(emb_dim, 16)
#         # self.mean_encoder_p = nn.Linear(self.output_emb_size, 16)
#         self.mean_encoder_p = nn.Linear(self.output_emb_size, emb_dim)
#         # self.log_v_encoder_p = nn.Linear(self.output_emb_size, 16)
#         # self.decoder = nn.Linear(17, 16)
#         self.decoder = nn.Linear(emb_dim + 1, emb_dim)

#         # self.decoder = nn.Linear(emb_dim, emb_dim)
#         return

#     def wasserstein(self, mean1, log_v1, mean2, log_v2):
#         p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
#         p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
#         return torch.sum(p1 + p2)

#     def init_all(self):
#         for name, param in self.named_parameters():
#             torch.nn.init.uniform_(param, -0.01, 0.01)

#     def optimize_all(self):
#         for name, param in self.named_parameters():
#             param.requires_grad_(True)
#         return

#     def init_flow(self):
#         for name, param in self.named_parameters():
#             if ('encoder') in name or ('decoder' in name):
#                 torch.nn.init.uniform_(param, -0.01, 0.01)

#     def optimize_flow(self):
#         for name, param in self.named_parameters():
#             if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
#                 param.requires_grad_(True)
#             else:
#                 param.requires_grad_(False)
#         return

#     def warm_item_id(self, x_dict):
#         # get original item id embeddings
#         item_ids = x_dict[self.item_id_name]
#         item_id_emb = self.origin_item_emb(item_ids).squeeze()
#         # get embedding of item features
#         item_embs = []
#         for item_f in self.item_features:
#             type = self.model.description[item_f][1]
#             name = "warmup_{}".format(item_f)
#             x = x_dict[item_f]
#             if type == 'spr':
#                 emb = self.warmup_emb_layer[name](x).squeeze()
#             elif type == 'ctn':
#                 emb = x
#             elif type == 'seq':
#                 emb = self.warmup_emb_layer[name](x) \
#                     .sum(dim=1, keepdim=True).squeeze()
#             else:
#                 raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
#             item_embs.append(emb)

#         mean = self.mean_encoder(item_id_emb)
#         sideinfo_emb = torch.concat(item_embs, dim=1)
#         mean_p = self.mean_encoder_p(sideinfo_emb)

#         reg_term = self.flow_model.get_loss(x_1=mean, x_0=mean_p)
# #         z_p = self.flow_model.sample(mean_p)
        
#         # 生成过程 (Sampling)
#         if self.training:
#             # 训练时为了加速，可以用 mean (真实ID) 作为生成的替代，或者做单步预测
#             # 这里为了简单直接用 Target，或者你可以用 self.fm.sample(mean_p, steps=1)
#             z_p = mean 
#         else:
#             # 推理时：从 Side Info 出发，走 10 步欧拉积分到达 ID Embedding
#             z_p = self.flow_model.sample(x_0=mean_p, steps=self.inference_steps)
#             # 在推理阶段，reg_term 设为 0
#             reg_term = torch.zeros(1, device=self.device)
#         recon_term = torch.zeros([1]).to(self.device)
#         freq = x_dict['count']
#         pred_p = self.decoder(torch.concat([z_p, freq], 1))
#         return pred_p, reg_term, recon_term

#     def forward(self, x_dict):
#         warm_id_emb, reg_term, recon_term = self.warm_item_id(x_dict)
#         target = self.model.forward_with_item_id_emb(warm_id_emb, x_dict)
#         return target, recon_term, reg_term


def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(timesteps.device) * 2 * math.pi
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



class FLOW(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 warm_features: list,
                 train_loader,
                 device,
                 item_id_name='item_id',
                 emb_dim=16,
                 timesteps=None,
                 inference_steps=10,):
        super(FLOW, self).__init__()
        self.timesteps = timesteps
        self.build(model, warm_features, train_loader, device, item_id_name, emb_dim)
        self.inference_steps = inference_steps
        self.time_emb_dim = 10  # 设定时间编码维度
        self.emb_dim = emb_dim
        self.condition_projector = nn.Sequential(
            nn.Linear(self.output_emb_size + 1, 64),
            nn.SiLU(), # SiLU/Swish 在生成模型中很常用
            nn.Linear(64, emb_dim)
        )
        input_dim = emb_dim + self.time_emb_dim + emb_dim
        
        self.velocity_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, emb_dim) # 输出维度必须与 x 相同
        )
        return

    def build(self,
              model: nn.Module,
              item_features: list,
              train_loader,
              device,
              item_id_name='item_id',
              emb_dim=16,
              ):
        self.model = model
        self.device = device
        assert item_id_name in model.item_id_name, \
            "illegal item id name: {}".format(item_id_name)
        self.item_id_name = item_id_name
        self.item_features = []
        self.output_emb_size = 0
        self.warmup_emb_layer = nn.ModuleDict()
        for item_f in item_features:
            assert item_f in model.features, "unkown feature: {}".format(item_f)
            size, type = self.model.description[item_f]
            if type == 'spr' or type == 'seq':
                self.output_emb_size += emb_dim
                self.warmup_emb_layer["warmup_{}".format(item_f)] = nn.Embedding(size, emb_dim)
            elif type == 'ctn':
                self.output_emb_size += 1
            else:
                raise ValueError('illegal feature tpye for warm: {}'.format(item_f))
            self.item_features.append(item_f)
        self.origin_item_emb = self.model.emb_layer[self.item_id_name]

        return

    def wasserstein(self, mean1, log_v1, mean2, log_v2):
        p1 = torch.sum(torch.pow(mean1 - mean2, 2), 1)
        p2 = torch.sum(torch.pow(torch.sqrt(torch.exp(log_v1)) - torch.sqrt(torch.exp(log_v2)), 2), 1)
        return torch.sum(p1 + p2)

    def init_all(self):
        for name, param in self.named_parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_all(self):
        for name, param in self.named_parameters():
            param.requires_grad_(True)
        return

    def init_flow(self):
        for name, param in self.named_parameters():
            if ('encoder') in name or ('decoder' in name):
                torch.nn.init.uniform_(param, -0.01, 0.01)

    def optimize_flow(self):
        for name, param in self.named_parameters():
            if ('encoder' in name) or ('decoder' in name) or ('warmup' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        return

    
    
    def get_raw_condition(self, x_dict):
        """辅助函数：提取原始 Side Info 并拼接"""
        item_embs = []
        for item_f in self.item_features: 
            type = self.model.description[item_f][1]
            name = "warmup_{}".format(item_f)
            x = x_dict[item_f]
            if type == 'spr':
                emb = self.warmup_emb_layer[name](x).squeeze()
            elif type == 'ctn':
                emb = x
            elif type == 'seq':
                emb = self.warmup_emb_layer[name](x).sum(dim=1,keepdim=True).squeeze()
            item_embs.append(emb)
        
        sideinfo_emb = torch.concat(item_embs, dim=1)
        freq = x_dict['count']
        if freq.dim() == 1: freq = freq.unsqueeze(1)
        
        # 返回拼接后的原始条件 [B, output_emb_size + 1]
        return torch.concat([sideinfo_emb, freq], dim=1)

    def forward(self, x_dict):
        """
        对齐 CVAR 的 forward 接口: return target, recon_term, reg_term
        """
        # 1. 准备真实数据 x1 (Target Item Embedding)
        item_ids = x_dict[self.item_id_name]
        x1 = self.origin_item_emb(item_ids).squeeze()
        
        # 2. 准备 Condition (经过 Projector)
        raw_condition = self.get_raw_condition(x_dict)
        c_emb = self.condition_projector(raw_condition) # [B, emb_dim]

        # 3. 采样时间 t 和噪声 x0
        bsz = x1.shape[0]
        t = torch.rand(bsz).to(self.device) # [B]
        x0 = torch.randn_like(x1)           # [B, emb_dim]
        
        # 4. 计算 Time Embedding
        t_emb = timestep_embedding_pi(t, self.time_emb_dim) # [B, time_emb_dim]
        
        # 5. Flow Matching 核心逻辑
        # 构造插值 x_t
        t_view = t.view(-1, 1)
        x_t = (1 - t_view) * x0 + t_view * x1
        # 目标速度 u_t
        u_t = x1 - x0
        
        # 网络预测 v_pred
        net_input = torch.cat([x_t, t_emb, c_emb], dim=1)
        v_pred = self.velocity_net(net_input)
        
        # 计算 Loss (MSE)
        fm_loss = torch.mean(torch.sum((v_pred - u_t) ** 2, dim=1))

        # 6. 计算主模型预测 (Target Prediction)
        # 训练时通常使用真实的 x1 来计算主任务 Loss，或者使用 x_t 也是一种策略
        # 但为了保持与原代码逻辑一致(原代码是用生成的 warm_emb)，
        # 且 Flow Matching 训练是独立的，这里我们直接用 x1 传给主模型，
        # 或者是 x0 经过 ODE 求解后的结果？
        # 考虑到训练效率，通常在 forward 里不进行 ODE 采样。
        # 原 CVAR 代码这里使用的是采样出的 z (reparameterization)，
        # 对于 FM，训练时直接用 x1 (真实 embedding) 传给主模型作为 target 是合理的，
        # 因为我们希望 backbone 适应真实的 embedding。
        target_pred = self.model.forward_with_item_id_emb(x1, x_dict)
        
        # --- 接口对齐 ---
        # target      -> target_pred (主任务预测)
        # recon_term  -> fm_loss     (Flow Matching Loss, 权重 1.0)
        # reg_term    -> 0.0         (正则项，FM 不需要，且权重仅 1e-4)
        
        return target_pred, 0.0001 * fm_loss, torch.tensor(0.0, device=self.device)

    @torch.no_grad()
    def warm_item_id(self, x_dict, steps=10):
        """
        推理/生成阶段专用函数
        替代原有的 warm_item_id 功能，返回生成的 Embedding
        """
        # 1. 准备 Condition
        raw_condition = self.get_raw_condition(x_dict)
        c_emb = self.condition_projector(raw_condition)
        
        bsz = c_emb.shape[0]
        
        # 2. 从噪声 x0 开始
        x = torch.randn(bsz, self.emb_dim).to(self.device)
        
        # 3. ODE 求解 (Euler)
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i * dt
            t_tensor = torch.full((bsz,), t_val).to(self.device)
            t_emb = timestep_embedding_pi(t_tensor, self.time_emb_dim)
            
            # 预测速度
            net_input = torch.cat([x, t_emb, c_emb], dim=1)
            v_pred = self.velocity_net(net_input)
            
            # 更新位置
            x = x + v_pred * dt
            
        # 返回生成的 Embedding (为了兼容接口，只返回一个值)
        # 原接口返回: pred_p, reg_term, recon_term
        # 这里仅用于生成，所以后两个可以为 None 或 0
        return x, 0.0, 0.0