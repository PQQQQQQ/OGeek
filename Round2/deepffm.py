
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.embedding_atten_v2 import EmbeddingAtten_v2
from module.FM import FFM,FM
from torch.nn import Module
import logging
from loss.focal_loss import bce_focal_loss
from loss.hinge_loss import hinge_loss
from module.dice import Dice
# Ref https://github.com/DiligentPanda/Tencent_Ads_Algo_2018/blob/master/src/model/DIN_ffm_v3_r.py
class DeepFFM(Module):
    def __init__(self, n_out, embedding_feat_infos_dict, k,
                 loss_cfg):
        '''
        :param n_out: the number of output of network, shoule be 1.
        :param loss_cfg:
        '''
        super(DeepFFM,self).__init__()
        self.embedding_feat_infos_dict = embedding_feat_infos_dict

        self.n_field = len(self.embedding_feat_infos_dict)
        self.n_embed_dim = k*self.n_field

        self.k = k
        self.loss_cfg = loss_cfg
        self.n_out = n_out

        self.construct_embedders()
        self.construct_loss(self.loss_cfg)

        self.n_feat = len(self.embedding_feat_infos_dict)
        # assert (self.n_field == self.n_feat)
        self.n_total_dim = sum([self.get_embedder(info.name).dim for k,info in self.embedding_feat_infos_dict.items()])

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fm = FFM(self.n_feat)

        self.n_output_feat = 64

        self.linear1 = nn.Linear(self.n_feat*self.n_embed_dim+self.n_feat*(self.n_feat+1)//2, 128)
        self.linear2 = nn.Linear(128, self.n_output_feat)
        self.linear3 = nn.Linear(64,self.n_out)#bias=False)
        self.dice2 = Dice(dim=self.n_output_feat)
        self.dice1 = Dice(dim=128)
        self.bn0 = nn.BatchNorm1d(num_features=self.n_feat*self.n_embed_dim+self.n_feat*(self.n_feat+1)//2)

        #self.init_weights()

    def init_weights(self):
        init = nn.init.xavier_normal
        init(self.linear1.weight)
        init(self.linear2.weight)
        init(self.linear3.weight)
        init(self.w_fm.weight)
        init(self.w_1ord.weight)
        logging.info("model init its weights.")

    def construct_embedders(self):
        for name,feat_info in self.embedding_feat_infos_dict.items():
   
            fconfig = {"dim": self.n_embed_dim//self.n_field, "atten": False}
            embedder_name = "embedder_{}".format(name)
            embedder = EmbeddingAtten_v2(num_embeddings=feat_info.n_val,
                                      embedding_dim=fconfig["dim"],
                                      n_field=self.n_field, atten=fconfig["atten"],
                                      mode="sum", norm_type=2, max_norm=1,
                                      padding_idx=feat_info.empty_val)
            # we simply use default orthogonal initialization
            embedder.init_weight(embedder_name,None)
            self.__setattr__(embedder_name,embedder)
            logging.info("{} constructs its embedder".format(name))

    def construct_loss(self,loss_cfg):
        from functools import partial
        if loss_cfg["name"] == "bce_focal":
            def loss_func(input, target, weight=None, size_average=True, reduce=True):
                return bce_focal_loss(input,target,gamma=loss_cfg["gamma"],weight=weight,size_average=size_average,reduce=reduce)
            self.loss=loss_func
            logging.info("model use {} loss with gamma={}.".format(loss_cfg["name"],loss_cfg["gamma"]))
        elif loss_cfg["name"] == "bce":
            def loss_func(input, target, weight=None, size_average=True, reduce=True):
                return F.binary_cross_entropy_with_logits(input,target,weight=weight,size_average=size_average)
            self.loss=loss_func
            logging.info("model use {} loss with gamma={}.".format(loss_cfg["name"],loss_cfg["gamma"]))
        elif loss_cfg["name"] == "hinge":
            def loss_func(input, target, weight=None, size_average=True, reduce=True):
                return hinge_loss(input, target, weight=weight, size_average=size_average)
            self.loss = loss_func
            logging.info("model use {} loss with gamma={}.".format(loss_cfg["name"], loss_cfg["gamma"]))
        else:
            raise NotImplementedError

    def get_embedder(self,name):
        embedder_name = "embedder_{}".format(name)
        return self.__getattr__(embedder_name)

    def embed(self,features_s,ref=None):
        '''

        :param features_s: it should be a dict!
        :return:
        '''
        embeded_features = {}
        for fname,feature in features_s.items():
            embedder = self.get_embedder(fname)
            fconfig = {"dim": self.n_embed_dim//self.n_field, "atten": False}
            if fconfig["atten"]:
                assert ref is not None
                embeded_feature = embedder(input=feature[0], offsets=feature[1], ref=ref)
            else:
                embeded_feature = embedder(input=feature[0], offsets=feature[1])
            B,L = embeded_feature.size()
            embeded_features[fname] = (embeded_feature.view(-1,L,1)@feature[2].view(-1,1,1)).view(-1,L).contiguous()
        return embeded_features
    def forward(self, samples):
        # features in sparse representation (feature values, offsets for each sample, ...), see Embedding class in pytorch.
        embedding_features_s = samples["embedding_features"]
#         print(embedding_features_s)
        embedding_features_d = self.embed(embedding_features_s, None)

        embedding_features = torch.cat(list(embedding_features_d.values()), dim=1)

        B, L = embedding_features.size()

        z = embedding_features

        # FFM
        r = self.fm(z.view(B, self.n_feat*self.n_field, self.n_embed_dim//self.n_field))


        x = torch.cat([embedding_features,r],dim=1)
        x = self.bn0(x)
        d = self.linear1(x)
        d = self.dice1(d)

        d = self.linear2(d)
        d = self.dice2(d)

        s = self.linear3(d)
        s = s.view(-1)

        target = samples["labels"]
        target_weight = samples["label_weights"]
        if target is None:
            return None, s, None, None, d
        l = self.loss(s,target,target_weight,size_average=False)


        # average loss manually
        model_loss = l / samples["size"]

        final_loss = model_loss

        return final_loss, s, model_loss, 0, d

    def get_train_policy(self):
        # params_group1 = []
        # params_group2 = []
        # for name,params in self.named_parameters():
        #     if name.find("embedder")!=-1:
        #         params_group1.append(params)
        #         print("{} are in group 1 trained with lr x0.1".format(name))
        #     else:
        #         params_group2.append(params)
        #         print("{} are in group 1 trained with lr x1".format(name))
        #
        #
        #
        # params = [
        #     {'params': params_group1, "lr_mult": 0.1, "decay_mult": 1},
        #     {'params': params_group2, "lr_mult": 1, "decay_mult": 1},
        # ]
        params = [{'params': self.parameters(),"lr_mult": 1, "decay_mult": 1},]
        return params
