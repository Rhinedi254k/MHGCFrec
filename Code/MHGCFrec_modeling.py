import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
device = torch.device("cpu")


class HGCAN(nn.Module):
    # Hypergraph Graph Convolution Attention Network (HGCAN)
    def __init__(self, embed_size, attention_size):
        super(HGCAN, self).__init__()
        self.embed_size = embed_size
        self.attention_size = attention_size
        self.attention_layer = nn.Linear(embed_size, attention_size)
        self.graph_conv_layer = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # HGCAN operations, involving attention mechanisms and graph convolutions
        batch_size, seq_len, embed_dim = x.size()
        attention_scores = F.softmax(self.attention_layer(x), dim=-1)
        output = torch.bmm(attention_scores.transpose(1, 2), x)
        output = self.graph_conv_layer(output)
        return output.squeeze(1)
#-----------------------------------------------------------------------------------------------------------------------
class aVGAEAN(nn.Module):

    def __init__(self, embed_size, latent_dim):
        super(aVGAEAN, self).__init__()

        # Generator part - adaptive VGAEAN
        self.encoder = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.SELU(),
            nn.Linear(embed_size, latent_dim * 2)  # Latent space (mu, logvar)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embed_size),
            nn.SELU(),
            nn.Linear(embed_size, embed_size)
        )

        self.hgcan = HGCAN(embed_size, attention_size=embed_size)

        # Discriminator part - based on the encoder-decoder structure
        self.discriminator = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.SELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random noise
        return mu + eps * std

    def Q(self, x):
        # Encoder: generate latent variables (mu, logvar)
        encoded = self.encoder(x)
        mu, logvar = encoded[:, :encoded.shape[1] // 2], encoded[:, encoded.shape[1] // 2:]
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)

        # HGCAN
        hgcan_output = self.hgcan(reconstruction.unsqueeze(1))
        disc_output = self.discriminator(hgcan_output)
        return reconstruction, disc_output, mu, logvar

    def sample_z(self, mu, logvar):
        return self.reparameterize(mu, logvar)

    def P(self, z):
        reconstruction = self.decoder(z)
        return reconstruction

    def adversarial_loss(self, z_real, z_fake):
        real_loss = torch.mean(torch.log(self.discriminator(z_real) + 1e-8))
        fake_loss = torch.mean(torch.log(1 - self.discriminator(z_fake) + 1e-8))
        return -(real_loss + fake_loss)
#-----------------------------------------------------------------------------------------------------------------------

class GKNN(nn.Module):
    def __init__(self, embed_size):
        super(GKNN, self).__init__()
        self.conv_layer = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)  # Convert to [batch, channels, length]
        return F.relu(self.conv_layer(x)).transpose(1, 2)
class DualGNN(torch.nn.Module):
    def __init__(self, user_size, poi_size, gender_size, age_size, occupation_size, category_size, landmark_size,
                 facility_size, rating_size, location_size, embed_size, attention_size, dropout):
        super(DualGNN, self).__init__()
        self.user_size = user_size
        self.poi_size = poi_size
        self.gender_size = gender_size
        self.age_size = age_size
        self.occupation_size = occupation_size
        self.category_size = category_size
        self.landmark_size = landmark_size
        self.facility_size = facility_size
        self.rating_size = rating_size
        self.location_size = location_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.attention_size = attention_size

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)

        self.user_embed = torch.nn.Embedding(self.user_size, self.embed_size)
        self.poi_embed = torch.nn.Embedding(self.poi_size, self.embed_size)
        nn.init.xavier_uniform(self.user_embed.weight)
        nn.init.xavier_uniform(self.poi_embed.weight)

        self.user_bias = torch.nn.Embedding(self.user_size, 1)
        self.poi_bias = torch.nn.Embedding(self.poi_size, 1)
        nn.init.constant(self.user_bias.weight, 0)
        nn.init.constant(self.poi_bias.weight, 0)

        self.miu = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        self.gender_embed = torch.nn.Embedding(self.gender_size, self.embed_size)
        self.gender_embed.weight.data.normal_(0, 0.05)
        self.age_embed = torch.nn.Embedding(self.age_size, self.embed_size)
        self.age_embed.weight.data.normal_(0, 0.05)
        self.occupation_embed = torch.nn.Embedding(self.occupation_size, self.embed_size)
        self.occupation_embed.weight.data.normal_(0, 0.05)

        self.category_embed = torch.nn.Embedding(self.category_size, self.embed_size)
        self.category_embed.weight.data.normal_(0, 0.05)
        self.landmark_embed = torch.nn.Embedding(self.landmark_size, self.embed_size)
        self.landmark_embed.weight.data.normal_(0, 0.05)
        self.facility_embed = torch.nn.Embedding(self.facility_size, self.embed_size)
        self.facility_embed.weight.data.normal_(0, 0.05)
        self.rating_embed = torch.nn.Embedding(self.rating_size, self.embed_size)
        self.rating_embed.weight.data.normal_(0, 0.05)
        self.location_embed = torch.nn.Embedding(self.location_size, self.embed_size)
        self.location_embed.weight.data.normal_(0, 0.05)


        #--------------------------------------------------
        self.dense_poi_self_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_poi_self_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_poi_onehop_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_poi_onehop_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_self_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_self_siinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_onehop_biinter = nn.Linear(self.embed_size, self.embed_size)
        self.dense_user_onehop_siinter = nn.Linear(self.embed_size, self.embed_size)
        init_weights(self.dense_poi_self_biinter)
        init_weights(self.dense_poi_self_siinter)
        init_weights(self.dense_poi_onehop_biinter)
        init_weights(self.dense_poi_onehop_siinter)
        init_weights(self.dense_user_self_biinter)
        init_weights(self.dense_user_self_siinter)
        init_weights(self.dense_user_onehop_biinter)
        init_weights(self.dense_user_onehop_siinter)

        self.dense_poi_cate_self = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_poi_cate_hop1 = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_user_cate_self = nn.Linear(2 * self.embed_size, self.embed_size)
        self.dense_user_cate_hop1 = nn.Linear(2 * self.embed_size, self.embed_size)
        init_weights(self.dense_poi_cate_self)
        init_weights(self.dense_poi_cate_hop1)
        init_weights(self.dense_user_cate_self)
        init_weights(self.dense_user_cate_hop1)

        self.dense_poi_gknn = GKNN(self.embed_size * 2)
        init_weights(self.dense_poi_gknn)
        self.dense_poi_gru = nn.GRU(self.embed_size * 2, self.embed_size)
        init_weights(self.dense_poi_gru)
        self.dense_user_gknn = GKNN(self.embed_size * 2)
        init_weights(self.dense_user_gknn)
        self.dense_user_gru = nn.GRU(self.embed_size * 2, self.embed_size)


        latent_dim = 30  #
        self.user_vgaean = aVGAEAN(embed_size, latent_dim)
        self.poi_vgaean = aVGAEAN(embed_size, latent_dim)

        # -------------------------------------------------concat
        #GKNN+GRU
        self.FC_pre = nn.Linear(2 * embed_size, 1)
        init_weights(self.FC_pre)

        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(p=0.2)

    def user_trust_score(self, emb_i, emb_j):
        # Compute cosine similarity between embeddings
        trust_score = torch.sum(emb_i * emb_j, dim=-1) / (torch.norm(emb_i, dim=-1) * torch.norm(emb_j, dim=-1) + 1e-8)
        return trust_score

    def feat_interaction(self, feature_embedding, fun_bi, fun_si, dimension):
        summed_features_emb_square = (torch.sum(feature_embedding, dim=dimension)).pow(2)
        squared_sum_features_emb = torch.sum(feature_embedding.pow(2), dim=dimension)
        deep_fm = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
        deep_fm = self.selu(fun_bi(deep_fm))
        bias_fm = self.selu(fun_si(feature_embedding.sum(dim=dimension)))
        nfm = deep_fm + bias_fm
        return nfm

    def forward(self, user, poi, user_self_cate, user_onehop_id, user_onehop_cate, poi_self_cate, poi_self_landmark,
                poi_self_facility, poi_self_rating, poi_self_location, poi_onehop_id, poi_onehop_cate,
                poi_onehop_landmark,
                poi_onehop_facility, poi_onehop_rating, poi_onehop_location, mode='training'):


        uids_list = user.to(device)
        sids_list = poi.to(device)
        # Initialize embeddings for users and POIs
        user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
        poi_embedding = self.poi_embed(torch.autograd.Variable(sids_list))

        # Assign user and POI embeddings based on mode
        if mode == 'training' or mode == 'Warm':  # Warm-start
            user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
            poi_embedding = self.poi_embed(torch.autograd.Variable(sids_list))
        if mode == 'Pcold': # POI cold-start
            user_embedding = self.user_embed(torch.autograd.Variable(uids_list))
        if mode == 'Ucold':  # User Cold-start
            poi_embedding = self.poi_embed(torch.autograd.Variable(sids_list))

        batch_size = poi_self_cate.shape[0]
        cate_size = poi_self_cate.shape[1]
        landmark_size = poi_self_landmark.shape[1]
        facility_size = poi_self_facility.shape[1]
        rating_size = poi_self_rating.shape[1]
        location_size = poi_self_location.shape[1]
        user_onehop_size = user_onehop_id.shape[1]
        poi_onehop_size = poi_onehop_id.shape[1]

        #------------------------------------------DualGNN-poi
        # N=2
        poi_onehop_id = self.poi_embed(Variable(poi_onehop_id))

        poi_onehop_cate = self.category_embed(Variable(poi_onehop_cate).view(-1, cate_size)).view(batch_size,poi_onehop_size,cate_size, -1)
        poi_onehop_landmark = self.landmark_embed(Variable(poi_onehop_landmark).view(-1, landmark_size)).view(batch_size, poi_onehop_size, landmark_size, -1)
        poi_onehop_facility = self.facility_embed(Variable(poi_onehop_facility).view(-1, facility_size)).view(batch_size, poi_onehop_size, facility_size, -1)
        poi_onehop_rating = self.rating_embed(Variable(poi_onehop_rating).view(-1, rating_size)).view(batch_size, poi_onehop_size, rating_size, -1)
        poi_onehop_location = self.location_embed(Variable(poi_onehop_location).view(-1, location_size)).view(batch_size, poi_onehop_size, location_size, -1)

        poi_onehop_feature = torch.cat([poi_onehop_cate, poi_onehop_landmark, poi_onehop_facility, poi_onehop_rating, poi_onehop_location], dim=2)
        poi_onehop_embed = self.dense_poi_cate_hop1(torch.cat([self.feat_interaction(poi_onehop_feature, self.dense_poi_onehop_biinter,  self.dense_poi_onehop_siinter, dimension=2), poi_onehop_id], dim=-1))

        # N=1
        poi_self_cate = self.category_embed(Variable(poi_self_cate))
        poi_self_landmark = self.landmark_embed(Variable(poi_self_landmark))
        poi_self_facility = self.facility_embed(Variable(poi_self_facility))
        poi_self_rating = self.rating_embed(Variable(poi_self_rating))
        poi_self_location = self.location_embed(Variable(poi_self_location))

        poi_self_feature = torch.cat(
            [poi_self_cate, poi_self_landmark, poi_self_facility, poi_self_rating, poi_self_location], dim=1)
        poi_self_feature = self.feat_interaction(poi_self_feature, self.dense_poi_self_biinter,
                                                  self.dense_poi_self_siinter, dimension=1)

        if mode == 'Pcold':
            _, _, poi_mu, poi_var = self.poi_vgaean.Q(poi_self_feature)
            poi_z = self.poi_vgaean.sample_z(poi_mu, poi_var)
            poi_embedding = self.poi_vgaean.P(poi_z)
        poi_self_embed = self.dense_poi_cate_self(torch.cat([poi_self_feature, poi_embedding], dim=-1))

        poi_gknn = self.sigmoid(self.dense_poi_gknn(torch.cat([poi_self_embed.unsqueeze(1).repeat(1, poi_onehop_size, 1), poi_onehop_embed], dim=-1)))
        poi_gknn = poi_gknn[:, :, :self.embed_size]
        gru_input = torch.cat([poi_self_embed, poi_onehop_embed.mean(dim=1)], dim=-1).unsqueeze(0)
        gru_output, _ = self.dense_poi_gru(gru_input)
        poi_gru = self.sigmoid(gru_output.squeeze(0))

        poi_onehop_embed_final = (poi_onehop_embed * poi_gknn).mean(1)
        poi_self_embed = (1 - poi_gru) * poi_self_embed

        poi_gcn_embed = self.selu(poi_self_embed + poi_onehop_embed_final)  # [batch, embed]

        #----------------------------------------------DualGNN-user
        # N=2
        user_onehop_id = self.user_embed(Variable(user_onehop_id))

        user_onehop_gender_emb = self.gender_embed(Variable(user_onehop_cate[:, :, 0]))
        user_onehop_age_emb = self.age_embed(Variable(user_onehop_cate[:, :, 1]))
        user_onehop_occupation_emb = self.occupation_embed(Variable(user_onehop_cate[:, :, 2]))

        user_onehop_feat = torch.cat([user_onehop_gender_emb.unsqueeze(2), user_onehop_age_emb.unsqueeze(2),
                                      user_onehop_occupation_emb.unsqueeze(2)], dim=2)
        user_onehop_embed = self.dense_user_cate_hop1(torch.cat([self.feat_interaction(user_onehop_feat, self.dense_user_onehop_biinter, self.dense_user_onehop_siinter, dimension=2), user_onehop_id], dim=-1))

        # N=1
        user_gender_emb = self.gender_embed(Variable(user_self_cate[:, 0]))
        user_age_emb = self.age_embed(Variable(user_self_cate[:, 1]))
        user_occupation_emb = self.occupation_embed(Variable(user_self_cate[:, 2]))

        user_self_feature = torch.cat([user_gender_emb.unsqueeze(1), user_age_emb.unsqueeze(1), user_occupation_emb.unsqueeze(1)], dim=1)
        user_self_feature = self.feat_interaction(user_self_feature, self.dense_user_self_biinter,  self.dense_user_onehop_siinter, dimension=1)

        if mode == 'Ucold':
            _, _, user_mu, user_var = self.poi_vgaean.Q(poi_self_feature)
            user_z = self.user_vgaean.sample_z(user_mu, user_var)
            user_embedding = self.user_vgaean.P(user_z)
        user_self_embed = self.dense_user_cate_self(torch.cat([user_self_feature, user_embedding], dim=-1))

        user_gknn = self.sigmoid(self.dense_user_gknn(torch.cat([user_self_embed.unsqueeze(1).repeat(1, user_onehop_size, 1), user_onehop_embed],dim=-1)))
        user_gknn = user_gknn[:, :, :self.embed_size]
        gru_input_u = torch.cat([user_self_embed, user_onehop_embed.mean(dim=1)], dim=-1).unsqueeze(0)
        gru_output_u, _ = self.dense_user_gru(gru_input_u)
        user_gru = self.sigmoid(gru_output_u.squeeze(0))
        user_onehop_embed_final = (user_onehop_embed * user_gknn).mean(dim=1)
        user_self_embed = (1 - user_gru) * user_self_embed

        user_gcn_embed = self.selu(user_self_embed + user_onehop_embed_final)

        #--------------------------------------------------norm
        _, _, poi_mu, poi_var = self.poi_vgaean.Q(poi_self_feature)
        poi_z = self.poi_vgaean.sample_z(poi_mu, poi_var)
        poi_preference_sample = self.poi_vgaean.P(poi_z)

        _, _, user_mu, user_var = self.poi_vgaean.Q(poi_self_feature)
        user_z = self.user_vgaean.sample_z(user_mu, user_var)
        user_preference_sample = self.user_vgaean.P(user_z)

        recon_loss = torch.norm(poi_preference_sample - poi_embedding) + torch.norm(user_preference_sample - user_embedding)
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(poi_z) + poi_mu ** 2 - 1. - poi_var, 1)) + \
                  torch.mean(0.5 * torch.sum(torch.exp(user_z) + user_mu ** 2 - 1. - user_var, 1))
        adv_loss = self.poi_vgaean.adversarial_loss(user_z, poi_z)

        # --------------------------------------------Recommendation
        bu = self.user_bias(Variable(uids_list))
        bi = self.poi_bias(Variable(sids_list))
        tmp = torch.cat([user_gcn_embed, poi_gcn_embed], dim=1)
        trust_scores = self.user_trust_score(user_gcn_embed, poi_gcn_embed)
        pred = self.FC_pre(tmp) + (user_gcn_embed * poi_gcn_embed).sum(1, keepdim=True) + bu + bi + (self.miu).repeat(batch_size, 1)
        pred += trust_scores.unsqueeze(1)

        return pred.squeeze(), recon_loss, kl_loss, adv_loss