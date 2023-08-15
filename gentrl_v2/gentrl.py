import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from math import pi, log
from gentrl_v2.lp import LP
from gentrl_v2.qnetwork import QNetwork
from gentrl_v2.replaybuffer import ReplayBuffer
import pickle
from pathlib import Path
from tqdm import tqdm
import csv
import gc
from rdkit import Chem, DataStructs
import numpy as np
from itertools import cycle, islice
from rdkit.Chem import AllChem, Descriptors, QED, RDConfig, Draw
from moses.metrics import mol_passes_filters, QED, SA, logP
from sklearn.neighbors import NearestNeighbors
import math
import random
from statistics import mean
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from collections import defaultdict

# Copied this from moses
def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except:
            return None
        return mol
    return smiles_or_mol


class TrainStats():
    def __init__(self):
        self.stats = dict()

    def update(self, delta):
        for key in delta.keys():
            if key in self.stats.keys():
                self.stats[key].append(delta[key])
            else:
                self.stats[key] = [delta[key]]

    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = []

    def print(self):
        for key in self.stats.keys():
            print(str(key) + ": {:4.4};".format(
                sum(self.stats[key]) / len(self.stats[key])
            ), end='')

        print()


class GENTRL(nn.Module):
    '''
    GENTRL model
    '''
    def __init__(self, enc, dec, latent_descr, feature_descr, state_dim, action_dim, tt_int=40,
                 tt_type='usual', beta=0.01, gamma=0.1, device="cuda:0", vae='vae'):
        super(GENTRL, self).__init__()

        self.enc = enc
        self.dec = dec

        self.num_latent = len(latent_descr)
        self.num_features = len(feature_descr)

        self.latent_descr = latent_descr
        self.feature_descr = feature_descr

        self.tt_int = tt_int
        self.tt_type = tt_type
        
        # custom
        self.vae = vae

        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type)

        self.beta = beta
        self.gamma = gamma

        self.device = device


    def get_elbo(self, x, y):
        means, log_stds = torch.split(self.enc.encode(x),
                              len(self.latent_descr), dim=1)
        latvar_samples = (means + torch.randn_like(log_stds) *
                          torch.exp(0.5 * log_stds))

        # x is input smiles
        # y is chemical properties
        # z is representation in the latent space

        # Presumably log p(x|z)
        rec_part = self.dec.weighted_forward(x, latvar_samples).mean()

        normal_distr_hentropies = (log(2 * pi) + 1 + log_stds).sum(dim=1)

        latent_dim = len(self.latent_descr)
        condition_dim = len(self.feature_descr)

        # p(z, y)
        zy = torch.cat([latvar_samples, y], dim=1)
        log_p_zy = self.lp.log_prob(zy)
        
        y_to_marg = latent_dim * [True] + condition_dim * [False]
        log_p_y = self.lp.log_prob(zy, marg=y_to_marg)

        z_to_marg = latent_dim * [False] + condition_dim * [True]
        log_p_z = self.lp.log_prob(zy, marg=z_to_marg)

        
        # log p(z|y)
        log_p_z_by_y = log_p_zy - log_p_y
        
        # log p(y|z)
        log_p_y_by_z = log_p_zy - log_p_z

        if self.vae == 'vae':
            kldiv_part = (-normal_distr_hentropies - log_p_zy).mean()
        elif self.vae == 'cvae':
            #conditional prior log_p_z_by_y
            log_p_z_by_y = log_p_z_by_y.unsqueeze(-1)      
            kldiv_part = torch.mean(means**2 + torch.exp(log_stds) - 1 - log_stds - log_p_z_by_y, [0, 1])

        elbo = rec_part - self.beta * kldiv_part + self.gamma * log_p_y_by_z.mean()

        return elbo, {
            'loss': -elbo.detach().cpu().numpy(),
            'rec': rec_part.detach().cpu().numpy(),
            'kl': kldiv_part.detach().cpu().numpy(),
            'log_p_y_by_z': log_p_y_by_z.mean().detach().cpu().numpy(),
            'log_p_z_by_y': log_p_z_by_y.mean().detach().cpu().numpy(),
            'normal_distr_hentropies' : normal_distr_hentropies.mean()
        }

    def save(self, folder_to_save='./'):
        if folder_to_save[-1] != '/':
            folder_to_save = folder_to_save + '/'
        torch.save(self.enc.state_dict(), folder_to_save + 'enc.model')
        torch.save(self.dec.state_dict(), folder_to_save + 'dec.model')
        torch.save(self.lp.state_dict(), folder_to_save + 'lp.model')

        pickle.dump(self.lp.order, open(folder_to_save + 'order.pkl', 'wb'))

    def load(self, folder_to_load='./'):
        if folder_to_load[-1] != '/':
            folder_to_load = folder_to_load + '/'

        order = pickle.load(open(folder_to_load + 'order.pkl', 'rb'))
        self.lp = LP(distr_descr=self.latent_descr + self.feature_descr,
                     tt_int=self.tt_int, tt_type=self.tt_type,
                     order=order)

        self.enc.load_state_dict(torch.load(folder_to_load + 'enc.model'))
        self.dec.load_state_dict(torch.load(folder_to_load + 'dec.model'))
        self.lp.load_state_dict(torch.load(folder_to_load + 'lp.model'))

    def train_as_vaelp(self, train_loader, train_name, val_loader=None, num_epochs=None, patience=None, lr=1e-3,
                        reinit_epochs=[1, 2, 5, 25, 50], start_epoch=0):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Variables for early stopping
        keep_training = True
        dissapointment = 0
        lowest_loss = 99999999999
 
        epoch_count = start_epoch
        
        to_reinit = False
        buf = None
                
        while keep_training:
            i = 0
            epoch_count += 1

            if epoch_count in reinit_epochs:
                to_reinit = True
            else:
                to_reinit = False

            # Training
            self.enc.train()
            self.dec.train()

            train_elbo_mean = 0
            train_rec_mean = 0
            train_kl_mean = 0
            train_log_p_y_by_z_mean = 0
            train_normal_dist_hent = 0

            for x_batch, y_batch in tqdm(train_loader):
                i += 1

                if i % 100 == 0:
                    gc.collect()


                y_batch = y_batch.float().to(self.lp.tt_cores[0].device)
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.view(-1, 1).contiguous()

                if to_reinit:
                    if (buf is None) or (buf.shape[0] < 5000):
                        enc_out = self.enc.encode(x_batch)
                        means, log_stds = torch.split(enc_out,
                                                      len(self.latent_descr),
                                                      dim=1)
                        z_batch = (means + torch.randn_like(log_stds) *
                                   torch.exp(0.5 * log_stds))
                        cur_batch = torch.cat([z_batch, y_batch], dim=1)
                        if buf is None:
                            buf = cur_batch
                        else:
                            buf = torch.cat([buf, cur_batch])
                    else:
                        descr = len(self.latent_descr) * [0] + len(self.feature_descr) * [1]

                        self.lp.reinit_from_data(buf, descr)
                        self.lp.cuda()
                        buf = None
                        to_reinit = False

                    continue

                elbo, cur_stats = self.get_elbo(x_batch, y_batch)
                optimizer.zero_grad()
                loss = -elbo
                
                train_elbo_mean += -elbo
                train_rec_mean += cur_stats['rec']
                train_kl_mean += cur_stats['kl']
                train_log_p_y_by_z_mean += cur_stats['log_p_y_by_z']
                train_normal_dist_hent += cur_stats['normal_distr_hentropies']

                loss.backward()
                
                #custom
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                
                optimizer.step()
                del x_batch, y_batch, elbo, cur_stats, loss
                torch.cuda.empty_cache()

            if val_loader:
                val_elbo_mean = 0
                val_rec_mean = 0
                val_kl_mean = 0
                val_log_p_y_by_z_mean = 0
                val_normal_dist_hent = 0
                
                # Validation
                self.enc.eval()
                self.dec.eval()
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        y_batch = y_batch.float().to(self.lp.tt_cores[0].device)
                        if len(y_batch.shape) == 1:
                            y_batch = y_batch.view(-1, 1).contiguous()
                        elbo, cur_stats = self.get_elbo(x_batch, y_batch)
                        
                        val_elbo_mean += -elbo
                        val_rec_mean += cur_stats['rec']
                        val_kl_mean += cur_stats['kl']
                        val_log_p_y_by_z_mean += cur_stats['log_p_y_by_z']
                        val_normal_dist_hent += cur_stats['normal_distr_hentropies']

            # Stopping criterion
            if num_epochs:
                if epoch_count == num_epochs:
                    keep_training = False
            
            if patience:
                if val_elbo_mean < lowest_loss:
                    lowest_loss = val_elbo_mean
                    dissapointment = 0
                else:
                    dissapointment += 1

                if dissapointment == patience:
                    keep_training = False

            if epoch_count % 50 == 0:
                p = Path("models") / train_name / str(epoch_count)
                p.mkdir(parents=True, exist_ok=True)
                self.save("./" + str(p) + "/")

            
            print("Epoch: " + str(epoch_count))
            print('Train ELBO: ' + str(train_elbo_mean / len(train_loader)))
            print("Train KL component: " + str(train_kl_mean / len(train_loader)))
            print("Train rec component: " + str(train_rec_mean / len(train_loader)))
            
            if val_loader:
                print("Val KL component: " + str(val_kl_mean / len(val_loader)))
                print("Val rec component: " + str(val_rec_mean / len(val_loader)))
                
            print('Entropies: ' + str(train_normal_dist_hent / len(train_loader)))
            print('p(z|y): ' + str(train_log_p_y_by_z_mean / len(train_loader)))
       
    # Define RL agent 
    def train_as_rl(self,
                    reward_fn,
                    train_name,
                    num_iterations=2000,
                    start_iteration=1,
                    batch_size=200,
                    lr_lp=1e-5, lr_dec=1e-6,
                    explore_perc=0.3):
        # We optimize the parameters of the learned prior
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        # We optimize the parameters of the fc layer from the decoder that interacts with the latent space directly
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)

        p = Path("models") / train_name / "RL"
        p.mkdir(parents=True, exist_ok=True)
        f = open(str(Path("models") / train_name / "RL" / (train_name + ".csv")), 'w')
        csv_writer = csv.writer(f)
        header = ["Iteration", "SMILES Explore", "SMILES Exploit"]
        csv_writer.writerow(header)
        

        for iter_count in tqdm(range(start_iteration, num_iterations + 1)):
            # Get exploitation latens
            exploit_size = int(batch_size * (1 - explore_perc))
            exploit_z = self.lp.sample(exploit_size, self.num_latent * ['s'] + self.num_features * ['m'])
            
            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            # Exploration latents are based on the mean and std of the exploit latens
            expl_size = int(batch_size * explore_perc)
            expl_z = torch.randn(expl_size, exploit_z.shape[1], device=self.device)
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            # Decode the latents for both exploitation and exploration
            z = torch.cat([exploit_z, expl_z]).to("cuda:0")
            smiles = self.dec.sample(z, 50, argmax=False)
            
            # Get the log prob of these smiles by summing their prob of the latents and the smiles given the latents
            zc = torch.zeros(z.shape[0], self.num_features, device=self.device)
            conc_zy = torch.cat([z, zc], dim=1).to("cuda:0")
            log_probs = self.lp.log_prob(conc_zy, marg=self.num_latent * [False] + self.num_features * [True])
            log_probs += self.dec.weighted_forward(smiles, z)
            
            # Reward: physicochemical properties
            r_list = [reward_fn(s) for s in smiles]
            rewards = torch.tensor(r_list).float().to(exploit_z.device)

            # Substract baseline
            rewards_bl = rewards - rewards.mean()
            
            # Compute the loss as the rewards multiplied by the log probs of getting those rewards
            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()

            valid_sm = [s for s in smiles if get_mol(s) is not None]


            if iter_count % 500 == 0:
                p = Path("models") / train_name / "RL" / str(iter_count)
                p.mkdir(parents=True, exist_ok=True)
                self.save("./" + str(p) + "/")
                print('Mean reward: ' + str(sum(r_list) / len(smiles)))
                print('Valid perc: ' + str(len(valid_sm) / len(smiles)))
                print('Unique perc: ' + str(0 if len(valid_sm) == 0 else len(set(valid_sm)) / len(valid_sm)))

            if iter_count % 100 == 0 or iter_count == 1:
                csv_writer.writerow([iter_count, [s for s in smiles[:expl_size] if get_mol(s) is not None], [s for s in smiles[expl_size:] if get_mol(s) is not None]])
            iter_count += 1
            
        f.close()
        return None

    def sample(self, num_samples, max_len=50):
        z = self.lp.sample(num_samples, self.num_latent * ['s'] + self.num_features * ['m'])
        smiles = self.dec.sample(z, max_len, argmax=False)

        return smiles
    
    def smiles_to_latent(self, smiles_list):
        """
        Function to convert list of molecules to their latent representations.
        """
        with torch.no_grad():
            means, log_stds = torch.split(self.enc.encode(smiles_list), self.state_dim, dim=1)
            latvar_samples = (means + torch.randn_like(log_stds) * torch.exp(0.5 * log_stds)).detach()

        return latvar_samples
    
    # Define IL1 agent
    def train_as_il(self,
                    expert_smiles,
                    num_iterations=2000,
                    batch_size=128,
                    compare_batch_size=128,
                    lr_lp=1e-5, lr_dec=1e-6,
                    initial_explore_perc=0.9, final_explore_perc=0.3, 
                    start_iter=1):
        # We optimize the parameters of the learned prior
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        # We optimize the parameters of the fc layer from the decoder that interacts with the latent space directly
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)
            
        fpgen = AllChem.GetRDKitFPGenerator()
        
        p = Path("models") / train_name / 'IL'
        p.mkdir(parents=True, exist_ok=True)
        f = open(str(Path("models") / train_name / 'IL' / (train_name + ".csv")), 'w')
        csv_writer = csv.writer(f)
        header = ["Iteration", "SMILES Explore", "SMILES Exploit"]
        csv_writer.writerow(header)

        explore_perc = initial_explore_perc
        num_decay_iterations = 3/4 * num_iterations
        decay_rate = (initial_explore_perc - final_explore_perc) / num_decay_iterations

        for iter_count in tqdm(range(start_iter, num_iterations + 1)):
            # Get exploitation latens
            exploit_size = int(batch_size * (1 - explore_perc))
            exploit_z = self.lp.sample(exploit_size, self.num_latent * ['s'] + self.num_features * ['m'])

            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            # Exploration latents are based on the mean and std of the exploit latens
            expl_size = int(batch_size * explore_perc)
            expl_z = torch.randn(expl_size, exploit_z.shape[1], device=self.device)
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            # Decode the latents for both exploitation and exploration
            z = torch.cat([exploit_z, expl_z]).to("cuda:0")
            smiles = self.dec.sample(z, 50, argmax=False)
            
            # Get the log prob of these smiles by summing their prob of the latents and the smiles given the latents
            zc = torch.zeros(z.shape[0], self.num_features, device=self.device)
            conc_zy = torch.cat([z, zc], dim=1).to("cuda:0")
            log_probs = self.lp.log_prob(conc_zy, marg=self.num_latent * [False] + self.num_features * [True])
            log_probs += self.dec.weighted_forward(smiles, z)

            # IL - Tanimoto similarity
            # get batch of experts for this iter
            batch_expert_smiles = random.sample(expert_smiles, compare_batch_size)
            batch_expert_mols = [get_mol(s) for s in batch_expert_smiles]
            batch_expert_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in batch_expert_mols]
            # get generated sample
            generated_mols = [get_mol(s) for s in smiles]
            generated_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) if mol is not None else None for mol in generated_mols] 

            # Reward: Tanimoto similarity
            tanimoto_sims = []
            for g_fp in generated_fps:
                if g_fp is not None:
                    best_similarity = 0 
                    for e_fp in batch_expert_fps:
                        similarity = DataStructs.TanimotoSimilarity(g_fp, e_fp)
                        best_similarity = max(best_similarity, similarity) 
                    tanimoto_sims.append(best_similarity)
                else:
                    tanimoto_sims.append(0)
                            
            rewards = torch.tensor(tanimoto_sims).float().to(exploit_z.device)
            r_list = tanimoto_sims

            rewards_bl = rewards - rewards.mean()

            if iter_count <= num_decay_iterations:
                explore_perc -= decay_rate
                explore_perc = max(explore_perc, final_explore_perc)

            # Compute the loss as the rewards multiplied by the log probs of getting those rewards
            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()

            valid_sm = [s for s in smiles if get_mol(s) is not None]


            if iter_count % 500 == 0:
                p = Path("models") / train_name / 'IL' / str(iter_count)
                p.mkdir(parents=True, exist_ok=True)
                self.save("./" + str(p) + "/")
                print('Mean reward: ' + str(sum(r_list) / len(smiles)))
                print('Valid perc: ' + str(len(valid_sm) / len(smiles)))
                print('Unique perc: ' + str(0 if len(valid_sm) == 0 else len(set(valid_sm)) / len(valid_sm)))

                

            if iter_count % 500 == 0 or iter_count == 1:
                csv_writer.writerow([iter_count, [s for s in smiles[:expl_size] if get_mol(s) is not None], [s for s in smiles[expl_size:] if get_mol(s) is not None]])
            iter_count += 1
            
        f.close()
        return None

    #Define IL2 agent
    def train_as_il2(self,
                    expert_smiles,
                    train_name,
                    num_iterations=2000,
                    batch_size=128,
                    compare_batch_size=128,
                    lr_lp=1e-5, lr_dec=1e-6,
                    initial_explore_perc=0.9, final_explore_perc=0.3, 
                    start_iter=1):
        # We optimize the parameters of the learned prior
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        # We optimize the parameters of the fc layer from the decoder that interacts with the latent space directly
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)
            
        fpgen = AllChem.GetRDKitFPGenerator()
        
        p = Path("models") / train_name / 'IL2'
        p.mkdir(parents=True, exist_ok=True)
        f = open(str(Path("models") / train_name / 'IL2' / (train_name + ".csv")), 'w')
        csv_writer = csv.writer(f)
        header = ["Iteration", "SMILES Explore", "SMILES Exploit"]
        csv_writer.writerow(header)

        explore_perc = initial_explore_perc
        num_decay_iterations = 3/4 * num_iterations
        decay_rate = (initial_explore_perc - final_explore_perc) / num_decay_iterations

        for iter_count in tqdm(range(start_iter, num_iterations + 1)):
            # Get exploitation latens
            exploit_size = int(batch_size * (1 - explore_perc))
            exploit_z = self.lp.sample(exploit_size, self.num_latent * ['s'] + self.num_features * ['m'])

            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            # Exploration latents are based on the mean and std of the exploit latens
            expl_size = int(batch_size * explore_perc)
            expl_z = torch.randn(expl_size, exploit_z.shape[1], device=self.device)
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            # Decode the latents for both exploitation and exploration
            z = torch.cat([exploit_z, expl_z]).to("cuda:0")
            smiles = self.dec.sample(z, 50, argmax=False)
            
            # Get the log prob of these smiles by summing their prob of the latents and the smiles given the latents
            zc = torch.zeros(z.shape[0], self.num_features, device=self.device)
            conc_zy = torch.cat([z, zc], dim=1).to("cuda:0")
            log_probs = self.lp.log_prob(conc_zy, marg=self.num_latent * [False] + self.num_features * [True])
            log_probs += self.dec.weighted_forward(smiles, z)

            # get batch of experts for this iter
            batch_expert_smiles = random.sample(expert_smiles, compare_batch_size)
            batch_expert_mols = [get_mol(s) for s in batch_expert_smiles]
            batch_expert_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in batch_expert_mols]
            # get generated sample
            generated_mols = [get_mol(s) for s in smiles]
            generated_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) if mol is not None else None for mol in generated_mols] 
            
            # Reward: Nearest neighbour Tanimoto similarity
            tanimoto_sims = []
            for idx, g_fp in enumerate(generated_fps):
                if g_fp is not None:
                    best_expert_similarity = 0
                    best_generated_similarity = 0
                    
                    for e_fp in batch_expert_fps:
                        similarity = DataStructs.TanimotoSimilarity(g_fp, e_fp)
                        best_expert_similarity = max(best_expert_similarity, similarity)
                    
                    for j, other_g_fp in enumerate(generated_fps):
                        if idx != j and other_g_fp is not None:
                            similarity = DataStructs.TanimotoSimilarity(g_fp, other_g_fp)
                            best_generated_similarity = max(best_generated_similarity, similarity)
                    
                    if best_generated_similarity > 0:
                        ratio = best_expert_similarity / best_generated_similarity
                    else:
                        ratio = best_expert_similarity
                    
                    tanimoto_sims.append(ratio)
                else:
                    tanimoto_sims.append(0)
            
            rewards = torch.tensor(tanimoto_sims).float().to(exploit_z.device)
            r_list = rewards
            
            rewards_bl = rewards - rewards.mean()


            if iter_count <= num_decay_iterations:
                explore_perc -= decay_rate
                explore_perc = max(explore_perc, final_explore_perc)

            # Compute the loss as the rewards multiplied by the log probs of getting those rewards
            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()

            valid_sm = [s for s in smiles if get_mol(s) is not None]


            if iter_count % 500 == 0:
                p = Path("models") / train_name / 'IL2' / str(iter_count)
                p.mkdir(parents=True, exist_ok=True)
                self.save("./" + str(p) + "/")
                print('Mean reward: ' + str(sum(r_list) / len(smiles)))
                print('Valid perc: ' + str(len(valid_sm) / len(smiles)))
                print('Unique perc: ' + str(0 if len(valid_sm) == 0 else len(set(valid_sm)) / len(valid_sm)))

                

            if iter_count % 500 == 0 or iter_count == 1:
                csv_writer.writerow([iter_count, [s for s in smiles[:expl_size] if get_mol(s) is not None], [s for s in smiles[expl_size:] if get_mol(s) is not None]])
            iter_count += 1
            
        f.close()
        return None
        

    # Define IL3 agent
    def train_as_il3(self,
                    expert_smiles,
                    train_name,
                    num_iterations=2000,
                    batch_size=128,
                    compare_batch_size=128,
                    lr_lp=1e-5, lr_dec=1e-6,
                    initial_explore_perc=0.9, final_explore_perc=0.3, 
                    start_iter=1):
        # We optimize the parameters of the learned prior
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        # We optimize the parameters of the fc layer from the decoder that interacts with the latent space directly
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)
                    
        fpgen = AllChem.GetRDKitFPGenerator()
        
        p = Path("models") / train_name / 'IL3'
        p.mkdir(parents=True, exist_ok=True)
        f = open(str(Path("models") / train_name / 'IL3' / (train_name + ".csv")), 'w')
        csv_writer = csv.writer(f)
        header = ["Iteration", "SMILES Explore", "SMILES Exploit"]
        csv_writer.writerow(header)

        explore_perc = initial_explore_perc
        num_decay_iterations = 3/4 * num_iterations
        decay_rate = (initial_explore_perc - final_explore_perc) / num_decay_iterations

        for iter_count in tqdm(range(start_iter, num_iterations + 1)):
            # Get exploitation latens
            exploit_size = int(batch_size * (1 - explore_perc))
            exploit_z = self.lp.sample(exploit_size, self.num_latent * ['s'] + self.num_features * ['m'])

            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            # Exploration latents are based on the mean and std of the exploit latens
            expl_size = int(batch_size * explore_perc)
            expl_z = torch.randn(expl_size, exploit_z.shape[1], device=self.device)
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            # Decode the latents for both exploitation and exploration
            z = torch.cat([exploit_z, expl_z]).to("cuda:0")
            smiles = self.dec.sample(z, 50, argmax=False)
            
            # Get the log prob of these smiles by summing their prob of the latents and the smiles given the latents
            zc = torch.zeros(z.shape[0], self.num_features, device=self.device)
            conc_zy = torch.cat([z, zc], dim=1).to("cuda:0")
            log_probs = self.lp.log_prob(conc_zy, marg=self.num_latent * [False] + self.num_features * [True])
            log_probs += self.dec.weighted_forward(smiles, z)

            # get batch of experts for this iter
            batch_expert_smiles = random.sample(expert_smiles, compare_batch_size)
            batch_expert_mols = [get_mol(s) for s in batch_expert_smiles]
            batch_expert_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in batch_expert_mols]
            # get generated sample
            generated_mols = [get_mol(s) for s in smiles]
            generated_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) if mol is not None else None for mol in generated_mols] 
            
            # Reward: Nearest neighbour Tanimoto similarity, penalty for similar molecules
            tanimoto_sims = []
            negative_reward = -0.5
            max_similarity = 0.6   

            for idx, g_fp in enumerate(generated_fps):
                if g_fp is not None:
                    best_expert_similarity = 0
                    best_generated_similarity = 0
                    n_generated_similar = 0
                    
                    for e_fp in batch_expert_fps:
                        similarity = DataStructs.TanimotoSimilarity(g_fp, e_fp)
                        best_expert_similarity = max(best_expert_similarity, similarity)
                    
                    for j, other_g_fp in enumerate(generated_fps):
                        if idx != j and other_g_fp is not None:
                            similarity = DataStructs.TanimotoSimilarity(g_fp, other_g_fp)
                            best_generated_similarity = max(best_generated_similarity, similarity)
                            if similarity > max_similarity:
                                n_generated_similar += 1

                    if best_generated_similarity > 0:
                        ratio = best_expert_similarity / best_generated_similarity
                    else:
                        ratio = best_expert_similarity
                    
                    reward = ratio + np.tanh(best_expert_similarity - best_generated_similarity) - 0.02 * n_generated_similar
                    tanimoto_sims.append(reward)
                else:
                    tanimoto_sims.append(negative_reward)

            rewards = torch.tensor(tanimoto_sims).float().to(exploit_z.device)

            r_list = rewards
            
            rewards_bl = rewards - rewards.mean()


            if iter_count <= num_decay_iterations:
                explore_perc -= decay_rate
                explore_perc = max(explore_perc, final_explore_perc)

            # Compute the loss as the rewards multiplied by the log probs of getting those rewards
            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()

            valid_sm = [s for s in smiles if get_mol(s) is not None]


            if iter_count % 500 == 0:
                p = Path("models") / train_name / 'IL3' / str(iter_count)
                p.mkdir(parents=True, exist_ok=True)
                self.save("./" + str(p) + "/")
                print('Mean reward: ' + str(sum(r_list) / len(smiles)))
                print('Valid perc: ' + str(len(valid_sm) / len(smiles)))
                print('Unique perc: ' + str(0 if len(valid_sm) == 0 else len(set(valid_sm)) / len(valid_sm)))

            if iter_count % 500 == 0 or iter_count == 1:
                csv_writer.writerow([iter_count, [s for s in smiles[:expl_size] if get_mol(s) is not None], [s for s in smiles[expl_size:] if get_mol(s) is not None]])
            iter_count += 1
            
        f.close()
        return None
        
    # Define IL4 agent
    def train_as_il4(self,
                    expert_smiles,
                    train_name,
                    num_iterations=2000,
                    batch_size=128,
                    compare_batch_size=128,
                    lr_lp=1e-5, lr_dec=1e-6,
                    initial_explore_perc=0.9, final_explore_perc=0.3, 
                    start_iter=1):
        # We optimize the parameters of the learned prior
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        # We optimize the parameters of the fc layer from the decoder that interacts with the latent space directly
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)
                    
        fpgen = AllChem.GetRDKitFPGenerator()
        
        p = Path("models") / train_name / 'IL4'
        p.mkdir(parents=True, exist_ok=True)
        f = open(str(Path("models") / train_name / 'IL4' / (train_name + ".csv")), 'w')
        csv_writer = csv.writer(f)
        header = ["Iteration", "SMILES Explore", "SMILES Exploit"]
        csv_writer.writerow(header)

        explore_perc = initial_explore_perc
        num_decay_iterations = 3/4 * num_iterations
        decay_rate = (initial_explore_perc - final_explore_perc) / num_decay_iterations

        for iter_count in tqdm(range(start_iter, num_iterations + 1)):
            # Get exploitation latens
            exploit_size = int(batch_size * (1 - explore_perc))
            exploit_z = self.lp.sample(exploit_size, self.num_latent * ['s'] + self.num_features * ['m'])

            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            # Exploration latents are based on the mean and std of the exploit latens
            expl_size = int(batch_size * explore_perc)
            expl_z = torch.randn(expl_size, exploit_z.shape[1], device=self.device)
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            # Decode the latents for both exploitation and exploration
            z = torch.cat([exploit_z, expl_z]).to("cuda:0")
            smiles = self.dec.sample(z, 50, argmax=False)
            
            # Get the log prob of these smiles by summing their prob of the latents and the smiles given the latents
            zc = torch.zeros(z.shape[0], self.num_features, device=self.device)
            conc_zy = torch.cat([z, zc], dim=1).to("cuda:0")
            log_probs = self.lp.log_prob(conc_zy, marg=self.num_latent * [False] + self.num_features * [True])
            log_probs += self.dec.weighted_forward(smiles, z)

            # get batch of experts for this iter
            batch_expert_smiles = random.sample(expert_smiles, compare_batch_size)
            batch_expert_mols = [get_mol(s) for s in batch_expert_smiles]
            batch_expert_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in batch_expert_mols]
            # get generated sample
            generated_mols = [get_mol(s) for s in smiles]
            generated_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) if mol is not None else None for mol in generated_mols] 
            
            # Reward: Nearest neighbour Tanimoto similarity, penalty for similar molecules, density reward
            # Get latent space positions
            with torch.no_grad():
                batch_expert_means, batch_expert_log_stds = torch.split(
                    self.enc.encode(batch_expert_smiles),
                    self.num_latent, dim=1
                )
                batch_expert_latents = (batch_expert_means + torch.randn_like(batch_expert_log_stds) * torch.exp(0.5 * batch_expert_log_stds)).cpu().detach().numpy()
            
            with torch.no_grad():
                g_means, g_log_stds = torch.split(
                    self.enc.encode(smiles),
                    self.num_latent, dim=1
                )
                g_latents = (g_means + torch.randn_like(g_log_stds) * torch.exp(0.5 * g_log_stds)).cpu().detach().numpy()
            
            nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(batch_expert_latents)

            # Calculate reward
            tanimoto_sims = []
            negative_reward = -0.5
            max_similarity = 0.6
            
            for idx, g_fp in enumerate(generated_fps):
                if g_fp is not None:
                    best_expert_similarity = 0
                    best_generated_similarity = 0
                    n_generated_similar = 0
                    all_generated_similarities = []
                    
                    for e_fp in batch_expert_fps:
                        similarity = DataStructs.TanimotoSimilarity(g_fp, e_fp)
                        best_expert_similarity = max(best_expert_similarity, similarity)
                    
                    for j, other_g_fp in enumerate(generated_fps):
                        # Do not compare the molecule with itself.
                        if idx != j and other_g_fp is not None:
                            similarity = DataStructs.TanimotoSimilarity(g_fp, other_g_fp)
                            best_generated_similarity = max(best_generated_similarity, similarity)
                            all_generated_similarities.append(similarity)
                            if similarity > max_similarity:
                                n_generated_similar += 1
                                
                    average_generated_similarity = sum(all_generated_similarities) / len(all_generated_similarities)
            
                    if average_generated_similarity > 0:
                        ratio = best_expert_similarity / average_generated_similarity
                    else:
                        ratio = best_expert_similarity
                    
                    # Calculate the density around the generated molecule.
                    distances, _ = nbrs.kneighbors([g_latents[idx]])
                    density_reward = np.exp(-np.mean(distances))
                    
                    reward = ratio + np.tanh(best_expert_similarity - average_generated_similarity) - 0.02 * n_generated_similar + density_reward
                    tanimoto_sims.append(reward)
                else:
                    tanimoto_sims.append(-0.5).
            
            rewards = torch.tensor(tanimoto_sims).float().to(exploit_z.device)
            r_list = rewards
            
            rewards_bl = rewards - rewards.mean()


            if iter_count <= num_decay_iterations:
                explore_perc -= decay_rate
                explore_perc = max(explore_perc, final_explore_perc)

            # Compute the loss as the rewards multiplied by the log probs of getting those rewards
            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()

            valid_sm = [s for s in smiles if get_mol(s) is not None]


            if iter_count % 500 == 0:
                p = Path("models") / train_name / 'IL4' / str(iter_count)
                p.mkdir(parents=True, exist_ok=True)
                self.save("./" + str(p) + "/")
                print('Mean reward: ' + str(sum(r_list) / len(smiles)))
                print('Valid perc: ' + str(len(valid_sm) / len(smiles)))
                print('Unique perc: ' + str(0 if len(valid_sm) == 0 else len(set(valid_sm)) / len(valid_sm)))

                

            if iter_count % 500 == 0 or iter_count == 1:
                csv_writer.writerow([iter_count, [s for s in smiles[:expl_size] if get_mol(s) is not None], [s for s in smiles[expl_size:] if get_mol(s) is not None]])
            iter_count += 1
            
        f.close()
        return None
            


    def train_as_il5(self,
                    expert_smiles,
                    decoy_smiles,
                    train_name,
                    num_iterations=2000,
                    batch_size=128,
                    compare_batch_size=128,
                    lr_lp=1e-5, lr_dec=1e-6,
                    initial_explore_perc=0.9, final_explore_perc=0.3, 
                    start_iter=1):
        # We optimize the parameters of the learned prior
        optimizer_lp = optim.Adam(self.lp.parameters(), lr=lr_lp)
        # We optimize the parameters of the fc layer from the decoder that interacts with the latent space directly
        optimizer_dec = optim.Adam(self.dec.latent_fc.parameters(), lr=lr_dec)
            
        fpgen = AllChem.GetRDKitFPGenerator()
        
        p = Path("models") / train_name / 'IL5'
        p.mkdir(parents=True, exist_ok=True)
        f = open(str(Path("models") / train_name / 'IL5' / (train_name + ".csv")), 'w')
        csv_writer = csv.writer(f)
        header = ["Iteration", "SMILES Explore", "SMILES Exploit"]
        csv_writer.writerow(header)

        explore_perc = initial_explore_perc
        num_decay_iterations = 3/4 * num_iterations
        decay_rate = (initial_explore_perc - final_explore_perc) / num_decay_iterations

        for iter_count in tqdm(range(start_iter, num_iterations + 1)):
            # Get exploitation latens
            exploit_size = int(batch_size * (1 - explore_perc))
            exploit_z = self.lp.sample(exploit_size, self.num_latent * ['s'] + self.num_features * ['m'])

            z_means = exploit_z.mean(dim=0)
            z_stds = exploit_z.std(dim=0)

            # Exploration latents are based on the mean and std of the exploit latens
            expl_size = int(batch_size * explore_perc)
            expl_z = torch.randn(expl_size, exploit_z.shape[1], device=self.device)
            expl_z = 2 * expl_z.to(exploit_z.device) * z_stds[None, :]
            expl_z += z_means[None, :]

            # Decode the latents for both exploitation and exploration
            z = torch.cat([exploit_z, expl_z]).to("cuda:0")
            smiles = self.dec.sample(z, 50, argmax=False)
            
            # Get the log prob of these smiles by summing their prob of the latents and the smiles given the latents
            zc = torch.zeros(z.shape[0], self.num_features, device=self.device)
            conc_zy = torch.cat([z, zc], dim=1).to("cuda:0")
            log_probs = self.lp.log_prob(conc_zy, marg=self.num_latent * [False] + self.num_features * [True])
            log_probs += self.dec.weighted_forward(smiles, z)

            # get batch of experts for this iter
            batch_expert_smiles = random.sample(expert_smiles, compare_batch_size)
            batch_expert_mols = [get_mol(s) for s in batch_expert_smiles]
            batch_expert_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in batch_expert_mols]
            # get batch of decoys for this iter
            batch_decoy_smiles = random.sample(decoy_smiles, compare_batch_size)
            batch_decoy_mols = [get_mol(s) for s in batch_decoy_smiles]
            batch_decoy_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) for mol in batch_decoy_mols]
            
            # get generated sample
            generated_mols = [get_mol(s) for s in smiles]
            generated_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2) if mol is not None else None for mol in generated_mols] 
            
            # Reward: Experts and decoys
            tanimoto_sims = []
            for idx, g_fp in enumerate(generated_fps):
                if g_fp is not None:
                    all_generated_similarities = []
                    all_expert_similarities = []
                    all_decoy_similarities = []
                    
                    for e_fp in batch_expert_fps:
                      similarity = DataStructs.TanimotoSimilarity(g_fp, e_fp)
                      all_expert_similarities.append(similarity)
                    
                    for d_fp in batch_decoy_fps:
                      similarity = DataStructs.TanimotoSimilarity(g_fp, d_fp)
                      all_decoy_similarities.append(similarity)

                    
                    for j, other_g_fp in enumerate(generated_fps):
                        if idx != j and other_g_fp is not None:
                            similarity = DataStructs.TanimotoSimilarity(g_fp, other_g_fp)
                            all_generated_similarities.append(similarity)
                    
                    best_expert_similarity = max(all_expert_similarities)
                    best_decoy_similarity = max(all_decoy_similarities)
                    average_generated_similarity = sum(all_generated_similarities) / len(all_generated_similarities)
                    
                    if average_generated_similarity > 0:
                        ratio = (best_expert_similarity*3) / average_generated_similarity
                    else:
                        ratio = (best_expert_similarity*3)
                    
                    if best_expert_similarity > best_decoy_similarity:
                      tanimoto_sims.append(ratio*3)
                    else:
                      tanimoto_sims.append(ratio)
                else:
                    tanimoto_sims.append(0)
            
            rewards = torch.tensor(tanimoto_sims).float().to(exploit_z.device)
            r_list = rewards
            
            rewards_bl = rewards - rewards.mean()


            if iter_count <= num_decay_iterations:
                explore_perc -= decay_rate
                explore_perc = max(explore_perc, final_explore_perc)

            # Compute the loss as the rewards multiplied by the log probs of getting those rewards
            optimizer_dec.zero_grad()
            optimizer_lp.zero_grad()
            loss = -(log_probs * rewards_bl).mean()
            loss.backward()
            optimizer_dec.step()
            optimizer_lp.step()

            valid_sm = [s for s in smiles if get_mol(s) is not None]


            if iter_count % 500 == 0:
                p = Path("models") / train_name / 'IL5' / str(iter_count)
                p.mkdir(parents=True, exist_ok=True)
                self.save("./" + str(p) + "/")
                print('Mean reward: ' + str(sum(r_list) / len(smiles)))
                print('Valid perc: ' + str(len(valid_sm) / len(smiles)))
                print('Unique perc: ' + str(0 if len(valid_sm) == 0 else len(set(valid_sm)) / len(valid_sm)))

                

            if iter_count % 500 == 0 or iter_count == 1:
                csv_writer.writerow([iter_count, [s for s in smiles[:expl_size] if get_mol(s) is not None], [s for s in smiles[expl_size:] if get_mol(s) is not None]])
            iter_count += 1
            
        f.close()
        return None
    