#Implemetation of the TASEP based Ribosome Flow Model by Tuller et al (2017)
# Copyright (C) 2021 by
#Author :  Deepti Vipin
# All rights reserved
# Released under MIT license (see LICENSE.txt)

import os
import sys
sys.path.append(os.getcwd())
import subprocess
import numpy as np
import time
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib._color_data as mcd
from decimal import Decimal
import matplotlib.ticker as mtick
import math
import seaborn as sns
import csv
import re
import faulthandler; faulthandler.enable()
from guppy import hpy

#time it
tic = time.clock()

# Set filenames from directory
# tai_file -->  calculated values for tRNA:RNA interaction
# mrna_abund_file --> file with mRNA counts
#trna_file --> tRNA conc derived
# init_mrna_file --> initiation rates of m
# fasta_file --> Fasta file for host
# output_dir --> dir to save output



#Read codon- anticodon pairs for amino acids
#aa_cdn = pd.read_excel(io=trna_file)
aa_cdn['AA-codon'] = aa_cdn.apply(lambda x: '-'.join(x), axis=1)
aa_cdn = aa_cdn.sort_values(by=['Codon']).reset_index(drop=True)

#read file with mRNA concentrations
#mrna_abund = pd.read_excel(io=mrna_abund_file)
mrna_abund.columns = ['gene','cond1','cond2','cond3','wt']

##Adjust for genes composition#####
mrna_abund['WT'] =mrna_abund['WT'].apply(lambda x: (x*60000)/
                   (mrna_abund['WT'].sum() -x))

wt_mrna = mrna_abund.set_index(['gene']).to_dict()['WT']

#Calculate codon elongation rates
#gcn_trna_abund = pd.read_excel(io=trna_file)
gcn_trna_abund.columns = ['codon','anti-codon','Wi']
gcn_tsum =(gcn_trna_abund.groupby('codon')['Wi'].sum()).to_dict()
gcn_tsum = { k.replace('U', 'T'): v for k, v in gcn_tsum.items() }

wt_tsum = pd.read_excel(io=tai_file, sheet_name="WT_16hYEPD")
wt_tsum =(wt_tsum.groupby('codon')['Wi'].sum()+1).reset_index()
wt_tsum = wt_tsum.set_index(['codon']).to_dict()['Wi']
wt_tsum = { k.replace('U', 'T'): v for k, v in wt_tsum.items() }

################################################
#convert genome to RFMIOs
with open(fasta_file, 'r') as f1:
	fa = f1.readlines()

# from https://gamma2.wordpress.com/2014/01/03/reading-a-fasta-file-with-python/
C = 25 # chunk size
seq_id = []
cds = {}

for line in fa:
	line = line.rstrip()
	if re.match(r"^>", line):
		seq_id=str.split(line.strip(),">")[1].split(" ")[0].split("_")[0]
		cds[seq_id] = ''
	else:
		cds[seq_id] = cds[seq_id] + line

#fasta to codons
mrna = {}
mrna_cdns = {}
mrna_length = {}
for key, value in cds.items():
    triplets = []
    for i in range(0,len(value)-(len(value)%3), 3):
        if value[i:i+3] in ['TAG','TAA','TGA']: break #stop codon
        triplets.append(value[i:i+3])
    #split mRNA to chunks size C
    chunks = [triplets[x:x+C] for x in range(0,len(triplets), C)]
    if len(chunks) <=3: continue
    mrna[key] = chunks
    #store triplets
    mrna_cdns[key] = triplets
    #store length
    mrna_length[key] = len(triplets)


#read file with mRNA initiations
#mrna_init = pd.read_excel(io=init_mrna_file, sheet_name)
#median = 0.0903852 (285 NaNs)
mrna_init['alpha_phys'] = mrna_init['alpha_phys'].fillna(0.0903852)

#Set global variables
mrna_ids0 = list(set(mrna_init['Gene_name']) & set(mrna_abund['gene'])) # len= 5064
mrna_init = mrna_init.set_index(['Gene_name']).to_dict()['alpha_phys']

def get_elong(mrna_abundance,tsum):
    #make dict with mrna initation rate and copy number
    mrna_init_copies ={}
    for gene in mrna_ids0:
        mrna_init_copies[gene] = [mrna_init[gene],math.ceil(mrna_abundance[gene])]
    mrna_ids = list(set(mrna_ids0) & set(mrna.keys())) # len= 5064
    translocation = {}
    for gene_id in mrna_ids:
        translocation[gene_id] = [sum([(tsum[ele]*mrna_init_copies[gene_id][1]) for ele in chunk]) for chunk in mrna[gene_id]]
    return (translocation,mrna_init_copies,mrna_ids)



def get_init_mat(mrna_ids):
    z0= 300000 #pool of ribosome constants
    xmat = [] #array holding mrnas no of sites
    for m in range(len(mrna_ids)):
        xmat.append(np.zeros(len(mrna[mrna_ids[m]])).tolist())
    x0 = [i for sub in xmat for i in sub] #flatten
    x0.append(z0) #add ribosome pool
    #get positions in flattened list
    x_size = [len(mrna[mrna_ids[m]]) for m in range(0,len(mrna_ids))]
    s_size = [0] +x_size[:-1]
    i=[]
    x_dim = []
    for s,e in zip(s_size,x_size):
        x_dim.append((s+sum(i),s+e+sum(i)))
        i.append(s)
    return(x0,x_dim)




def rfmio(x,t,translocation,mrna_init_copies,mrna_ids,x_dim):
   z = x[-1]
   in_flow = 0
   out_flow = 0
   dx =[]
   for m in range(0,len(mrna_ids)):
       p = np.array(x[x_dim[m][0]: x_dim[m][1]])
       ids = mrna_ids[m]
       ld = translocation[ids]
       ld_init = mrna_init_copies[ids][0]
       ld0 = ld_init* np.tanh(z)
       n = p.size -1
       dp = np.zeros(p.size)
       dp[0] = ld0*(1-p[0]) - ld[0]*p[0]*(1-p[1])
       for i in range(1,n-1):
           dp[i] = ld[i-1]*p[i-1]*(1-p[i]) - ld[i]*p[i]*(1-p[i+1])
       dp[n-1] = ld[n-2]*p[n-2]*(1-p[n-1]) - ld[n-1]*p[n-1]
       dp[n] = ld[n-1]*p[n-1]
       dx.append(dp)
       in_flow += ld0*(1-p[0])
       out_flow += ld[n-1]*p[n-1]
   dx = [site for gene in dx for site in gene]
   dz= np.float64(out_flow - in_flow)
   dx.append(dz)
   return dx

def protein_rate(rfm_sol):
    protein_synth ={}
    for k,v in rfm_sol.items():
        protein_synth[k] = v[-1][-1]
    protein_synth =  pd.DataFrame.from_dict(protein_synth, orient='index')
    return protein_synth

def site_occupancy(rfm_sol):
    ribo_occupancy ={}
    for k,v in rfm_sol.items():
        ribo_occupancy[k] = v[-1][:-1]
    ribo_occupancy =  pd.DataFrame.from_dict(ribo_occupancy, orient='index')
    return ribo_occupancy

def protein_rate_norm(rfm_sol,mrna_length):
    protein_synth ={}
    for k,v in rfm_sol.items():
        protein_synth[k] = (v[-1][-1])/mrna_length[k]
    protein_synth =  pd.DataFrame.from_dict(protein_synth, orient='index')
    return protein_synth


t = np.linspace(0,2000,100)
wt_elong, wt_init_copies,wt_mrna_ids  = get_elong(wt_mrna,wt_tsum)
wt_x0, wt_xdim = get_init_mat(wt_mrna_ids[0:3])
wt_sol = odeint(rfmio,wt_x0,t, args=(wt_elong,wt_init_copies,wt_mrna_ids[0:3],wt_xdim))

np.save(output_dir, wt_sol)

wt_synth = protein_rate(wt_sol)
wt_occupancies = site_occupancy(wt_sol)

wt_occupancies.to_csv(output_dir+"/Occupancies_WT.csv", sep='\t')
wt_synth.to_csv(output_dir+"SyntRate_WT.csv", sep='\t')

toc = time.clock()
print("--- %s seconds ---" % (toc - tic))
