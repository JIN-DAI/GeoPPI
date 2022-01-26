import random


# --- GLOABEL ---
AALETTER = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# --- Helper Functions ---

# Read FASTA Files
def read_FASTA(file):
    fa_info = []  # list to store information
    fa_seq = []  # list to store sequences

    # read fasta file
    with open(file, "r") as fa_in:
        for line in fa_in.readlines():
            line = line.rstrip()
            if line[0] == ">":  # start of information line
                fa_info.append(line)
                fa_seq.append("")
            else:
                fa_seq[-1] = fa_seq[-1] + line
    
    return fa_info, fa_seq


# Write FASTA Files
def write_FASTA(file, fa_info, fa_seq):
    if len(fa_info) != len(fa_seq):
        raise ValueError("info and seq must be of the same length")
    
    # open file to write
    with open(file, "w") as fa_out:
        for i in range(len(fa_info)):
            fa_out.write(fa_info[i] + "\n")
            fa_out.write(fa_seq[i] + "\n")


# Compare Two Squences
def compare_seq(seq_a, seq_b):
    if len(seq_a) != len(seq_b):
        raise ValueError("Sequence a and b must be of the same length")
    return "".join(["-" if a == b else "+" for a, b in zip(seq_a, seq_b)])


# read mut_profile
def read_mut_profile(mut_profile, verbose=True):
    mut_profile_dict = {}
    
    with open(mut_profile, 'r') as f:
        for line in f:
            linestr = line.strip().split()
            if linestr[0] == 'PROB':
                mut_prob = [float(l) for l in linestr[1:]]  # mutation probability
            if linestr[0] == 'RES':
                mut_aa_list = "".join(linestr[3:])
                mut_prob_list = [mut_prob[idx]/len(linestr[3+idx]) for idx, l in enumerate(linestr[3:]) for _ in l]
                mut_profile_dict[int(linestr[1])] = (mut_aa_list, mut_prob_list, linestr[2])
    
    if verbose:
        print("Single point mutation scheme:")
        print("   {:3s} {:3s} {:20s} [{:}]".format("Idx", "Src", "Tgt", "Prob"))
        for key, value in mut_profile_dict.items():
            print("   {:3d}  {:1s}  {:20s} [{:}]"
                  .format(key, value[2], value[0], " ".join(map(lambda s: "%.2f"%s, value[1]))))
    
    return mut_profile_dict


# List of Strings to Specify Mutation
'''
input : MAADGYLP
        MDADGYLG
output: AA2D,PA8G
'''
def mutation_list(seq_src, seq_obj, mut_profile_dict=None, mut_chain_ch=""):
    if len(seq_src) != len(seq_obj):
        raise ValueError("Sequence src and obj must be of the same length")
    if not mut_profile_dict is None and len(seq_src) != len(mut_profile_dict):
        raise ValueError("Sequence src and mut_profile_dict must be of the same length")
    if not mut_profile_dict is None and len(seq_obj) != len(mut_profile_dict):
        raise ValueError("Sequence obj and mut_profile_dict must be of the same length")
    
    if mut_profile_dict is None:
        mut_index = list(range(len(seq_src)))
    else:
        mut_index = mut_profile_dict.keys()
        
    mut_list = []
    for src, obj, key in zip(seq_src, seq_obj, mut_index):
        if src != obj:
            mut_list.append("{:}{:}{:d}{:}".format(src, mut_chain_ch, key, obj))
    
    return mut_list


# String to Describe Mutation Information as Input of `GeoPPI`
'''
Example of mutation information string:  
`WTRes1` `CH1` `ResNo1` `muRes1`, `WTRes2` `CH2` `ResNo2` `muRes2`, ... , `WTRes3` `CH3` `ResNo3` `muRes3`
'''
def individual_list_string(seq_src, seq_obj, mut_profile_dict=None, mut_chain_ch=""):
    return ",".join(mutation_list(seq_src, seq_obj, mut_profile_dict, mut_chain_ch))


# return a string only including amino acid in mutation region
def extract_mutation_sites(seq, mut_profile_dict, seq_start_idx):
    return "".join([seq[key-seq_start_idx] for key in mut_profile_dict.keys()])


# return a string of full sequence with mutated amino acid
def inject_mutation_sites(mut_sub_seq, seq, mut_profile_dict, seq_start_idx):
    if len(mut_sub_seq) != len(mut_profile_dict):
        raise ValueError("mut_sub_seq and mut_profile_dict must be of the same length")
        
    seq_new = list(seq)
    for idx, key in enumerate(mut_profile_dict.keys()):
        #print(key, seq_new[key-seq_start_idx], mut_profile_dict[key][0])
        seq_new[key-seq_start_idx] = mut_sub_seq[idx] 
    return "".join(seq_new)


# --- GA Functions ---
#  Selection Function
def selection_pair(seq_pool, fitness_func):
    return random.choices(seq_pool, weights=[fitness_func(seq) for seq in seq_pool], k=2)


# Crossover at Single Point
def single_point_crossover(seq_a, seq_b):
    if len(seq_a) != len(seq_a):
        raise ValueError("Sequence a and b must be of same length")
    
    length = len(seq_a)
    if len(seq_a) < 2:
        return seq_a, seq_b
    
    p = random.randint(1, length-1)
    return seq_a[:p]+seq_b[p:], seq_b[:p]+seq_a[p:]


# Point Mutation Function
def mutation(seq, mut_profile_dict, num=1, whole_chain=True):
    if not whole_chain and len(seq) != len(mut_profile_dict):
        raise ValueError("seq and mut_profile_dict must be of same length")
    
    seq = list(seq)
    
    # dealing with a.a. index difference in seq and mut_profile_dict
    if whole_chain:
        # (index in in the whole sequence, index in the whole sequence)
        idx_tuple = tuple(zip([i-1 for i in mut_profile_dict.keys()], list(mut_profile_dict.keys())))
    else:
        # (index in the sub sequence of mutation region, index in the whole sequence)
        idx_tuple = tuple(zip([i for i in range(len(mut_profile_dict))], list(mut_profile_dict.keys())))
    
    # select indice to mutate
    idx_to_mut = random.sample(idx_tuple, k=num)
    
    # mutate each index
    for idx in idx_to_mut:
        seq[idx[0]] = random.choices(mut_profile_dict[idx[1]][0], weights=mut_profile_dict[idx[1]][1])[0]
    
    return "".join(seq)

