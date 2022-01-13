import numpy as np
import sys,os, gc
import csv, glob
import os.path as path
import torch, pickle
from models import *
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from pymol import cmd  # to run predict_ddg() in jupyter notebook
import shutil
from gen_interface_func import gen_interface_func


def gen_graph_data(pdbfile, mutinfo, interfile, cutoff, if_info=None):
    max_dis = 12
    # load pdb file
    with open(pdbfile) as pdbfile:
        lines = pdbfile.read().splitlines()
    # residues on interface
    chainid = [x.split('_')[0] for x in mutinfo]
    interface_res = read_inter_result(interfile, if_info, chainid)
    if len(interface_res)==0:
        print('Warning: We do not find any interface residues between the two parts: {}. Please double check your inputs. Thank you!'.format(if_info))
    # build graph
    sample = build_graph(lines, interface_res, mutinfo, cutoff, max_dis)
    return sample

# parse temp/interface.txt to interface_res
def read_inter_result(path, if_info=None, chainid=None, old2new=None):
    if if_info is not None:
        info1 = if_info.split('_')
        pA = info1[0]
        pB = info1[1]
        mappings = {}
        for a in pA:
            for b in pB:
                if a not in mappings:
                    mappings[a] = [b]
                else:
                    mappings[a] += [b]
                if b not in mappings:
                    mappings[b] = [a]
                else:
                    mappings[b] += [a]

        target_chains = []
        for chainidx in chainid:
            if chainidx in mappings:
                target_chains += mappings[chainidx]

        target_inters = []
        for chainidx in chainid:
            target_inters += ['{}_{}'.format(chainidx,y) for y in target_chains]+\
                    ['{}_{}'.format(y,chainidx) for y in target_chains]

        target_inters =list(set(target_inters))
    else:
        target_inters = None
    
    # load {workdir}/interface.txt
    with open(path) as inter:
        interlines = inter.read().splitlines()
    # save to interface_res
    interface_res = []
    for line in interlines:
        iden = line[:3]
        if target_inters is None:
            if iden.split('_')[0] not in chainid and iden.split('_')[1] not in chainid:
                continue
        else:
            if iden not in target_inters:
                continue
        infor = line[4:].strip().split('_')  # chainid, resid
        assert len(infor)==2
        interface_res.append('_'.join(infor))

    if old2new is not None:
        mapps = {x[:-4] :y[:-4]  for x,y in old2new.items()}
        interface_res = [mapps[x] for x in interface_res if x in mapps]

    return interface_res

def build_graph(lines, interface_res, mutinfo, cutoff=3, max_dis=12, noisedict = None):
    atomnames = ['C','N','O','S']
    residues = ['ARG','MET','VAL','ASN','PRO','THR','PHE','ASP','ILE',\
            'ALA','GLY','GLU','LEU','SER','LYS','TYR','CYS','HIS','GLN','TRP']
    res_code = ['R','M','V','N','P','T','F','D','I',\
            'A','G','E','L','S','K','Y','C','H','Q','W']
    res2code ={x:idxx for x,idxx in zip(residues, res_code)}

    atomdict = {x:i for i,x in enumerate(atomnames)}
    resdict = {x:i for i,x in enumerate(residues)}
    V_atom = len(atomnames)
    V_res = len(residues)

    # Loop 1: loop over pdb file to prepare variables inter_coors_matrix and chain2id
    chain2id = []
    interface_coordinates = []
    line_list= []
    mutant_coords = []
    for line in lines:
        if line[0:4] == 'ATOM':
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname  = line[16:21].strip()
            chainid =  line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            if elemname not in atomdict:
                continue
            
            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue

            if chainid not in chain2id:
                chain2id.append(chainid)

            line_token = '{}_{}_{}_{}'.format(atomname,resname,chainid,res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue

            atomid = atomdict[elemname]  # atom name to index
            resid  = resdict[resname]  # residue name to index
            
            cr_token = '{}_{}'.format(chainid, res_idx)
            
            coords = torch.tensor([x,y,z])
            float_cd  = [float(x) for x in coords]
            cd_tensor = torch.tensor(float_cd)
            
            ## if cr_token is both in interface_res and mutinfo, 
            ## then append to interface_coordinates twice?
            if cr_token in interface_res:
                interface_coordinates.append(cd_tensor)
            if mutinfo is not None and cr_token in mutinfo:
                interface_coordinates.append(cd_tensor)
                interface_res.append(cr_token)  # expanded with mutinfo
                mutant_coords.append(cd_tensor)  # not used?

    inter_coors_matrix = torch.stack(interface_coordinates)
    chain2id = {x:i for i,x in enumerate(chain2id)}

    # Loop 2: loop over pdb file to get features
    n_features = V_atom+V_res+1+1+3 +1 +1+1+1
    line_list = []
    atoms = []
    flag_mut = False
    res_index_set = {}
    global_resid2noise = {}
    for line in lines:
        if line[0:4] == 'ATOM':
            features = [0]*n_features
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname  = line[16:21].strip()
            chainid =  line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            if elemname not in atomdict:
                continue

            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue
            
            line_token = '{}_{}_{}_{}'.format(atomname,resname,chainid,res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue

            atomid = atomdict[elemname]
            resid  = resdict[resname]
            cr_token = '{}_{}'.format(chainid, res_idx)
            
            coords = torch.tensor([x,y,z])
            float_cd  = [float(x) for x in coords]
            cd_tensor = torch.tensor(float_cd)
            
            # --- A: features of nodes ---
            # 0,1,2,3 : one-hot vector for elemname in atomnames
            features[atomid] = 1
            
            # 4,5,...,22,23 : one-hot vector for resname in residues
            features[V_atom+resid] = 1
            
            # 24 : whether residue on interface
            if cr_token in interface_res:
                features[V_atom+V_res] = 1
            
            # 25 : whether chain contains mutated residues
            if mutinfo is not None:
                for inforrr in mutinfo:
                    mut_chainid = inforrr.split('_')[0]
                    if chainid==mut_chainid:
                        features[V_atom+V_res+1] = 1
            
            # 26 : index of chain
            features[V_atom+V_res+2] = chain2id[chainid]
            
            # 27 : index of chainid_res_idx from 1 to ... 
            # related to positional encoding ?
            if cr_token not in res_index_set:
                res_index_set[cr_token] = len(res_index_set)+1
            features[V_atom+V_res+3] = res_index_set[cr_token]

            # 28 : whether CA atom
            # noisedict???
            if atomname=='CA':
                features[V_atom+V_res+4] = res_index_set[cr_token]
                if noisedict is not None and cr_token in noisedict:
                    global_resid2noise[res_index_set[cr_token]] = noisedict[cr_token]

            flag = False
            # cd_tensor: 1x3; inter_coors_matrix: Nx3; 
            # minus:expand cd_tensor along row then substract with inter_coors_matrix 
            dissss = torch.norm(cd_tensor-inter_coors_matrix,dim=1)  # 2-norm
            flag = (dissss<max_dis).any()  # flag to identify atoms contact with the others

            # 29-31 : coordinates x, y, z
            features[V_atom+V_res+5:V_atom+V_res+8] = float_cd

            res_iden_token = '{}_{}_{}'.format(chainid, res_idx, resname).upper()
            if  mutinfo is not None and cr_token in mutinfo:
                # 32: whether residue is mutated. Similar to 25
                features[V_atom+V_res+8]=1
                flag_mut = True
                flag = True
            
            # only consider atoms contacting with others or on mutation sites
            if flag:
                atoms.append(features)

    if mutinfo is not None and len(interface_res)>0:
        assert flag_mut==True
    
    if len(atoms)<5:
        return None
    
    # --- edges ---
    atoms = torch.tensor(atoms, dtype=torch.float)
    N = atoms.size(0)
    atoms_type = torch.argmax(atoms[:,:4],1)
    atoms_type = atoms_type.unsqueeze(1).repeat(1,N)
    # row*4[0,4,8,12] + column[0,1,2,3]: 
    # totally 4x4=16 combination to represent the combination of atom types
    edge_type = atoms_type*4+atoms_type.t()  

    pos = atoms[:,-4:-1]  # coordinates [x, y, z], shape Nx3
    row = pos[:,None,:].repeat(1,N,1)  # shape NxNx3, dim 1 is repeated
    col = pos[None,:,:].repeat(N,1,1)  # shape NxNx3, dim 0 is repeated
    direction = row-col  # [x_i-x_j, y_i-y_j, z_i-z_j]_{ij}
    del row, col
    # distance and its inverse
    distance = torch.sqrt(torch.sum(direction**2,2))+1e-10  # [r_i - r_j]_{ij}
    distance1 = (1.0/distance)*(distance<float(cutoff)).float()  # 1/r_{ij}
    del distance
    # set diagonal values to 1
    diag = torch.diag(torch.ones(N))
    dist = diag+ (1-diag)*distance1
    del distance1, diag    
    # adjacency matrix of interface atoms' graph
    # value: 0 or 1;
    # shape: NxN
    flag = (dist>0).float() # r_{ij} with distance less than cutoff -> 1
    direction = direction*flag.unsqueeze(2)  # not used??
    del direction, dist
    # [row, column] indices of nonzero element in flag
    # shape: (number of nonzero elements in flag)x2
    edge_sparse = torch.nonzero(flag)
    # value 0-15 to represent contacts of different types of atoms
    # shape: (number of nonzero elements in flag),
    edge_attr_sp = edge_type[edge_sparse[:,0],edge_sparse[:,1]]
    if noisedict is None:
        savefilecont = [atoms, edge_sparse, edge_attr_sp]
    else:
        savefilecont = [atoms, edge_sparse, edge_attr_sp, global_resid2noise]
    return savefilecont

def main():
    gnnfile = 'trainedmodels/GeoEnc.tor'
    gbtfile = 'trainedmodels/gbt-s4169.pkl'
    idxfile = 'trainedmodels/sortidx.npy'
    pdbfile = sys.argv[1]
    mutationinfo = sys.argv[2]
    if_info = sys.argv[3]


    try:
        sorted_idx = np.load(idxfile)
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(idxfile))
        #sorted_idx = [i for i in range(1000)]  # meanless, just for testing the follows


    os.system('cp {} ./'.format(pdbfile))
    pdbfile = pdbfile.split('/')[-1]
    pdb = pdbfile.split('.')[0]
    workdir = 'temp'
    cutoff = 3

    if path.exists('./{}'.format(workdir)):
        os.system('rm -r {}'.format(workdir))
    os.system('mkdir {}'.format(workdir))

    # generate the `interface residues
    os.system('python gen_interface.py {} {} {} > {}/pymol.log'.format(pdbfile,if_info,workdir,workdir))
    interfacefile = '{}/interface.txt'.format(workdir)

    # Extract mutation information
    graph_mutinfo = []
    flag = False
    # single-point mutation
    info = mutationinfo
    wildname = info[0]
    chainid = info[1] 
    resid = info[2:-1]
    mutname = info[-1]
    if wildname==mutname:flag = True
    graph_mutinfo.append('{}_{}'.format(chainid,resid))

    # build a pdb file that is mutated to it self
    with open('individual_list.txt','w') as f:
        cont = '{}{}{}{};'.format(wildname,chainid,resid,wildname)
        f.write(cont)
    comm = './foldx --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(\
                                pdbfile,  'individual_list.txt', workdir, './',workdir)
    os.system(comm)
    os.system('mv {}/{}_1.pdb   {}/wildtype.pdb '.format(workdir, pdb, workdir))

    # build the mutant file
    with open('individual_list.txt','w') as f:
        cont = '{}{}{}{};'.format(wildname,chainid,resid,mutname)
        f.write(cont)
    comm = './foldx --command=BuildModel --pdb={}  --mutant-file={}  --output-dir={} --pdb-dir={} >{}/foldx.log'.format(\
                                pdbfile,  'individual_list.txt', workdir, './',workdir)
    os.system(comm)

    wildtypefile = '{}/wildtype.pdb'.format(workdir, pdb)
    mutantfile = '{}/{}_1.pdb'.format(workdir, pdb)

    try:
        A, E, _ = gen_graph_data(wildtypefile, graph_mutinfo, interfacefile , cutoff, if_info)
        A_m, E_m, _= gen_graph_data(mutantfile, graph_mutinfo, interfacefile , cutoff, if_info)
    except:
        print('Data processing error: Please double check your inputs is correct! Such as the pdb file path, mutation information and binding partners. You might find more error details at {}/foldx.log'.format(workdir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GeometricEncoder(256)
    try:
        model.load_state_dict(torch.load(gnnfile,map_location='cpu'))
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(gnnfile))


    model.to(device)
    model.eval()
    A = A.to(device)
    E = E.to(device)
    A_m = A_m.to(device)
    E_m = E_m.to(device)

    try:
        with open(gbtfile, 'rb') as pickle_file:
            forest = pickle.load(pickle_file)
    except:
        print('File reading error: Please redownload the file {} via the following command: \
                wget https://media.githubusercontent.com/media/Liuxg16/largefiles/8167d5c365c92d08a81dffceff364f72d765805c/gbt-s4169.pkl -P trainedmodels/'.format(gbtfile))

    ddg = GeoPPIpredict(A,E,A_m,E_m, model, forest, sorted_idx, flag)
 
    print('='*40+'Results'+'='*40)
    if ddg<0:
        mutationeffects = 'destabilizing'
        print('The predicted binding affinity change (wildtype-mutant) is {} kcal/mol ({} mutation).'.format(ddg,mutationeffects))
    elif ddg>0:
        mutationeffects = 'stabilizing'
        print('The predicted binding affinity change (wildtype-mutant) is {} kcal/mol ({} mutation).'.format(ddg,mutationeffects))
    else:
        print('The predicted binding affinity change (wildtype-mutant) is 0.0 kcal/mol.')


    os.system('rm ./{}'.format(pdbfile))
    os.system('rm ./individual_list.txt')


# ----------------------------------------------------------------
def predict_ddg(pdbfile, mutationinfo, if_info, workdir_suffix="", remove_temp=True):
    gnnfile = 'trainedmodels/GeoEnc.tor'
    gbtfile = 'trainedmodels/gbt-s4169.pkl'
    idxfile = 'trainedmodels/sortidx.npy'


    try:
        sorted_idx = np.load(idxfile)
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(idxfile))
        #sorted_idx = [i for i in range(1000)]  # meanless, just for testing the follows


    pdb = pdbfile.split('/')[-1]
    pdb = pdb.split('.')[0]
    workdir = 'temp'+workdir_suffix
    cutoff = 3  # ?

    if path.exists('./{}'.format(workdir)):
        shutil.rmtree(workdir)  #os.system('rm -r {}'.format(workdir))
    os.mkdir(workdir)  #os.system('mkdir {}'.format(workdir))
    pdbfile = shutil.copy(pdbfile, workdir)  #os.system('cp {} ./'.format(pdbfile))

    # --- interfacefile ---
    # generate the interface residues
    #os.system('python gen_interface.py {} {} {} > {}/pymol.log'.format(pdbfile, if_info,workdir,workdir))
    gen_interface_func(pdbfile, if_info, workdir)
    interfacefile = '{}/interface.txt'.format(workdir)

    # --- wildtypefile & mutantfile & graph_mutinfo ---
    # Extract mutation information  
    '''
    # single-point mutation
    info = mutationinfo
    wildname = info[0]
    chainid = info[1] 
    resid = info[2:-1]
    mutname = info[-1]
    '''
    # multiple-point mutation
    wildname = []
    chainid = []
    resid = []
    mutname = []
    for info in mutationinfo.split(","):
        wildname.append(info[0])
        chainid.append(info[1])
        resid.append(info[2:-1])
        mutname.append(info[-1])
    #
    graph_mutinfo = []
    for i in range(len(chainid)):
        graph_mutinfo.append('{}_{}'.format(chainid[i], resid[i]))
    # flag to specify unmutated case
    flag = False
    if wildname==mutname: flag = True

    # build a pdb file that is mutated to it self
    with open('{}/individual_list.txt'.format(workdir),'w') as f:
        cont = ''
        separator = list(','*(len(wildname)-1)+';')
        for i in range(len(wildname)):
            cont += '{}{}{}{}{}'.format(wildname[i],
                                        chainid[i],
                                        resid[i],
                                        wildname[i],  # itself
                                        separator[i])
        f.write(cont)
    comm = './foldx --command=BuildModel --pdb={} --mutant-file={} --output-dir={} --pdb-dir={} > {}/foldx.log'.format(\
                                pdb+".pdb", '{}/individual_list.txt'.format(workdir), workdir, workdir, workdir)
    os.system(comm)
    os.system('mv {}/{}_1.pdb   {}/wildtype.pdb '.format(workdir, pdb, workdir))

    # build the mutant file
    with open('{}/individual_list.txt'.format(workdir),'w') as f:
        cont = ''
        separator = list(','*(len(wildname)-1)+';')
        for i in range(len(wildname)):
            cont += '{}{}{}{}{}'.format(wildname[i],
                                        chainid[i],
                                        resid[i],
                                        mutname[i],  # mutant
                                        separator[i])
        f.write(cont)
    comm = './foldx --command=BuildModel --pdb={} --mutant-file={} --output-dir={} --pdb-dir={} > {}/foldx.log'.format(\
                                pdb+".pdb", '{}/individual_list.txt'.format(workdir), workdir, workdir, workdir)
    os.system(comm)

    wildtypefile = '{}/wildtype.pdb'.format(workdir, pdb)
    mutantfile = '{}/{}_1.pdb'.format(workdir, pdb)
    
    # --- gen_graph_data ---
    try:
        A, E, _ = gen_graph_data(wildtypefile, graph_mutinfo, interfacefile, cutoff, if_info)
        A_m, E_m, _ = gen_graph_data(mutantfile, graph_mutinfo, interfacefile , cutoff, if_info)
    except:
        print('Data processing error: Please double check your inputs is correct! Such as the pdb file path, mutation information and binding partners. You might find more error details at {}/foldx.log'.format(workdir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GeometricEncoder(256)
    try:
        model.load_state_dict(torch.load(gnnfile, map_location='cpu'))
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(gnnfile))
        
    model.to(device)
    model.eval()
    A = A.to(device)
    E = E.to(device)
    A_m = A_m.to(device)
    E_m = E_m.to(device)
    
    try:
        with open(gbtfile, 'rb') as pickle_file:
            forest = pickle.load(pickle_file)
    except:
        print('File reading error: Please redownload the file {} via the following command: \
                wget https://media.githubusercontent.com/media/Liuxg16/largefiles/8167d5c365c92d08a81dffceff364f72d765805c/gbt-s4169.pkl -P trainedmodels/'.format(gbtfile))

    ddg = GeoPPIpredict(A, E, A_m, E_m, model, forest, sorted_idx, flag)

    #print('='*40+'Results'+'='*40)
    if ddg<0:
        mutationeffects = 'destabilizing'
        #print('The predicted binding affinity change (wildtype-mutant) is {} kcal/mol ({} mutation).'.format(ddg,mutationeffects))
    elif ddg>0:
        mutationeffects = 'stabilizing'
        #print('The predicted binding affinity change (wildtype-mutant) is {} kcal/mol ({} mutation).'.format(ddg,mutationeffects))
    else:
        #print('The predicted binding affinity change (wildtype-mutant) is 0.0 kcal/mol.')
        None

    #os.system('rm ./{}'.format(pdbfile))
    #os.system('rm ./individual_list.txt')
    if remove_temp:
        shutil.rmtree(workdir)
    
    return ddg


# wrapper for multiprocessing.apply_async
def predict_ddg_wrapper(name, args):
    return name, predict_ddg(*args)

if __name__ == "__main__":
    main()

