import glob, os, random
import sys
from pymol import cmd
from InterfaceResidues import interfaceResidues

def gen_interface_func(pdbobject, interface_info, workdir):
	#pdbobject = sys.argv[1]
	namepdb = os.path.basename(pdbobject)
	name = namepdb.split('.')[0]

	#interface_info = sys.argv[2]
	chainsAB = interface_info.split('_')
	chainsAB = chainsAB[0]+chainsAB[1]

	#workdir = sys.argv[3]
	cmd.load(pdbobject)
	interfaces = []

	for i in range(len(chainsAB)):
		for j in range(i+1,len(chainsAB)):
			cha, chb = chainsAB[i], chainsAB[j]
			if cha == chb: continue
			# run script interfaceResidue in pymol to get residues on interface
			# script outputs results to temp/temp.txt 
			#cmd.do('interfaceResidue {}, chain {}, chain {}'.format(name, cha, chb))
			interfaceResidues(name, cA="c. "+cha, cB="c. "+chb, outputPath=workdir)
			# parse output file to list `interfaces`
			mapp = {'chA':cha, 'chB':chb}
			ffile = open('{}/temp.txt'.format(workdir), 'r')
			for line in ffile.readlines():
				linee = line.strip().split('_')
				resid = linee[0]
				chainn = mapp[linee[1]]
				inter='{}_{}_{}_{}'.format(cha, chb, chainn, resid)
				if inter not in interfaces:
					interfaces.append(inter)
			ffile.close()
			os.remove('{}/temp.txt'.format(workdir))

	ffile = open('{}/interface.txt'.format(workdir), 'w')
	for x in interfaces:
		ffile.write(x+'\n')
	ffile.close()

	cmd.save(pdbobject)
	cmd.delete('all')

	return interfaces


