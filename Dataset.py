import json
import random
import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx
from typing import List, Union


class XASDataset(InMemoryDataset):
    
    # class variables
    ATOM_FEATURES = {
            'atomic_num': [6.0, 8.0],
            'degree': [0, 1, 2, 3, 4],
            'num_Hs': [0.0, 1.0, 2.0],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3
                ],
            }
    # Total number of atom features
    ATOM_FDIM = sum(len(choices) for choices in ATOM_FEATURES.values()) + 4
    # Number of bond features?
    BOND_FDIM = 14

    atom_ml = False


    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, atom_ml=False):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.atom_ml = atom_ml
 

    @property
    def raw_file_names(self):
        #return ['data_coronene_4sets_0.6_final.json']
        return ['data_coronene.json']


    @property
    def processed_file_names(self):
        return ['data_mol.pt']


    def onek_encoding_unk(self, value:int, choices:List[int]) -> List[int]:
        """
        Creates a one-hot encoding with an extra category for uncommon values.

        :param value: The value for which the encoding should be one.
        :param choices: A list of possible values.
        :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
                 If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
        """
        # Create vector of zeros for the atom feature
        encoding = [0.0] * (len(choices))
        # Find index value of specific atom in vector
        index = choices.index(value)
        # Set to 1
        encoding[index] = 1.0

        return encoding

    
    def mol_to_nx(self, mol, spec):
        # Create graph object
        G = nx.Graph()
        
        # For each atom in molecule
        for atom in mol.GetAtoms():
            # Add a node to graph and create one-hot encoding vector for atom features
            G.add_node(atom.GetIdx(), x=self.atom_features(atom))
            
        # For each bond in molecule
        for bond in mol.GetBonds():
            # Add edge to graph and create one-hot encoding vector of bond features
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                       edge_attr=self.bond_features(bond))
        
        # Normalize spectra to 1.0
        max_intensity = np.max(spec)
        norm_spec = 1.0 * (spec / max_intensity)
        # Set spectra to graph
        G.graph['spectrum'] = norm_spec
        
        return G

    
    def atom_features(self, atom:Chem.rdchem.Atom, functional_groups:List[int] = None) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for an atom.

        :param atom: An RDKit atom.
        :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
        :return: A list containing the atom features.
        """
        numOs = 0
        if atom is None:
            features = [0] * ATOM_FDIM
        else:
            for a in atom.GetNeighbors():
                if a.GetAtomicNum() == 8:
                    numOs += 1
            # Get the values of all the atom features and add all up to the feature vector
            features = self.onek_encoding_unk(atom.GetAtomicNum(), self.ATOM_FEATURES['atomic_num']) + \
                self.onek_encoding_unk(int(atom.GetDegree()), self.ATOM_FEATURES['degree']) + \
                self.onek_encoding_unk(int(atom.GetTotalNumHs()), self.ATOM_FEATURES['num_Hs']) + \
                self.onek_encoding_unk(int(atom.GetHybridization()), self.ATOM_FEATURES['hybridization']) + \
                [1 if atom.GetIsAromatic() else 0] + \
                [1 if numOs==0 else 0] + \
                [1 if numOs==1 else 0] + \
                [1 if numOs==2 else 0]
            #if functional_groups is not None:
            #    features += functional_groups
                
        return features

    
    def bond_features(self, bond:Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
        """
        Builds a feature vector for a bond.

        :param bond: An RDKit bond.
        :return: A list containing the bond features.
        """
        if bond is None:
            fbond = [1] + [0] * (BOND_FDIM - 1)
        else:
            # Get the bond type and create one-hot enconding vector
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                int(bt == Chem.rdchem.BondType.SINGLE),
                int(bt == Chem.rdchem.BondType.DOUBLE),
                int(bt == Chem.rdchem.BondType.AROMATIC),
                int(bond.GetIsConjugated() if bt is not None else 0),
                int(bond.IsInRing() if bt is not None else 0)
            ]
        return fbond

    
    def count_atoms(self, mol, atomic_num):
        # Works for all atoms, input atomic_num
        num_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == atomic_num:  # Atomic number 8 corresponds to oxygen
                num_atoms.append(atom.GetIdx())
        return num_atoms
    

    def message_pass(self, nodes, edges):
        outx = []
        s = -1
        for i in enumerate(edges[0,]):
            if s == int(i[1]):
                continue
                
            x = nodes[int(i[1])]
            s = int(i[1])
            count = 1

            for a in enumerate(edges[0,]):
                if int(a[1]) == int(i[1]):
                    x = x + nodes[int(edges[1, int(a[1])])]
                    count = count + 1
            x = x / count
            outx.append(x)
        outx = torch.stack(outx, 0)
        
        return outx
    

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]
        
        # Open the data file and load the data
        with open(self.raw_paths[0], "rb") as file:
                dictionaries = json.load(file)
        
        # Create a list of all the molecule names from the data
        tot_ids = list(dictionaries[0].keys())
        atom_count = []
        atom_num = []
        print('Total number of molecules', len(tot_ids)) 
        
        # For each molecule in the dataset
        for mol_id in tot_ids:
            mol = Chem.MolFromSmiles(dictionaries[0][mol_id])
            # Find the total number of atoms of a given atomic number
            tot_atoms = self.count_atoms(mol, 6)
            atom_count.append(tot_atoms)
            atom_num.append(len(tot_atoms))
        print('Number of atoms in each molecule ', atom_num)
        
        idx = 0
        data_list = []
        
        # For each molecule in dataset
        for i in range(len(tot_ids)):
            # Get the molecular structure from dictionary
            test_mol = Chem.MolFromSmiles(dictionaries[0][tot_ids[i]])
            # Get all spectra data from dictionary
            test_spec = dictionaries[1][tot_ids[i]]
            
            # For a whole molecule ML model
            if not self.atom_ml:
                # Create empty array for summed spectra
                tot_spec = np.zeros(len(test_spec[str(0)]))
                
                # For each atom in the molecule
                for j in atom_count[i]:
                    # Sum up all atom spectra
                    tot_spec += test_spec[str(j)]
                    
                # Create a graph 
                gx = self.mol_to_nx(test_mol, tot_spec)
                # Convert graph to pytorch geometric graph
                pyg_graph = from_networkx(gx)
                pyg_graph.idx = idx
                pyg_graph.smiles = Chem.MolToSmiles(test_mol)
                data_list.append(pyg_graph)
                idx += 1
                
            else: # For individual atom ML model
                
                # For every atom in each molecule
                for j in atom_count[i]:
                    # For individual atom ML model
                    spec_dict = test_spec[str(j)]
                    gx = self.mol_to_nx(test_mol, spec_dict)
                    pyg_graph = from_networkx(gx)
                    
            
                    msg = self.message_pass(pyg_graph.x, pyg_graph.edge_index)
                    msg = self.message_pass(msg, pyg_graph.edge_index)
                    msg = self.message_pass(msg, pyg_graph.edge_index)
                    pyg_graph.vector = msg[j]
            
                    pyg_graph.idx = idx
                    pyg_graph.smiles = Chem.MolToSmiles(test_mol)
                    neighbors = [x.GetIdx() for x in test_mol.GetAtomWithIdx(j).GetNeighbors()]
                    pyg_graph.atom_num = j
                    pyg_graph.neighbors = neighbors
                    data_list.append(pyg_graph)
                    idx += 1
        
        random.Random(258).shuffle(data_list)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        