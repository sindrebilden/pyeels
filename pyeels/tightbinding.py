from pyeels.crystal import Crystal
from pyeels.band import Band
import matplotlib.pyplot as plt
import spglib as spg
import pythtb as tb
import numpy as np

class TightBinding:
    """ Tight binding class constructed around the pythTB package """
    def __init__(self, crystal):
        """ Create instance of the Tight binding model 
        
        :type  crystal: crystal object
        :param crystal: a crystal object containing atoms
        """
        
        self.crystal = crystal
        self.crystal.brillouinZone.band_model = "Tight Binding"
        self.spg = (crystal.lattice, crystal.getAtomPositons(), crystal.getAtomNumbers())
        
        self._orbital_positons = []
        for atom in crystal.atoms:
            for orbital in atom.orbitals:
                self._orbital_positons.append(atom.position)        
        
        self.setGrid()
        self.model = tb.tb_model(3,3,self.crystal.lattice, self._orbital_positons)
        
    def setGrid(self, mesh=3):
        """ Define the resolution of the reciprocal space
        
        :type  mesh: ndarray, list, int
        :param mesh: The number of point along a reciprocal latticevector
        """
        if isinstance(mesh, (float, int)):
            mesh = np.ones((3,),dtype=np.int)*int(mesh)

        if isinstance(mesh, (list, tuple)):
            mesh = np.asarray(mesh)

        if isinstance(mesh, np.ndarray):
            self.crystal.brillouinZone.mesh = mesh
            mapping, grid = spg.get_ir_reciprocal_mesh(mesh, self.spg, is_shift=[0, 0, 0])
            
            if np.any(mesh==np.array([1, 1, 1])):
                mesh+= (mesh==np.array([1, 1, 1]))*1
            k_grid = grid[np.unique(mapping)]/(mesh-1)

            k_list = []
            for i, map_id in enumerate(mapping[np.unique(mapping)]):
                k_list.append((grid[mapping==map_id]/(mesh-1)).tolist()) #np.dot(,self.cell.brillouinZone)
            self.k_grid = k_grid
            self.k_list = k_list
        else:
            _logger.warning("Unknown type {} for mesh, try ndarray.".format(type(mesh)))
        
        
    def display_pythtb(self):
        """ Displat the info from pythTB """
        self.model.display()
        
    def calculate(self, eig_vectors = False):
        """ Calculate band energies for the given k-points and place them as band objects in the crystal, 
        NB! replaces the full band structure
        
        :type  eig_vectors: bool
        :param eig_vectors: if eigen vectors are returned
        """
        self.crystal.brillouinZone.bands = []
        
        if eig_vectors:
            energies, waves = self.model.solve_all(self.k_grid,eig_vectors=eig_vectors)
        else:
            energies = self.model.solve_all(self.k_grid,eig_vectors=eig_vectors)
            waves = np.stack([np.zeros(energies.shape),np.ones(energies.shape)], axis=2)
        
        for i, band in enumerate(energies):
            self.crystal.brillouinZone.add_band(Band(k_grid=self.k_grid, k_list=self.k_list, energies=band, waves=waves[i]))

    
    def bandstructure(self, ylim=(None,None),  bands=(None,None), color=None, ax=None):
        """ Plot a representation of the band structure
        
        :type  ylim: tuple, list
        :param ylim: lower and upper limit of y-values (ymin,ymax)
        """

        """ seekpath automatic lines"""
        #path = sp.get_explicit_k_path((lattice, positions, numbers), False, recipe="hpkot", threshold=1e-5,reference_distance=1)
        #expath = path['explicit_kpoints_abs'][:5]
        #labels = path['explicit_kpoints_labels'][:5]

        """ manual lines"""
        path=[[0.0,0.0,0.5],[0.5,0.0,0.5],[0.5,0,0.0],[0.0,0.0,0.0],[0,0,0.5],[2./3.,1./3.,0.5],[2./3.,1./3.,0.0],[0,0,0]]
        label=(r'$A $',      r'$L$',       r'$M$',   r'$\Gamma$', r'$A $', r'$H$',  r'$K$',r'$\Gamma $')

        
        # call function k_path to construct the actual path
        (k_vec,k_dist,k_node)=self.model.k_path(path,301,report=False)

        evals =self.model.solve_all(k_vec)
        
        fig = None
        if not ax:
            fig, ax = plt.subplots(figsize=(8,6))
            fig.tight_layout()

            ax.set_title("Bandstructure for Zno based on Kobayashi")
            ax.set_ylabel("Band energy")

            # specify horizontal axis details
            ax.set_xlim([0,k_node[-1]])
            # put tickmarks and labels at node positions
            ax.set_xticks(k_node)
            ax.set_xticklabels(label)
            # add vertical lines at node positions

            for n in range(len(k_node)):
                if label[n] == r'$\Gamma$':
                    ax.axvline(x=k_node[n],linewidth=1, color='k')
                else:
                    ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    
        for band in evals[bands[0]:bands[1]]:
            ax.plot(k_dist, band, color=color)

        if not fig:
            return ax
        else:
            ax.set_ylim(ylim)
            return ax, fig

    def __repr__(self):
        return "Parabolic band model for: \n \n {} \n".format(self.crystal)





class WursiteSP3(TightBinding):
    
    def __init__(self, crystal):
        TightBinding.__init__(self, crystal)
        
    def get_hopping_parameters(self):
        """ Get a list of all hopping parameters in the model"""
        return [self.Vss, self.Vxx, self.Vxy, self.Vsapc, self.Vpasc ]
        
        
    def set_hopping_parameters(self, Vss, Vxx, Vxy, Vsapc, Vpasc):
        """ Set all hopping parameters in the model
        
        :type  Vss: float
        :param Vss: The hopping parameter from s to s
        
        :type  Vxx: float
        :param Vxx: The hopping parameter from px of the anion to px of the cation
        
        :type  Vxy: float
        :param Vxy: The hopping parameter from px of the anion to py of the cation
        
        :type  Vsapc: float
        :param Vsapc: The hopping parameter from s of the anion to p of the cation
        
        :type  Vpasc: float
        :param Vpasc: The hopping parameter from p of the anion to s of the cation
        """
        self.Vss = Vss
        self.Vxx = Vxx
        self.Vxy = Vxy
        self.Vsapc = Vsapc
        self.Vpasc = Vpasc
        
        """###############     Vertical bonding system    ##################"""      
        self.UVss = 0.25*Vss
        self.UVzz = 0.25*(Vxx+2*Vxy)
        self.UVxx = 0.25*(Vxx-Vxy)
        self.UVsz = -0.25*np.sqrt(3)*Vsapc
        self.UVzs =  0.25*np.sqrt(3)*Vpasc
        
        
        """###############    Horizontal bonding system   ##################"""
        self.UHss = self.UVss
        
        self.UHyy = self.UVxx
        self.UHzz = 1/9 * (8*self.UVxx +   self.UVzz)
        self.UHxx = 1/9 * (  self.UVxx + 8*self.UVzz)
        
        self.UHsz = -1/3 * self.UVsz
        self.UHzs = -1/3 * self.UVzs
        
        self.UHsx = -2*np.sqrt(2)/3 * self.UVsz
        self.UHxs = -2*np.sqrt(2)/3 * self.UVzs
        
        self.UHzx =  2*np.sqrt(2)/9 * (self.UVzz-self.UVxx)
        self.UHxz =  2*np.sqrt(2)/9 * (self.UVzz-self.UVxx)

        """##############################################################"""

        """  ONSITE  """

        for i, initial_atom in enumerate(self.crystal.atoms):
            for io, initial_orbital in enumerate(initial_atom.orbitals):
                self.model.set_onsite(initial_orbital.onsite, ind_i=(i*len(self.crystal.atoms)+io), mode='reset')
        


        """  HOPPING  """
                                                 
        self._M03(0,3)
        self._M12(1,2)
        
        self._M13(1,3)
        self._M02(0,2)
                
        
    def _M03(self, i, f):
        """ The M14 transition matrix designed by Kobayashi et Al."""
        self.model.set_hop(self.UVss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self.UVsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self.UVzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self.UVzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self.UVxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self.UVxx, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, -1], mode='reset')            
        
    def _M12(self, i, f):
        """ The M14 transition matrix designed by Kobayashi et Al."""
        self.model.set_hop(self.UVss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UVsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UVzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UVzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UVxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UVxx, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')  
              
    def _M02(self, i, f):
        """ The M13 transition matrix designed by Kobayashi et Al."""
        #s-s
        self.model.set_hop(self.UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self.UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #s-z
        self.model.set_hop(self.UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self.UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #s-x
        self.model.set_hop(     self.UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        #s-y
        self.model.set_hop( np.sqrt(3)/2*self.UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self.UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """##########################################"""
        #z-s
        self.model.set_hop(self.UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self.UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #z-z
        self.model.set_hop(self.UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self.UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #z-x
        self.model.set_hop(     self.UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        #z-y
        self.model.set_hop( np.sqrt(3)/2*self.UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self.UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """#########################################"""
        #x-s
        self.model.set_hop(     self.UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #x-z
        self.model.set_hop(     self.UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #x-x
        self.model.set_hop(     self.UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='add')        
        #x-y
        self.model.set_hop(-np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """###########################################"""
        #y-s
        self.model.set_hop( np.sqrt(3)/2*self.UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self.UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #y-z
        self.model.set_hop( np.sqrt(3)/2*self.UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self.UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #y-x
        self.model.set_hop(-np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')  
        #y-y
        self.model.set_hop(     self.UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='add')    

    def _M13(self, i, f):
        """ The M24 transition matrix designed by Kobayashi et Al."""
        #s-s
        self.model.set_hop(self.UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self.UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #s-z
        self.model.set_hop(self.UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self.UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #s-x
        self.model.set_hop(   -self.UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        #s-y
        self.model.set_hop(-np.sqrt(3)/2*self.UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self.UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """##########################################"""
        #z-s
        self.model.set_hop(self.UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self.UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #z-z
        self.model.set_hop(self.UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self.UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self.UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #z-x
        self.model.set_hop(   -self.UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        #z-y
        self.model.set_hop(-np.sqrt(3)/2*self.UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self.UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """#########################################"""
        #x-s
        self.model.set_hop(   -self.UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #x-z
        self.model.set_hop(   -self.UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self.UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #x-x
        self.model.set_hop(     self.UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 1, 0], mode='add')        
        #x-y
        self.model.set_hop(-np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """###########################################"""
        #y-s
        self.model.set_hop(-np.sqrt(3)/2*self.UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self.UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #y-z
        self.model.set_hop(-np.sqrt(3)/2*self.UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self.UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #y-x
        self.model.set_hop(-np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self.UHxx-self.UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')  
        #y-y
        self.model.set_hop(     self.UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self.UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self.UHxx+self.UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 1, 0], mode='add')        
    
    def f0(self,conjugate=False):
    
        conjugate = -2*conjugate+1

        f = np.array([
            [ 0,  0, 0],
            [-1,  0, 0],
            [-1, -1, 0]
        ])*conjugate

        w = np.array([
            1, 
            1, 
            1
        ])
        return f,w
    
    def f1(self,conjugate=False):
    
        conjugate = -2*conjugate+1

        f = np.array([
            [ 0,  0, 0],
            [-1,  0, 0],
            [-1, -1, 0]
        ])*conjugate

        w = np.array([
            1, 
            -1/2, 
            -1/2
        ])
        return f,w
    
    def f2(self,conjugate=False):
    
        conjugate = -2*conjugate+1

        f = np.array([
            [ 0,  0, 0],
        ])*conjugate

        w = np.array([
            1
        ])
        return f,w
    
    def fplus(self,conjugate=False):
    
        conjugate = -2*conjugate+1

        f = np.array([
            [-1,  0, 0],
            [-1, -1, 0],
        ])*conjugate

        w = np.array([
            1, 
            1
        ])
        return f,w
    
    
    def fminus(self,conjugate=False):
    
        conjugate = -2*conjugate+1

        f = np.array([
            [-1,  0, 0],
            [-1, -1, 0],
        ])*conjugate

        w = np.array([
            1, 
            -1
        ])
        return f,w
    
    def __repr__(self):
        return "Wursite SP3 Tight binding model for: \n \n {} \n".format(self.crystal)