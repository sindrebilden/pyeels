from pyeels.crystal import Crystal
from pyeels.atom import Atom
from pyeels.band import Band
import matplotlib.pyplot as plt
import spglib as spg
import pythtb as tb
import numpy as np
from scipy.optimize import minimize
import logging
from matplotlib.colors import to_rgb, hsv_to_rgb, rgb_to_hsv
_logger = logging.getLogger(__name__)

class TightBinding:
    """ Tight binding class constructed around the 'PythTB <http://physics.rutgers.edu/pythtb/>' package """
    def __init__(self, crystal):
        """ Create instance of the Tight binding model 
        
        :type  crystal: crystal object
        :param crystal: a crystal object containing atoms
        """
        
        self._crystal = crystal
        self._crystal.brillouinzone.band_model = "Tight Binding"
        self._spg = (crystal.lattice, crystal.get_atom_positons(), crystal.get_atom_numbers())
        
        
        if not hasattr(self, 'wannier'):
            self._orbital_positons = []
            for atom in crystal.atoms:
                for orbital in atom.orbitals:
                    self._orbital_positons.append(atom.position)        
            
            self.setGrid()

            self.model = tb.tb_model(3,3,self._crystal.lattice, self._orbital_positons)



    @classmethod
    def check_class(cls):
        return cls.__bases__ == TightBinding.__bases__
        
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
            for i in range(len(mesh)):
                if (mesh[i]%2==0):
                    mesh[i] += 1

            self._crystal.brillouinzone.mesh = mesh
            mapping, grid = spg.get_ir_reciprocal_mesh(mesh, self._spg, is_shift=[0, 0, 0])
            
            if np.any(mesh==np.array([1, 1, 1])):
                mesh+= (mesh==np.array([1, 1, 1]))*1

            self._k_grid = grid/(mesh-1)
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
        self._crystal.brillouinzone.bands = []
        
        if eig_vectors:
            energies, waves = self.model.solve_all(self._k_grid,eig_vectors=eig_vectors)
        else:
            energies = self.model.solve_all(self._k_grid,eig_vectors=eig_vectors)
            waves = np.stack([np.zeros(energies.shape,dtype=np.complex128),np.ones(energies.shape,dtype=np.complex128)], axis=2)
        
        for i, band in enumerate(energies):
            self._crystal.brillouinzone.add_band(Band(k_grid=self._k_grid, energies=band, waves=waves[i]))
    

    def set_orbital_colors(self, orbitals=None, colors=None):
        """ Create a set of colors representing the orbital bais set
        
        :type  orbitals: list
        :param orbitals: a list of orbital symbols/numbers, identical symbols/numbers get identical color

        :type  colors: list
        :param colors: a list of chosen colors, if left None a set will be generated
        """
        
        if isinstance(orbitals, type(None)):
            orbitals = list(range(len(self._orbital_positons)))
            numbers = orbitals
            symbols = []
            for orbital in orbitals:
                symbols.append('Orbital_{}'.format(orbital))

        else:
            symbols = []
            for orbital in orbitals:
                if not orbital in symbols:
                    symbols.append(orbital)

            numbers = []
            for i, orbital in enumerate(orbitals):
                for j, symbol in enumerate(symbols):
                    if orbital is symbol:
                        numbers.append(j)


        if isinstance(colors, type(None)):
            unique_colors = []
            for i in range(0,len(symbols)):
                print((len(symbols)-i)/(len(symbols)))
                unique_colors.append(((len(symbols)-i)/(len(symbols)) , 1, 1 ))
                
            colors = []
            for number in numbers:
                colors.append(hsv_to_rgb(unique_colors[number]))
        else:
            unique_colors = colors
            if len(colors) < len(symbols):
                raise ValueError("The length of colors is shorter than the number of unique symbos")
            else:
                colors = []
                for number in numbers:
                    colors.append(to_rgb(unique_colors[number]))
        
        self.type_symbols = symbols
        self.type_numbers = numbers
        self.type_colors = colors = np.asarray(colors)

        return symbols, numbers,  colors
    
        

    def bandstructure(self, path, labels, point_density=301, ylim=(None,None),  bands=(None,None), color=None, linestyle=None, marker=None, markersize=10, markevery=None, ax=None):
        """ Plot a representation of the band structure
        
        :type  ylim: tuple, list
        :param ylim: lower and upper limit of y-values (ymin,ymax)
        :returns: figure, ax
        """

        """ seekpath automatic lines"""
        #path = sp.get_explicit_k_path((lattice, positions, numbers), False, recipe="hpkot", threshold=1e-5,reference_distance=1)
        #expath = path['explicit_kpoints_abs'][:5]
        #labels = path['explicit_kpoints_labels'][:5]

        """ manual lines"""
#        path=[[0.0,0.0,0.5],[0.5,0.0,0.5],[0.5,0,0.0],[0.0,0.0,0.0],[0,0,0.5],[2./3.,1./3.,0.5],[2./3.,1./3.,0.0],[0,0,0]]
#        label=(r'$A $',      r'$L$',       r'$M$',   r'$\Gamma$', r'$A $', r'$H$',  r'$K$',r'$\Gamma $')

        
        # call function k_path to construct the actual path
        (k_vec,k_dist,k_node)=self.model.k_path(path, point_density, report=False)

        if (color=='point_type') or (color=='band_type'):
            evals, evec = self.model.solve_all(k_vec, True)

            if not hasattr(self, 'type_colors'):
                self.set_orbital_colors()
            
        else:
            evals = self.model.solve_all(k_vec)
        
        fig = None
        if not ax:
            fig, ax = plt.subplots(figsize=(8,6))
            fig.tight_layout()

        #ax.set_title("Bandstructure for Zno based on Kobayashi")
        ax.set_ylabel("Band energy")

        # specify horizontal axis details
        ax.set_xlim([0,k_node[-1]])
        # put tickmarks and labels at node positions
        ax.set_xticks(k_node)
        ax.set_xticklabels(labels)
        # add vertical lines at node positions

        for n in range(len(k_node)):
            if labels[n] == r'$\Gamma$':
                ax.axvline(x=k_node[n],linewidth=1, color='k')
            else:
                ax.axvline(x=k_node[n],linewidth=0.5, color='k')
    
        maximum = np.array([1,1,1])

        if (color=='point_type') or (color=='band_type'):
            """ If coloring by type, the color is calculated by the wave nature of the points/bands"""

            for band, wave in zip(evals[bands[0]:bands[1]],evec[bands[0]:bands[1]]):
                if (color=='band_type'):  
                    rgb = np.dot((np.absolute(wave)**2).mean(axis=0).round(2),self.type_colors)
                    ax.plot(k_dist, band, color=(np.minimum(maximum, rgb)), linestyle=linestyle, marker=marker)

                elif (color=='point_type'):
                    for k, e, w in zip(k_dist, band, wave):
                        rgb = np.dot((np.absolute(w)**2).round(2),self.type_colors)
                        ax.scatter(k,e, color=(np.minimum(maximum, rgb)), s=markersize**2)
        else:
            for band in evals[bands[0]:bands[1]]:
                ax.plot(k_dist, band, color=color, linestyle=linestyle, marker=marker, markersize=markersize, markevery=markevery)

        if not fig:
            return ax
        else:
            ax.set_ylim(ylim)
            return fig, ax

    def density_of_states(self, energybins):
        """ calculates the density of states (DOS) in the material
        :type  energybins: ndarray
        :param energybins: numpy array of the energy bins for DOS
        """

        DOS = np.zeros(energybins.shape)

        if len(self._crystal.brillouinzone.bands) > 0:
            for band in self._crystal.brillouinzone.bands:
                DOS += band.density_of_states(energybins)
            return DOS
        else:
            raise ValueError("No bands found in crystal, run calculate() to calculate bands.")

    def __repr__(self):
        return "Parabolic band model for: \n \n {} \n".format(self._crystal)


class Wannier(TightBinding):
    """ A wannier interface for the TightBinding class"""

    def __init__(self, path, prefix, zero_level=0):
        self.path = path
        self.prefix = prefix
        self.wannier = tb.w90(path, prefix)
        self.zero_level = zero_level

        self._orbital_positons = []
        for orbital_position in self.wannier.red_cen:
            self._orbital_positons.append(list(orbital_position))

        TightBinding.__init__(self, self.create_crystal())       

        self.set_model(zero_level=self.zero_level)

    def set_model(self, zero_level=None, min_hopping_norm=0.01, max_distance=None, ignorable_imaginary_part=0.01):

        if zero_level:
            self.zero_level = zero_level

        self.model = self.wannier.model(zero_energy=zero_level, min_hopping_norm=min_hopping_norm, max_distance=max_distance, ignorable_imaginary_part=ignorable_imaginary_part)


    def create_crystal(self):

        wannier_crystal = Crystal(self.wannier.lat)        

        with open("{}{}_centres.xyz".format(self.path,self.prefix)) as file:
            num_orbitals = int(file.readline())
            #Skip orbitals
            for i in range(0,num_orbitals+1):
                file.readline()
            
            for line in file.readlines():
                (atom, x,y,z) = line.strip('\n').split('       ')
                
                number = Atom._ATOMS.index(atom)

                position = np.asarray([x,y,z]).astype(float)
                position = np.dot(position,np.linalg.inv(self.wannier.lat))

                wannier_crystal.add_atom(Atom(position=position, number=number))

        return wannier_crystal

    def shells(self):
        """ Return all pair distances between the orbitals
        Directly taken from PythTB wannier example

        :returns: wannier shells
        """
        return self.wannier.shells()

    def hoppings(self):
        """ Plot hopping terms as a function of distance on a log scale 
        Directly taken from PythTB wannier example

        :returns: figure of hopping terms
        """
        (dist,ham) = wannier.dist_hop()
        fig, ax = plt.subplots()
        ax.scatter(dist,np.log(np.abs(ham)))
        ax.set_xlabel("Distance (A)")
        ax.set_ylabel(r"$\log H$ (eV)")
        fig.tight_layout()


class WurtziteSP3(TightBinding):
    """ A wurtzite Tight Binding model that includes s,px,py,pz orbitals at each site.
    Model designed by 'Kobayashi et Al. 1983 <https://link.aps.org/doi/10.1103/PhysRevB.28.935>'"""
    def __init__(self, crystal):
        TightBinding.__init__(self, crystal)

        self.update_onsites()



    def get_parameters(self):
        """ Get a list of the parameters of the wurtzite model

        :returns: list of [Esa, Epa, Esc, Epc, Vss, Vxx, Vxy, Vsapc, Vpasc]
        """
        return self.get_onsites() + self.get_hopping_parameters()

    def set_parameters(self, Esa, Epa, Esc, Epc, Vss, Vxx, Vxy, Vsapc, Vpasc):
        """ Set all parameters of the wurtzite model

        :type  Esa: float
        :param Esa: the onsite energy of s  of the anion
        :type  Epa: float
        :param Epa: the onsite energy of p of the anion

        :type  Esc: float
        :param Esc: the onsite energy of s  of the cation
        :type  Epc: float
        :param Epc: the onsite energy of p of the cation

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

        self.set_onsites(Esa, Epa, Esc, Epc)
        self.set_hopping_parameters(Vss, Vxx, Vxy, Vsapc, Vpasc)
        
    def get_onsites(self):
        """ Get a list of the onset energies of the wurtzite model

        :returns: list of [Esa, Epa, Esc, Epc]
        """

        ani = [0, 0, 0, 0]
        cat = [0, 0, 0, 0]

        onsites = [ani, ani, cat, cat]

        for i, initial_atom in enumerate(self._crystal.atoms):
            for io, initial_orbital in enumerate(initial_atom.orbitals):
                onsites[i][io] = initial_orbital.onsite

        return ani[:2]+cat[:2]


    def set_onsites(self, Esa, Epa, Esc, Epc):
        """  Set all onsite energies in the wurtzite model and update them afterwards
        :type   Esa: float
        :param  Esa: the onsite energy of s  of the anion

        :type  Epa: float
        :param Epa: the onsite energy of p of the anion

        :type   Esc: float
        :param  Esc: the onsite energy of s  of the cation

        :type  Epc: float
        :param Epc: the onsite energy of p of the cation
         """

        ani = [Esa, Epa, Epa, Epa]
        cat = [Esc, Epc, Epc, Epc]

        onsites = [ani, ani, cat, cat]

        for i, initial_atom in enumerate(self._crystal.atoms):
            for io, initial_orbital in enumerate(initial_atom.orbitals):
                initial_orbital.onsite = onsites[i][io]

        self.update_onsites()

    def update_onsites(self):
        """ Update the onsite parameters from the orbital objects"""
        for i, initial_atom in enumerate(self._crystal.atoms):
            for io, initial_orbital in enumerate(initial_atom.orbitals):
                self.model.set_onsite(initial_orbital.onsite, ind_i=(i*len(self._crystal.atoms)+io), mode='reset')
        


    def get_hopping_parameters(self):
        """ Get a list of all hopping parameters in the model
        :returns: Vss, Vxx, Vxy, Vsapc, Vpasc. See :func:'set_hopping_parameters' for info."""
        return [self._Vss, self._Vxx, self._Vxy, self._Vsapc, self._Vpasc ]
        
        
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
        self._Vss = Vss
        self._Vxx = Vxx
        self._Vxy = Vxy
        self._Vsapc = Vsapc
        self._Vpasc = Vpasc
        
        """###############     Vertical bonding system    ##################"""      
        self._UVss = 0.25*Vss
        self._UVzz = 0.25*(Vxx+2*Vxy)
        self._UVxx = 0.25*(Vxx-Vxy)
        self._UVsz = -0.25*np.sqrt(3)*Vsapc
        self._UVzs =  0.25*np.sqrt(3)*Vpasc
        
        
        """###############    Horizontal bonding system   ##################"""
        self._UHss = self._UVss
        
        self._UHyy = self._UVxx
        self._UHzz = 1/9 * (8*self._UVxx +   self._UVzz)
        self._UHxx = 1/9 * (  self._UVxx + 8*self._UVzz)
        
        self._UHsz = -1/3 * self._UVsz
        self._UHzs = -1/3 * self._UVzs
        
        self._UHsx = -2*np.sqrt(2)/3 * self._UVsz
        self._UHxs = -2*np.sqrt(2)/3 * self._UVzs
        
        self._UHzx =  2*np.sqrt(2)/9 * (self._UVzz-self._UVxx)
        self._UHxz =  2*np.sqrt(2)/9 * (self._UVzz-self._UVxx)

        """##############################################################"""

        """  HOPPING  """
                                                 
        self._M03(0,3)
        self._M12(1,2)
        
        self._M13(1,3)
        self._M02(0,2)
                
        
    def _M03(self, i, f):
        """ The M14 transition matrix designed by Kobayashi et Al."""
        self.model.set_hop(self._UVss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, -1], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, -1], mode='reset')            
        
    def _M12(self, i, f):
        """ The M14 transition matrix designed by Kobayashi et Al."""
        self.model.set_hop(self._UVss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UVxx, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')  
              
    def _M02(self, i, f):
        """ The M13 transition matrix designed by Kobayashi et Al."""
        #s-s
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #s-z
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #s-x
        self.model.set_hop(     self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        #s-y
        self.model.set_hop( np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """##########################################"""
        #z-s
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #z-z
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #z-x
        self.model.set_hop(     self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        #z-y
        self.model.set_hop( np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """#########################################"""
        #x-s
        self.model.set_hop(     self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #x-z
        self.model.set_hop(     self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #x-x
        self.model.set_hop(     self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='add')        
        #x-y
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        """###########################################"""
        #y-s
        self.model.set_hop( np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[-1, -1, 0], mode='reset')
        #y-z
        self.model.set_hop( np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[-1, -1, 0], mode='reset')
        #y-x
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[-1, -1, 0], mode='reset')  
        #y-y
        self.model.set_hop(     self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[-1, -1, 0], mode='add')    

    def _M13(self, i, f):
        """ The M24 transition matrix designed by Kobayashi et Al."""
        #s-s
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHss, ind_i=i*4+0, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #s-z
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHsz, ind_i=i*4+0, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #s-x
        self.model.set_hop(   -self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHsx, ind_i=i*4+0, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        #s-y
        self.model.set_hop(-np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHsx, ind_i=i*4+0, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """##########################################"""
        #z-s
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzs, ind_i=i*4+1, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #z-z
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(self._UHzz, ind_i=i*4+1, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #z-x
        self.model.set_hop(   -self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHzx, ind_i=i*4+1, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        #z-y
        self.model.set_hop(-np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHzx, ind_i=i*4+1, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """#########################################"""
        #x-s
        self.model.set_hop(   -self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxs, ind_i=i*4+2, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #x-z
        self.model.set_hop(   -self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(0.5*self._UHxz, ind_i=i*4+2, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #x-x
        self.model.set_hop(     self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHxx, ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+2, ind_j=f*4+2, ind_R=[1, 1, 0], mode='add')        
        #x-y
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+2, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        """###########################################"""
        #y-s
        self.model.set_hop(-np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHxs, ind_i=i*4+3, ind_j=f*4+0, ind_R=[1, 1, 0], mode='reset')
        #y-z
        self.model.set_hop(-np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/2*self._UHxz, ind_i=i*4+3, ind_j=f*4+1, ind_R=[1, 1, 0], mode='reset')
        #y-x
        self.model.set_hop(-np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop( np.sqrt(3)/4*(self._UHxx-self._UHyy), ind_i=i*4+3, ind_j=f*4+2, ind_R=[1, 1, 0], mode='reset')  
        #y-y
        self.model.set_hop(     self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[0, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 0, 0], mode='reset')
        self.model.set_hop(-0.5*self._UHyy, ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 1, 0], mode='reset')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 0, 0], mode='add')
        self.model.set_hop(3/4*(self._UHxx+self._UHyy), ind_i=i*4+3, ind_j=f*4+3, ind_R=[1, 1, 0], mode='add')        
    
    
    def __repr__(self):
        return "Wursite SP3 Tight binding model for: \n \n {} \n".format(self._crystal)

class TbFitter():
    """ A Tight Binding Fitting class 
    Based on a class written by Lars Musland
    """
    mu = 0
    T = 1
    def __init__(self, TB_model, fitting_k, fitting_E, band_range=(0,-1), monitor=False, tolerance=None):
        """ Create instance of the fitting class
        
        :type  mode: TightBinding instance
        :param mode: the Tight Binding model
        :type  fitting_k: np.ndarray
        :param fitting_k: the k-points for fitting reference
        :type  fitting_E: np.ndarray
        :param fitting_E: the reference energies at the k-points
        :type  band_range: tuple
        :param band_range: lowest and highest band index in a tuple
        :type  monitor: boolean
        :param monitor: If true, the standard deviation is printed for each iteration
        :type  tolerance: float
        :param tolerance: the highest standard deviation accepted
        """
        self.TB_model = TB_model
        self.monitor = monitor
        self.fitting_k = fitting_k
        self.fitting_E = fitting_E
        self.tolerance = tolerance
        self.band_range = band_range
        
    def fit(self, mu=None, T=None, monitor=None):
        """ Perform the fitting, the optimized fitting parameters is left in the TB-model
        
        :type  mu: float
        :param mu: the center energy of the weighting function 
        :type   T: float
        :param  T: the range factor of the weighting function
        :type  monitor: boolean
        :param monitor: If true, the standard deviation is printed for each iteration

        :returns: passes the result of the minimize() function
        """
        if mu:
            self.mu = mu
        if T:
            self.T = T
            
        if not isinstance(monitor,type(None)):
            self.monitor = monitor
        initial_args = self.TB_model.get_parameters()
        
        return minimize(self.fit_function, initial_args, tol=self.tolerance)
        
    def fit_function(self, args):
        """ Fitting function passed to the minimize() function in self.fit()        

        :type  args: tuple
        :param args: the arguments of the Tight Binding model
        """
        Esa, Epa, Esc, Epc, Vss, Vxx, Vxy, Vsapc, Vpasc = args
        self.TB_model.set_parameters(Esa, Epa, Esc, Epc, Vss, Vxx, Vxy, Vsapc, Vpasc)

        E =self.TB_model.model.solve_all(self.fitting_k)[self.band_range[0]:self.band_range[1]]
    
        
        diff=self.fitting_E-E

        diff=abs(diff)**2
        
        val=sum((diff*self.weightfun(self.fitting_E,E)).ravel())
        
        if self.monitor:
            print(val)
            
        self.lastval=val
        self.weightsum=sum((self.weightfun(self.fitting_E,E)).ravel())
        return val
    
    def weightfun(self,E_ref,E_calc):
        """ A weight function for reference and calculated energies 
        
        :type   E_ref: np.ndarray
        :param  E_ref: the reference energies

        :type  E_calc: np.ndarray
        :param E_calc: the calculated energies
        """
        w_ref=1./np.cosh((E_ref-self.mu)/self.T)**2/self.T/8
        w_calc=1./np.cosh((E_calc-self.mu)/self.T)**2/self.T/8
        return w_ref+w_calc