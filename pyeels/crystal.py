import numpy as np
import spglib as spg
from pyeels.atom import Atom
from pyeels.brillouinzone import BrillouinZone

class Crystal:
    def __init__(self, lattice):
        """ Initialize an instance of crystal with lattice hosting atom objects
        
        :type  lattice: ndarray
        :param lattice: a 3x3 array with lattice parameters [a,b,c]
        """

        self.atoms = []
        self.setLattice(lattice)
        
    def setLattice(self, lattice):
        """ Sett lattice parameters and calculate relevant properties 
        
        :type  lattice: ndarray
        :param lattice: a 3x3 array with lattice parameters [a,b,c]
        """
        self.lattice = lattice
        self.a = lattice[0]
        self.b = lattice[1]
        self.c = lattice[2]
        
        self._setVolume()
        self._setSpaceGroup()
        self._setBrillouinZone()
        
    def _setVolume(self):
        """ Calculate the volume of the crystal"""
        self.volume = np.dot(self.a, np.cross(self.b, self.c))
        
    def _setSpaceGroup(self):
        self.spacegroup = spg.get_spacegroup(
            (self.lattice,
             self.getAtomPositons(),
             self.getAtomNumbers()), 
             symprec=1e-5)
        
    def _setBrillouinZone(self):
        self.brillouinZone = BrillouinZone(self)
        
    def add_atom(self,atom):
        """ Add an atom object to the crystal
        :type  atom: atom object
        :param atom: the instance of atom to be placed in the crystal
        """
        existing = False
        for existing_atom in self.atoms:
            if np.all(atom.position == existing_atom.position):
                existing = True
        if not existing:
            atom.position = self._reduceCoordinate(atom.position)
            self.atoms.append(atom)
            self._setSpaceGroup()
            self._setBrillouinZone()
            return "Placed atom at {}".format(atom.position)
        else:
            _logger.warning("An atom is already at {}, try another coordinate.".format(atom.position))
    
    def getAtomNumbers(self):
        numbers = []
        for atom in self.atoms:
            numbers.append(atom.number)
        #if not numbers:
        #    _logger.warning("No atoms found, will give number 0 as a substitute")
        #    return [0]
        #else:
        return numbers

    def getAtomPositons(self):
        positions = []
        for atom in self.atoms:
            positions.append(atom.position)
        #if not positions:
        #    _logger.warning("No atoms found, will give position [0, 0, 0] as a substitute")
        #    return [[0, 0, 0]]
        #else:
        return positions         
            
    def _reduceCoordinate(self, coordinate):
        if np.any(coordinate>=np.array([1, 1, 1])) or np.any(coordinate<np.array([0, 0, 0])):
            _logger.warning("Coordinate {} oustide cell, reduces to a closer coordinate.".format(coordinate))
            coordinate = coordinate-(coordinate>=np.array([1, 1, 1]))*1
            coordinate = coordinate+(coordinate<np.array([0, 0, 0]))*1

            return self._reduceCoordinate(coordinate)
        else:
            return coordinate
            
    def __repr__(self):
        """ Representation of the crystal object """
        string = "CRYSTAL:\nSpacegroup: {}\n\nLattice:\n{}\nAtoms:\n".format(self.spacegroup,self.lattice)
        
        for i, atom in enumerate(self.atoms):
            string += "{}: {}\n".format(i,atom)
        return string