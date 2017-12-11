import numpy as np
from pyeels.orbital import Orbital
import logging
_logger = logging.getLogger(__name__)

class Atom(object):
    """ Atom object hosting orbitals """
    def __init__(self, position=np.array([0, 0, 0]), number=0):
        """ Initialize an instance of atom
        
        :type  position: ndarray, list, tuple
        :param position: The onsite energy of the orbital
        
        :type  number: int
        :param number: The atomic number of the atom (Z)
        """
        position = np.asarray(position)
        self.number = number
        self.position = position
        self.orbitals = []
    
    def add_orbital(self, orbital):
        """ Add an orbital to the hosting atom
        :type  orbital: orbital object
        :param orbital: an orbital in the atom
        """
        existing = False
        for existing_orbital in self.orbitals:
            if orbital.label == existing_orbital.label:
                existing = True
        if not existing:
            self.orbitals.append(orbital)
            return "Placed {} orbital with onsite {}, on {}".format(orbital.label,orbital.onsite, self)
        else:
            _logger.warning("An orbital with label {} allready exist, try different label.".format(orbital.label))
    
    def set_position(self, position):
        """ Set the position of the atom
        :type  position: ndarray, list, tuple
        :param position: The onsite energy of the orbital
        """
        self.position = position

    def set_number(self, number):
        """ Set the atomic number
        :type  number: int
        :param number: The atomic number of the atom (Z)
        """
        self.position = position
        
    def number_of_orbitals(self):
        """ Returns the number of orbitals in the atom 
        :returns: the number of orbitals in the atom
        """
        return len(self.orbitals)     
    
    def __repr__(self):
        """ Representation of the Atom object """

        return "Atom with Z={} at {}".format(self.number, self.position)
    
class Oxygen(Atom):
    """ A special case of Atom designed for Oxygen"""
    def __init__(self, position):
        Atom.__init__(self, position, 8)
        self.add_orbital(Orbital(-19.046, "S"))
        self.add_orbital(Orbital(  4.142, "Pz"))
        self.add_orbital(Orbital(  4.142, "Px"))
        self.add_orbital(Orbital(  4.142, "Py"))
    
    def __repr__(self):
        return "Oxygen at {}".format(self.position)
    
class Zinc(Atom):
    """ A special case of Atom designed for Zinc"""
    def __init__(self, position):
        
        Atom.__init__(self, position, 30)
        self.add_orbital(Orbital(  1.666, "S"))
        self.add_orbital(Orbital( 12.368, "Pz"))
        self.add_orbital(Orbital( 12.368, "Px"))
        self.add_orbital(Orbital( 12.368, "Py"))
        
    def __repr__(self):
        return "Zinc at {}".format(self.position)
    

class Gallium(Atom):
    """ A special case of Atom designed for Gallium"""
    def __init__(self, position):
        
        Atom.__init__(self, position, 30)
        self.add_orbital(Orbital(  1.438, "S"))
        self.add_orbital(Orbital( 10.896, "Pz"))
        self.add_orbital(Orbital( 10.896, "Px"))
        self.add_orbital(Orbital( 10.896, "Py"))
    
    def __repr__(self):
        return "Gallium at {}".format(self.position)
    
class Nitrogen(Atom):
    """ A special case of Atom designed for Nitrogen"""
    def __init__(self, position):
        
        Atom.__init__(self, position, 30)
        self.add_orbital(Orbital(-11.012, "S"))
        self.add_orbital(Orbital(  0.005, "Pz"))
        self.add_orbital(Orbital(  0.005, "Px"))
        self.add_orbital(Orbital(  0.005, "Py"))
        
    def __repr__(self):
        return "Nitrogen at {}".format(self.position)   