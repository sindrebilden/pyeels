import numpy as np
from pyeels.band import Band
import logging
_logger = logging.getLogger(__name__)

class BrillouinZone:
    def __init__(self, crystal):
        """ Initialize an instance of Brillouin Zone with reciprocal lattice hosting band objects
        
        :type  lattice: ndarray
        :param lattice: a 3x3 array with lattice parameters [a*,b*,c*]
        """
        self._set_reciprocal_space(crystal)
        
        self.bands = []

    def _set_reciprocal_space(self, crystal):
        """ Constructs the reciprocal space from a crystal objec
        
        :type  crystal: crystal object
        :param crystal: the crystal object containing real space information
        """
        self.a = np.cross(crystal.b, crystal.c) / crystal.volume * 2*np.pi #reciprocal aangstrom
        self.b = np.cross(crystal.c, crystal.a) / crystal.volume * 2*np.pi #reciprocal aangstrom
        self.c = np.cross(crystal.a, crystal.b) / crystal.volume * 2*np.pi #reciprocal aangstrom

        self.lattice = np.vstack([self.a, self.b, self.c])

    def add_band(self,band):
        """ Add a band object to the Brillouin Zone
        
        :type  atom: band object
        :param atom: the instance of atom to be placed in the crystal
        """
        existing = False
        for existing_band in self.bands:
            if band == existing_band:
                existing = True
        if not existing:
            self.bands.append(band)
        else:
            raise _logger.warning("An equal band already exist")
            
    def __repr__(self):
        """ Representation of the Brillouin Zone object """
        return "Brillouin Zone\nLattice:\n{}\n\nBands:\n{}".format(self.lattice, self.bands)
    