import logging
_logger = logging.getLogger(__name__)

class Orbital(object):
    """ Orbital object with a onsite energy """
    def __init__(self, onsite, label):
        """
        Initialize an instance of orbital
        
        :type  onsite: float
        :param onsite: The onsite energy of the orbital
        
        :type  label: string
        :param label: The name of the orbital (examples: s,p,d,f)
        """
        
        self.onsite = onsite
        self.label = label

    def __repr__(self):
        """ Representation of the orbital object """
        return "{} with onsite {}\n".format(self.label, self.onsite)