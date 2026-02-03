"""
Guides to .obj files:
 	- https://all3dp.com/2/obj-file-format-simply-explained/
	- https://en.wikipedia.org/wiki/Wavefront_.obj_file
	- https://paulbourke.net/dataformats/obj/

"""

class OrigamiContainer:
    def __init__(self):
        self._origami_filepath = None
        self._origami_pyrepr = None

    @staticmethod
    def extract_file(origami_filepath):
        origamiContainer = OrigamiContainer
        origamiContainer._origami_filepath = origami_filepath
        return origamiContainer

    @staticmethod
    def extract_pyrepr(origami_pyrepr):
        """
        Use the origami representation native to python
        and used by the SensitivityAnalysis code.
        
        :param origami_pyrepr: Description
        """
        origamiContainer = OrigamiContainer
        origamiContainer._origami_pyrepr = origami_pyrepr
        return origamiContainer
    
    def _interpret_obj(self):
        return
    
    def _interpret_fold(self):
        return
    
    def _interpret_dxf(self):
        return
    
    def _interpret_json(self):
        return
    
    def get_pyrepr(self):
        coords = None
        indices = None
        return coords, indices
    
    def export_obj(self):
        return
    
    def export_fold(self):
        return

    def export_dxf(self):
        return
    
    def export_json(self):
        return
    
