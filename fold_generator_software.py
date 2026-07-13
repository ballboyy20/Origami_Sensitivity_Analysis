# A compilation of functions used to generate .fold patterns
import matplotlib.pyplot as plt
from source.Bloom_Yoshimura import Bloom_Yoshimura
from source.Bloom_Yoshimura import generate_miura_fold

# Bloom Generation Code (don't edit)
def set_up_bloom(m=5,h=1,s=1,file_name=None, show_plot=None, Show_Origin=1, Show_Points=1, Show_facets=0, Show_Lines=1, Line_Width=1, Line_Style=1, Invert_Creases=0):
    Show_Origin = bool(Show_Origin)
    Show_Points = bool(Show_Points)
    Show_facets = bool(Show_facets)
    Show_Lines = bool(Show_Lines)
    Line_Width = float(Line_Width)
    Line_Style = bool(Line_Style)
    Invert_Creases = bool(Invert_Creases)

    bloom = Bloom_Yoshimura(m,h,s) # M,H,S
    bloom.plot_origin = Show_Origin
    bloom.plot_points = Show_Points
    bloom.plot_facets = Show_facets

    bloom.plot_lines = Show_Lines
    bloom.line_width = Line_Width
    bloom.line_style = Line_Style
    bloom.crease_is_invert = Invert_Creases
    

    bloom.graph()
    if show_plot is not None:
        plt.close('all')

    bloom.export_to_fold(filename=file_name)

