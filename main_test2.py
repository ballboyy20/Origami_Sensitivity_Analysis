#%%
from flask import json
import numpy as np
import matplotlib.pyplot as plt

from source.SensitivityAnalysis import SensitivityModel
from source.Bloom_Yoshimura import Bloom_Yoshimura
from source.visualization import plot_sensitivity_violin, plot_fold_pattern
from source.Bloom_Yoshimura import generate_miura_fold

# --- Pattern Generation Setup ---
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



### --- Flashers --- ###
#%% Flashers - Model T
filename_r1_h2_m5_flasher = "r1_h2_m5_flasher.fold"
r1_h2_m5_flasher = SensitivityModel(filename_r1_h2_m5_flasher)
r1_h2_m5_flasher.analyze_sensitivity(show_plot='yes', show_colorbar=True, save_path="figures/r1_h2_m5_flasher_sensitivity.pdf")
# r1_h2_m5_flasher.step_and_reanalyze(step_scale=0.00000005, show_plot='yes')
plt.close('all')

#%%
filename_r1_h1_m6_flasher = "r1_h1_m6_flasher.fold"
r1_h1_m6_flasher = SensitivityModel(filename_r1_h1_m6_flasher)
r1_h1_m6_flasher.analyze_sensitivity(show_plot='yes', show_colorbar=True, save_path="figures/r1_h1_m6_flasher_sensitivity.pdf")
plt.close('all') 

#%%
filename_r1_h2_m6_flasher = "r1_h2_m6_flasher.fold"
r1_h2_m6_flasher = SensitivityModel(filename_r1_h2_m6_flasher)
r1_h2_m6_flasher.analyze_sensitivity(show_plot='yes', show_colorbar=True, save_path="figures/r1_h2_m6_flasher_sensitivity.pdf")
plt.close('all') 

### --- Blooms --- ###
#%% Bloom Yoshimura Y6_1
filename_Y6_1 = "Y6_1.fold"
set_up_bloom(m=6,h=1,s=1,file_name=filename_Y6_1) 
Y6_1 = SensitivityModel(filename_Y6_1)
Y6_1.analyze_sensitivity(show_plot='yes', show_colorbar=True, save_path="figures/Y6_1_sensitivity.pdf")
# Y6_1.check_integration_rigidity(num_steps=500, step_size= 0.01)
# Y6_1.animate_nonlinear_folding(num_steps=1000, step_size=0.01,interval=10)
plt.close('all')

#%% Bloom Yoshimura Y5_1 & Y6_2
filename_Y5_1 = "Y5_1.fold"
set_up_bloom(m=5,h=1,s=1,file_name=filename_Y5_1) 
Y5_1 = SensitivityModel(filename_Y5_1)
Y5_1.analyze_sensitivity(show_plot='yes', show_colorbar=True, save_path="figures/Y5_1_sensitivity.pdf")
plt.close('all')

#%%
filename_Y6_2 = "Y6_2.fold"
set_up_bloom(m=6,h=2,s=1,file_name=filename_Y6_2) 
Y6_2 = SensitivityModel(filename_Y6_2)
Y6_2.analyze_sensitivity(show_plot='yes', show_colorbar=True, save_path="figures/Y6_2_sensitivity.pdf")
# Y6_2.check_integration_rigidity(num_steps=500, step_size= 0.01)
Y6_2.animate_nonlinear_folding(num_steps=500, step_size=0.01)
plt.close('all') 

#%%
### --- Validation --- ###
#%% Birdsfoot
filename = "birdsfoot4.fold"
birdsfoot_model = SensitivityModel(filename)
birdsfoot_model.analyze_sensitivity(show_plot='yes')
birdsfoot_model.step_and_reanalyze(step_scale=0.0005, show_plot='yes')
# birdsfoot_model.check_integration_rigidity(num_steps=200, step_size=0.01)
# birdsfoot_model.animate_nonlinear_folding(num_steps=200, step_size=0.01, interval=100)

#%% Miura-ori
filename = "miura_ori.fold"
fold_data = generate_miura_fold(cols=5, rows=5, dx=20.0, dy=20.0, tilt=10.0)
with open(filename, "w") as f:   
    json.dump(fold_data, f, indent=2)

miura_ori = SensitivityModel(filename)
miura_ori.analyze_sensitivity(show_plot='yes', save_path="figures/miura_ori_sensitivity.pdf")
miura_ori.step_and_reanalyze(step_scale=0.0005, show_plot='yes')




# %%
