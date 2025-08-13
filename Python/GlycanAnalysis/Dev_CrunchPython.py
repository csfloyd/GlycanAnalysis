#################################################
################  Import things #################
#################################################

# %% Basic imports
import numpy as np
import timeit
import random
import copy
import matplotlib.pyplot as plt
from sklearn import datasets
import pickle
import glycowork
import CandyCrunch

# %% ESM model setup
# need to previously `run pip3 install esm`
import esm.pretrained
from glycowork.ml.inference import get_esm1b_representations, get_lectin_preds
from glycowork.ml.models import prep_model
from glycowork.glycan_data.loader import glycan_binding as gb

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S() # will need to download if the first time

# %% Package installation and setup
#@markdown # Step 1: Install all neccessary packages

from IPython.utils.io import capture_output
import os
import subprocess

# from google.colab import drive
# drive.mount('/content/drive')

try:
    with capture_output() as captured:
        # Note: Shell commands should be run in terminal instead
        import glycowork.motif.draw
        from glycowork.motif.draw import GlycoDraw
        from CandyCrunch.prediction import *
        from CandyCrunch.analysis import *
        from IPython.core.display import display,HTML
        import imgkit
        from PIL import Image
        import bokeh
        import io
        import warnings
        import base64

except Exception as e:
    print(e)
    raise

# %% Your next code cell here
def prob_dist(vals):
    return vals / np.sum(vals)

def svg_to_base64(svg):
    svg_bytes = io.BytesIO(svg.encode())
    svg_data = svg_bytes.read()
    return 'data:image/svg+xml;base64,' + base64.b64encode(svg_data).decode('utf-8')

def format_name(name):
    return f'<span style="font-size: 16px">{name}</span>'

def format_main_image(main_image):
    return f'<div style="text-align: center"><img src="{svg_to_base64(main_image)}" /></div>'

def format_alt_images(alt_images):
    svg_list = []
    for alt_image in alt_images:
      svg_list.append(f'<div style="text-align: center; margin-bottom: 10px;"><img src="{svg_to_base64(alt_image)}" /></div>')
    return f''.join(svg_list)

def svg_list_to_html(svg_list):
    cell_html = ''
    for svg_file in svg_list:
        svg_base64 = svg_to_base64(svg_file)
        cell_html += f'<img src="data:image/svg+xml;base64,{svg_base64}" style="display:block; margin: 10px 0;">'
    return cell_html

new_style = """
<style>
table {
  border-collapse: collapse;
}

th {
  text-align: center;
  background-color: #f2f2f2;
  padding: 8px;
}

td {
  text-align: center;
  padding: 8px;
}

.image {
  text-align: center;
}

.image img {
  width: 200px;
  height: auto;
}

tr.start-page {
    page-break-before: always;
}
tr.end-page {
    page-break-after: always;
}
</style>
"""

# %%
## put in file and directory names
directory_address  = '/Users/csfloyd/Dropbox/Projects/GlycanAnalysis/Data/'  #@param {type:"string"}
filename = 'data/GPST000350/JC_200217P1N_200218002345.xlsx'  #@param {type:"string"}
# filename = '21091610_mouse_IgG_C14.mzML'
#filename = 'LC_Cardicola_forsteri_OG_1_BB1_01_2324.d.mzXML'


# ## load file
# spectra_filepath = ''
# if ".xlsx" in filename:
#   spectra_filepath = directory_address + filename
# else:
#   if  "." in filename:
#     filename = filename.split('.')[0]
#   input_address = directory_address + filename
#   for extension,extraction_function in [('.mzML',process_mzML_stack),('.mzXML',process_mzXML_stack)]:
#     if os.path.isfile(input_address + extension):
#       if not os.path.isfile(filename + extension + '.xlsx'):
#         extraction_function(input_address + extension,intensity=True).to_excel(filename+extension+'.xlsx',index=False)
#       spectra_filepath = filename + extension + '.xlsx'
#       break

spectra_filepath = directory_address + filename

### Select the model parameters:
glycan_class = 'O' #@param ["O", "N", "lipid", "free"]
mode = 'negative' #@param ["negative", "positive"]
liquid_chromatography = 'PGC' #@param ["PGC", "C18", "other"]
trap = 'linear' #@param ["linear", "orbitrap", "amazon", "ToF", "QToF", "other"]
modification = 'reduced' #@param ["reduced", "permethylated", "2AA", "2AB" , "custom"]
#@markdown ##### custom_modification_mass is only passed if modification is set to 'custom'
custom_modification_mass = 0 #@param {type: "number"}

if modification == 'custom':
  mass_tag = custom_modification_mass
else:
  mass_tag = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
df_out,spectra_out = wrap_inference(spectra_filepath, glycan_class,
                                    mode = mode, modification = modification, mass_tag = mass_tag, lc = liquid_chromatography, trap = trap,
                                    spectra=True,experimental=False,supplement=False)

glycan_pred_list = [x[0][0] if x else [] for x in df_out['predictions'].tolist()]
glycan_probs_list = [f'{round(x[0][1]*100,2)}%' if x else 'N/a' for x in df_out['predictions'].tolist()]
glycan_img_list = [GlycoDraw(x).as_svg() if x else '' for x in glycan_pred_list]

glycan_all_preds = []
for preds in df_out['predictions'].tolist():
  glycan_all_preds.append([x[0] for x in preds])
["<br>".join(x) for x in glycan_all_preds]
alt_preds = [x[1:] for x in glycan_all_preds]

glycan_all_probs = []
for preds in df_out['predictions'].tolist():
  glycan_all_probs.append([round(x[1]*100,2) for x in preds])
glycan_all_probs_string = [[str(x)+'%' for x in y] for y in glycan_all_probs]
["<br>".join(x) for x in glycan_all_probs_string]

display_df = df_out.reset_index()
display_df = display_df[[x for x in display_df.columns if x not in ['top_fragments','adduct']]]
display_df['predictions'] = glycan_pred_list
display_df['predicted_snfg'] = glycan_img_list
display_df['prediction_probability'] = glycan_probs_list
display_df = display_df.rename(columns={"index": "m/z","predictions": "predicted_IUPAC",})
if 'rel_abundance' in df_out:
  display_df = display_df.rename(columns={"rel_abundance": "abundance"})
  display_df = display_df[['m/z', 'composition', 'predicted_IUPAC', 'ppm_error', 'prediction_probability', 'num_spectra', 'charge', 'RT', 'abundance', 'predicted_snfg']]
  display_df['abundance'] = [round(x,2) for x in display_df['abundance'].tolist()]
else:
  display_df = display_df[['m/z', 'composition', 'predicted_IUPAC', 'ppm_error', 'prediction_probability', 'num_spectra', 'charge', 'RT','predicted_snfg']]
display_df.index.name = 'prediction ID'
display_df['m/z'] = [round(x,2) for x in display_df['m/z'].tolist()]
display_df['predicted_IUPAC'] = ["<br>".join(x) for x in glycan_all_preds]
display_df['prediction_probability'] = ["<br>".join(x) for x in glycan_all_probs_string]
display_df['alternative_snfg'] = [[GlycoDraw(x).as_svg() for x in shot if GlycoDraw(x)] for shot in alt_preds]



format_dict = {'predictions':format_name, 'predicted_snfg': format_main_image, 'alternative_snfg': format_alt_images}

html_table = display_df.to_html(escape=False, formatters=format_dict)
html_table = html_table.replace('<th>', '<th style="font-size: 20px">')



html_table = new_style + html_table


# Display HTML table
display(HTML(html_table))
