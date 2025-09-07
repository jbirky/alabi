# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://github.com/pradyunsg/furo/blob/main/src/furo/assets/styles/variables/_index.scss

# -- Path setup --------------------------------------------------------------

import os
import sys

# Check if we're building on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    # On ReadTheDocs, use the installed package
    import alabi
else:
    # For local builds, use the development version
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------

project = "alabi"
copyright = "2021, Jess Birky"
author = "Jess Birky"

# The full version, including alpha/beta/rc tags
release = "0.0.1"

html_theme = "furo"
html_title = "alabi"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "furo.sphinxext",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinx_togglebutton",
    # "sphinx_gallery.gen_gallery",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Mock imports for packages that might not be available during doc build
autodoc_mock_imports = ['george', 'emcee', 'dynesty', 'corner', 'h5py', 'tqdm', 'pybind11']

# nbsphinx specific settings
nbsphinx_execute = 'never'  # Never execute notebook cells during build
nbsphinx_allow_errors = True  # Continue through errors

# Control output display
nbsphinx_codecell_lexer = 'none'  # Don't highlight output as code

# Limit output length (truncate long outputs)
nbsphinx_output_max_lines = 10  # Show at most 10 lines of output

# Remove empty code cells from display
nbsphinx_remove_empty_cells = True

# Control which outputs to show
nbsphinx_output_stderr = False  # Don't show stderr outputs

# Custom CSS class for styling outputs
nbsphinx_code_css_class = 'nbsphinx-code-cell'

# Timeout for notebook execution (if executing)
nbsphinx_timeout = 60

# Disable problematic prolog/epilog for now
# nbsphinx_prolog = ""
# nbsphinx_epilog = ""

# Additional nbsphinx configuration
# nbsphinx_custom_formats = {
#     '.md': ['jupytext.reads', {'fmt': 'mystnb'}],
# }

# Disable notebook prolog/epilog that was causing ReadTheDocs build issues
# TODO: Fix template string concatenation issues and re-enable
# 
# Enable notebook downloads
# nbsphinx_prolog = r"""
# {% set docname = env.doc2path(env.docname, base=None) | string %}
# 
# .. only:: html
# 
#     .. role:: raw-html(raw)
#         :format: html
# 
#     .. nbinfo::
#         
#         This page was generated from `{{ docname }}`__.
#         {% if env.config.html_baseurl %}
#         Interactive online version:
#         :raw-html:`<a href="https://mybinder.org/v2/gh/jbirky/alabi/HEAD?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`
#         {% endif %}
# 
#     __ https://github.com/jbirky/alabi/blob/main/{{ docname|e }}
# 
# .. raw:: latex
# 
#     \nbsphinxstartnotebook{\scriptsize\noindent\strut
#     \textcolor{gray}{The following section was generated from
#     \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
# """

# Add download links for notebooks
# nbsphinx_epilog = r"""
# .. only:: html
# 
#     .. container:: sphx-glr-download sphx-glr-download-python
# 
#         :download:`Download Python source code: {{ env.docname.split('/')[-1] }}.py <{{ env.docname.split('/')[-1] }}.py>`
# 
#     .. container:: sphx-glr-download sphx-glr-download-jupyter
# 
#         :download:`Download Jupyter notebook: {{ env.docname.split('/')[-1] }}.ipynb <{{ env.docname.split('/')[-1] }}.ipynb>`
# """

# -- Options for HTML output -------------------------------------------------

html_static_path = ["_static"]

# Add custom CSS
html_css_files = [
    'custom.css',
]

# Add custom JavaScript
html_js_files = [
    'notebook-outputs.js',
]

html_theme_options = {
    "light_css_variables": {
        "font-stack": "Roboto Light, sans-serif",
        "font-stack--monospace": "Courier, monospace",
        "color-background-secondary": "#eff1f6", # sidebar color
        "color-inline-code-background": "#eff1f6", # inline code
        "color-sidebar-item-background--hover": "white", # sidebar highlight
        # "color-brand-primary": "#b32d00", # dark red, sidebar
        # "color-brand-content": "#b32d00", # dark red, main page
        "color-brand-primary": "#004080", # dark blue
        "color-brand-content": "#0059b3", # dark blue
        # "color-brand-primary": "#006600", # dark green
        # "color-brand-content": "#006600", # dark green
    },
}

# # Build example gallery
# sphinx_gallery_conf = {
#      "examples_dirs": "examples",   # path to your example scripts
#      "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
#      "filename_pattern": "/plot_",
#      "ignore_pattern": r"__init__\.py",
#      "download_all_examples": False,
#      "first_notebook_cell": "%matplotlib inline",  # Add matplotlib inline to notebooks
# }
