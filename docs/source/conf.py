# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# https://github.com/pradyunsg/furo/blob/main/src/furo/assets/styles/variables/_index.scss

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../../alabi'))


# -- Project information -----------------------------------------------------

project = 'alabi'
copyright = '2021, Jessica Birky'
author = 'Jessica Birky'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

html_theme = "furo"
html_title = "alabi"

# -- General configuration ---------------------------------------------------

extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # Our custom extension, only meant for Furo's own documentation.
    "furo.sphinxext",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinx_togglebutton",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_static_path = ['_static']

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

# Build example gallery
sphinx_gallery_conf = {
     'examples_dirs': '../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'filename_pattern': '/plot_',
     'ignore_pattern': r'__init__\.py',
     'download_all_examples': False,
}