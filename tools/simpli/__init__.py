"""
Defiens default tasks,
makes ~/.simpli/jsons/default_tasks.json, and
initializes simpli as an Jupyter extension.
"""
from os import listdir, remove, symlink
from os.path import dirname, islink, join, realpath, split

from IPython.core.display import display_html

from .support import establish_filepath, get_home_dir


# ==============================================================================
# Default tasks
# ==============================================================================
def just_return(value):
    """
    Just return.
    :param value:
    :return: obj
    """

    return value


def link_json(filepath):
    """
    Soft link JSON filepath to $HOME/.simpli/jsons/ directory.
    :param filepath: str; JSON filepath
    :return: None
    """

    destination = join(SIMPLI_JSON_DIR, split(filepath)[1])
    if islink(destination):
        remove(destination)
    symlink(filepath, destination)


def reset_jsons():
    """
    Delete all files except default_tasks.json in $HOME/.simpli/jsons/ directory.
    :return: None
    """

    for f in listdir(SIMPLI_JSON_DIR):
        if f != 'default_tasks.json':
            remove(join(SIMPLI_JSON_DIR, f))


def set_notebook_theme(filepath):
    """
    Set notebooks theme.
    :param filepath: str; .css
    :return: None
    """

    html = """<style> {} </style>""".format(open(filepath, 'r').read())
    display_raw_html(html)


def display_raw_html(html, hide_input_cell=True):
    """
    Execute raw HTML.
    :param html: str; HTML
    :param hide_input_cell: bool;
    :return: None
    """

    if hide_input_cell:
        html += """<script> $('div .input').hide()"""
    display_html(html, raw=True)


# ==============================================================================
# Link ~/.simpli/jsons/default_tasks.json
# ==============================================================================
HOME_DIR = get_home_dir()
SIMPLI_DIR = join(HOME_DIR, '.simpli')
SIMPLI_JSON_DIR = join(SIMPLI_DIR, 'jsons/')
establish_filepath(SIMPLI_JSON_DIR)
link_json(join(dirname(realpath(__file__)), 'default_tasks.json'))


# ==============================================================================
# Set up Jupyter Notebook extension
# ==============================================================================
def _jupyter_nbextension_paths():
    """
    Required function to add things to the nbextension path.
    :return: list; List of 1 dictionary
    """

    # section: the path is relative to the simpli/ directory
    # (if viewing from the repository: it's simpli/simpli/)
    #
    # dest: Jupyter sets up: server(such as localhost:8888)/nbextensions/dest/
    #
    # src: Jupyter sees this directory (not all files however) when it looks at
    # dest (server/nbextensions/dest/)
    #
    # require: Jupyter loads this file; things in this javascript will be seen
    # in the javascript namespace
    to_return = {
        'section': 'notebook',
        'src': 'static',
        'dest': 'simpli',
        'require': 'simpli/main',
    }

    return [to_return]


def _jupyter_server_extension_paths():
    """
    Required function to add things to the server extension path.
    :return: list; List of 1 dictionary
    """

    to_return = {
        'module': 'simpli',
    }
    return [to_return]


def load_jupyter_server_extension(nbapp):
    """
    Function to be called when server extension is loaded.
    :param nbapp: NotebookWebApplication; handle to the Notebook web-server
    :return: None
    """

    # Print statement to show extension is loaded
    nbapp.log.info('\n**************\n*** Simpli ***\n**************\n')
