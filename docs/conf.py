import os
import sys
import textwrap


sys.path.insert(0, os.path.abspath("../pymllm"))
sys.path.insert(0, os.path.abspath("../"))
autodoc_mock_imports = ["torch"]
project = "MLLM <br>"
version = "2.0.0-beta"
release = "2.0.0"
author = "MLLM Contributors"
copyright = "2024-2025, %s" % author

enable_doxygen = os.environ.get("MLLM_ENABLE_DOXYGEN", "false").lower() == "true"

extensions = [
    "sphinx_tabs.tabs",
    "sphinx_toolbox.collapse",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "myst_parser",
]

# API Doc Info
autoapi_type = "python"
autoapi_dirs = ["../pymllm"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]
autoapi_keep_files = False  # Useful for debugging the generated rst files
autoapi_generate_api_docs = True
autodoc_typehints = "description"
autoapi_ignore = []

if enable_doxygen:
    extensions.extend(["breathe", "exhale"])

this_file_dir = os.path.abspath(os.path.dirname(__file__))
if enable_doxygen:
    doxygen_xml_dir = os.path.join(this_file_dir, "xml")
    breathe_projects = {"mllm": doxygen_xml_dir}
    breathe_default_project = "mllm"

    repo_root = os.path.dirname(this_file_dir)

    # Setup the exhale extension
    exhale_args = {
        "containmentFolder": f"{os.path.join(this_file_dir, 'CppAPI')}",
        "rootFileName": "library_root.rst",
        "rootFileTitle": "Library API",
        "doxygenStripFromPath": repo_root,
        "exhaleExecutesDoxygen": True,
        "exhaleUseDoxyfile": True,
        "verboseBuild": True,
        "contentsDirectives": False,
        "pageLevelConfigMeta": ":github_url: https://github.com/UbiquitousLearning/mllm/",
        "contentsTitle": "Page Contents",
        "kindsWithContentsDirectives": ["class", "file", "namespace", "struct"],
        "afterTitleDescription": textwrap.dedent(
            """
            Welcome to the developer reference for the MLLM C++ API.
        """
        ),
    }

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
language = "en"
languages = ["en", "zh"]
exclude_patterns = ["build"]
pygments_style = "sphinx"
todo_include_todos = False
# == html settings
html_theme = "furo"
html_static_path = ["_static"]
footer_copyright = "© 2024-2025 MLLM"
footer_note = " "
# html_theme_options = {
#     "light_logo": "img/logo.svg",
#     "dark_logo": "img/logo.png",
# }
header_links = [
    ("Home", "https://github.com/UbiquitousLearning/mllm"),
    ("Github", "https://github.com/UbiquitousLearning/mllm"),
]
html_context = {
    "footer_copyright": footer_copyright,
    "footer_note": footer_note,
    "header_links": header_links,
    "display_github": True,
    "github_user": "UbiquitousLearning",
    "github_repo": "mllm",
    "github_version": "main/docs/",
    "theme_vcs_pageview_mode": "edit",
}
# == latex
latex_engine = "xelatex"
