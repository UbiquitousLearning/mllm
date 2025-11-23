QNN Environment Setup
=====================

Overview
--------

This section describes how to set up the QNN development environment, following the official QNN documentation. For more details, see: `QNN Linux Setup <https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html>`_.

Prerequisites
-------------

The QNN backend relies on two main SDKs:

- **Qualcomm QNN SDK**: Required for QNN backend compilation
- **Hexagon SDK**: Required for QNN custom operator(LLaMAOpPackage in mllm) compilation

Version Requirements
~~~~~~~~~~~~~~~~~~~~

- **QNN**: Linux v2.34+
- **Hexagon SDK**: Linux 6.x (For Hexagon SDK 5.x, refer to v1 branch for correct makefile Hexagon SDK Tool version)

.. warning::
   Some accounts may not have permission to access the Hexagon SDK and may need to contact Qualcomm for support.

SDK Download and Installation
-----------------------------

QNN SDK Installation
~~~~~~~~~~~~~~~~~~~~~

1. Download the QNN SDK from the `official Qualcomm website <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>`_
2. Unzip the downloaded file
3. Set the environment variable ``QNN_SDK_ROOT`` to point to the unzipped directory

Hexagon SDK Installation
~~~~~~~~~~~~~~~~~~~~~~~~

The `Hexagon SDK <https://www.qualcomm.com/developer/software/hexagon-npu-sdk>`_ is Qualcomm's official development environment for programming and optimizing applications on the Hexagon DSP — the core processor architecture used in Snapdragon chips for efficient, low-power computation.

By installing and sourcing the Hexagon SDK, developers can build the `custom op package <https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/op_packages.html>`_, which is the LLaMAOpPackage in this project, enabling HVX capabilities.

To install the Hexagon SDK, follow these steps:

1. Download the Hexagon SDK using `QPM <https://qpm.qualcomm.com/>`_ (Qualcomm Package Manager)
2. Install the SDK following the QPM instructions

.. note::
   If you encounter 'Login Failed' when using qpm-cli, check Qualcomm's agreements at <https://www.qualcomm.com/agreements>_ and update your account agreements.

Environment Setup
-----------------

After downloading and installing both SDKs, set up the environment by running the following commands:

.. code-block:: bash

   # Set up QNN SDK environment
   source <path-to-qnn-sdk>/bin/envsetup.sh
   
   # Set up Hexagon SDK environment
   source <path-to-hexagon-sdk>/setup_sdk_env.source

Environment Variables Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After setting up the environment, verify that the following environment variables are correctly set:

.. code-block:: bash

   echo $QNN_SDK_ROOT      # Should point to /path/to/your/qnn/sdk
   echo $HEXAGON_SDK_ROOT  # Should point to /path/to/your/hexagon/sdk

.. note::
   These environment variables are essential for the QNN op package compilation process.

Op Package Compilation
-----------------------

To use QNN offload, both CPU and HTP QNN op packages are required. The following steps will build the QNN op packages needed by the project.

Prerequisites for Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure the following environment variables are set:

- ``QNN_SDK_ROOT``
- ``HEXAGON_SDK_ROOT`` 
- ``ANDROID_NDK_ROOT``

Compilation Commands
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd mllm/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/
   make htp_aarch64 && make htp_v75

This will build the necessary QNN op packages for both AArch64 and HVX v75 targets.

Development Tips
----------------

LSP Configuration for HVX Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable Language Server Protocol (LSP) support for HVX development, configure clangd to use the Hexagon toolchain:

1. Create or edit ``.vscode/settings.json`` in your project root
2. Add the following configuration:

.. code-block:: json

   {
     "clangd.path": "$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/8.7.06/Tools/bin/hexagon-clangd"
   }

Generating Compilation Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate the ``compile_commands.json`` file for the Op package:

.. code-block:: bash

   cd mllm/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/
   compiledb make htp_v75 -C .

This compilation database is useful for IDE features like code completion and error highlighting.

Next Steps
----------

After completing the environment setup, you can proceed to:

- Model conversion and quantization
- Building the project with QNN backend
- Running QNN-accelerated models

For detailed instructions on these steps, refer to the respective documentation sections.
