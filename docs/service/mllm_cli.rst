MLLM CLI
============

Overview
--------

This document describes the MLLM command-line interface (CLI) tool, which operates within a client-server architecture. The system is designed to provide network access to MLLM's core inference capabilities. The backend service is written in Go and interacts with the core C++ MLLM library through a C API. The frontend can be a Go-based command-line client or a standard GUI client like Chatbox that communicates with the service via an OpenAI-compatible API.

**Currently, the system officially supports the following models:**

* **LLM**: ``mllmTeam/Qwen3-0.6B-w4a32kai``
* **OCR**: ``mllmTeam/DeepSeek-OCR-w4a8-i8mm-kai``

This guide covers three main areas:

1.  **System Architecture and API**: An explanation of the components and the C API bridge.
2.  **Build Configuration**: Instructions on how to adapt the build scripts for different environments.
3.  **Compilation and Deployment**: A step-by-step guide to build and run the entire stack.

Section 3 will guide you through the complete steps to reproduce the mllm_cli. Before you begin, we highly recommend reading and understanding Section 1 (System Architecture and API) and Section 2 (Build Configuration) first. This will help you follow the guide in Section 3 more smoothly.

System Architecture
-------------------

The system consists of three primary components: the C++ core with a C API wrapper, a Go backend service, and a client (Go CLI or GUI).

1. C/C++ Core & C API Layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core MLLM functionalities are implemented in C++. To allow communication with other languages like Go, a C-style API is exposed.

**Key Data Structures (`mllm/c_api/Object.h`)**

The C API uses shared data structures to pass information between Go and C++.

* ``MllmCType``: An enum that defines the type of data being handled, such as integers, floats, tensors, or custom objects.
* ``MllmCAny``: A versatile union-based struct that can hold different types of values, from primitive types to pointers for complex objects. This is the primary data exchange format.

**Key API Functions (`mllm/c_api/Runtime.h`, `mllm/c_api/Runtime.cpp`)**

These C functions wrap the C++ service logic, making them callable from Go via `cgo`.

* `createQwen3Session(const char* model_path)`: Loads a model from the specified path and creates a session handle.
* `createDeepseekOCRSession(const char* model_path)`: Loads a DeepSeek-OCR model from the specified path and creates a session handle. This session is specifically designed to handle visual inputs and OCR tasks.
* `insertSession(const char* session_id, MllmCAny handle)`: Registers the created session with a unique ID in the service.
* `sendRequest(const char* session_id, const char* json_request)`: Sends a user's request (in JSON format) to the specified session for processing.
* `pollResponse(const char* session_id)`: Polls for a response from the model. This is used for streaming results back to the client.
* `freeSession(MllmCAny handle)`: Releases the resources associated with a session.

2. Go Service Layer (`mllm-server`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `mllm-server` is an HTTP server written in Go. It acts as a bridge between network clients and the MLLM C++ core.

* **Initialization (Dual Model Support)**: On startup, the server checks for two command-line arguments:
    
    * ``--model-path``: Path to the Qwen3 LLM model.
    * ``--ocr-model-path``: Path to the DeepSeek-OCR model.
    
    If provided, the server initializes the respective sessions (`createQwen3Session` and/or `createDeepseekOCRSession`) and registers them with their directory names as Session IDs.

* **API Endpoint**: It exposes an OpenAI-compatible endpoint at ``/v1/chat/completions``.

* **Request Handling & OCR Preprocessing**: 
    When a request arrives, the server inspects the ``model`` parameter.
    
    * **Text Requests**: Routed directly to the Qwen3 session.
    * **OCR Requests**: If the model name contains "OCR", the server triggers a preprocessing step (`preprocessRequestForOCR`). It detects Base64 encoded images in the payload, decodes them, saves them to temporary files on the device, and modifies the request to point the C++ backend to these file paths.

* **Streaming Response**: Results are streamed back to the client over HTTP using Server-Sent Events (SSE).

**Key Service Layer Files**

The `mllm-server` functionality is implemented across several key Go files:

* ``pkg/server/server.go``
    * **Purpose**: Defines the HTTP server instance itself. It is responsible for starting the server (`Start()`), setting the listening address (e.g., ``:8080``), registering API routes, and managing graceful shutdown (`Shutdown()`).
* ``pkg/server/handlers.go``
    * **Purpose**: Contains the core logic for the API endpoint (`chatCompletionsHandler`). It is responsible for:
        1. Parsing the incoming JSON request from the client.
        2. Retrieving or validating the model session from the `mllmService` (`GetSession`).
        3. Forwarding the request to the C++ core (`session.SendRequest`).
        4. Continuously polling (`session.PollResponse`) for responses from the C++ layer.
        5. Streaming the responses back to the client in the standard Server-Sent Events (SSE) format.
        6. For OCR models, it identifies Base64 encoded images in the request, decodes them into temporary files on the Android device, and updates the request payload with the local file paths.
* ``pkg/mllm/service.go``
    * **Purpose**: Acts as the Go-level **session manager** (`Service`). It holds a map that links model IDs (e.g., "Qwen3-0.6B-w4a32kai") to their active MLLM sessions (`*mllm.Session`). `handlers.go` uses this to find the correct session instance.
* ``pkg/api/types.go``
    * **Purpose**: Defines the shared data structures (Go structs) used for communication between the client and server. This includes `OpenAIRequest` (request body) and `OpenAIResponseChunk` (response body), ensuring the data format adheres to the OpenAI-compatible API specification.

3. Go Client (`mllm-client`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `mllm-client` is an interactive command-line tool that allows users to chat with the model.

* **User Interaction**: It reads user input from the console.
* **API Communication**: It formats the user input into an OpenAI-compatible JSON request and sends it to the `mllm-server`.
* **Response Handling**: It receives the SSE stream from the server, decodes the JSON chunks, and prints the assistant's response to the console in real-time.

.. note::
    **Current Limitation**: The ``mllm-client`` is currently hardcoded to use the **Qwen3** model only. It does not support switching to the DeepSeek-OCR model or uploading images. For OCR tasks, please use a GUI client like Chatbox.

Alternative Client: Chatbox
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the ``mllm-client`` command-line tool, you can use a graphical user interface (GUI) client like **Chatbox**, as it supports the OpenAI-compatible API.

**Chatbox Configuration**

To connect Chatbox (running on your host machine) to your ``mllm-server`` (running on the Android device), you must first forward the device's port to your local machine using `adb`.

.. code-block:: bash

   # Forward local port 8081 to the Android device's port 8080.
   adb forward tcp:8081 tcp:8080

After running this command, configure Chatbox with the following settings:

* **Name**: ``mllmTeam`` (or any name you prefer)
* **API Mode**: ``OpenAI API Compatible``
* **API Key**: (Can be left blank or any value; the server does not currently check it)
* **API Host**: ``http://localhost:8081``
* **API Path**: ``/v1/chat/completions``
* **Model**: **[Important]** You must manually add the model name by clicking **+ New**.

The name **MUST match the folder name** of the model directory on the Android device.
  
  * For the LLM, enter: ``Qwen3-0.6B-w4a32kai`` (or your specific LLM folder name).
  * For OCR, enter: ``DeepSeek-OCR-w4a8-i8mm-kai`` (or your specific OCR folder name).

.. note::
    The server uses the directory name (e.g., ``filepath.Base``) as the session ID. If you enter a different name in Chatbox, the server will return a "Model not found" error.

Once configured, you can click the **Check** button to ensure the connection is successful. Please note that this step must be performed while the server is running.

Build Configuration Guide
-------------------------

The Go build tasks (`build_android_mllm_server.yaml` and `build_android_mllm_client.yaml`) use hardcoded paths that are specific to the build server's environment. If you are setting up a new build environment, you **must** modify these paths before proceeding to compilation.

Understanding the Build Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core of the cross-compilation logic for Go is within the `ShellCommandTask` of the `.yaml` build files. It sets several environment variables to configure `cgo` for cross-compiling to Android ARM64.

* ``GOOS=android``, ``GOARCH=arm64``: Tells Go to build for Android ARM64.
* ``CGO_ENABLED=1``: Enables ``cgo`` to allow Go to call C/C++ code.
* ``CC`` and ``CXX``: Specifies the C and C++ compilers from the Android NDK, used to compile any C/C++ parts within the Go program.
* ``CGO_CFLAGS``: Tells the C compiler where to find the MLLM C API header files (e.g., ``Runtime.h``).
* ``CGO_LDFLAGS``: Tells the linker where to find the compiled MLLM shared libraries (``.so`` files) that the final executable needs to link against.

Modifying Hardcoded Paths
~~~~~~~~~~~~~~~~~~~~~~~~~

The two most critical variables you will need to change are `CGO_CFLAGS` and `CGO_LDFLAGS`.

**Example from `build_android_mllm_server.yaml`**:

.. code-block:: yaml

   # ...
   export CGO_LDFLAGS="-L/root/zty_workspace/mllm_zty/build-android-arm64-v8a/bin"
   export CGO_CFLAGS="-I/root/zty_workspace/mllm_zty"
   # ...

**How to Modify**:

1.  **``CGO_CFLAGS="-I/path/to/your/project/root"``**
    The `-I` flag specifies an include directory. This path should point to the root of the MLLM project directory on your build server, where the `mllm/c_api/` headers are located. In the example, this is `/root/zty_workspace/mllm_zty`. Change this to match your project's location.

2.  **``CGO_LDFLAGS="-L/path/to/your/compiled/libs"``**
    The `-L` flag specifies a library directory. This path must point to the directory where the C++ build (Step 1) placed the `.so` files. In the example, this is `/root/zty_workspace/mllm_zty/build-android-arm64-v8a/bin`. If your build output directory is different, you must update this path accordingly.

By correctly updating these two paths in both `build_android_mllm_server.yaml` and `build_android_mllm_client.yaml`, you can adapt the build process to any server environment.

Compilation and Deployment
--------------------------

This section provides the complete workflow for compiling all C++ and Go components, deploying them to an Android target, and running the system.

Prerequisites
~~~~~~~~~~~~~

* A build environment, such as a server or Docker container, with the Android NDK and Go compiler installed, hereinafter referred to as the 'build server'.
* An Android target device with `adb` access enabled.
* `rsync` and `scp` for file synchronization between your development machine and the build server.

Step 1: Compile C++ Core Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we compile the MLLM C++ core, which produces the essential shared libraries (`.so` files).

1.  **Sync Code to Build Server**:
    Synchronize your local project directory with the build server.

    .. code-block:: bash

       # Replace <port>, <user>, and <build-server-ip> with your server details
       rsync -avz --checksum -e 'ssh -p <port>' --exclude 'build' --exclude '.git' ./ <user>@<build-server-ip>:/your_workspace/your_programname/

2.  **Run the Build Task**:
    On the build server, execute the build task. This task uses `tasks/build_android.yaml` to configure and run CMake.

    Before executing this step, you also need to ensure that the hardcoded directories in build_android.yaml have been modified to match your requirements. The modification method is the same as for the Go compilation file mentioned earlier.
    
    .. code-block:: bash

       # These commands are run on your build server.
       cd /your_workspace/your_programname/
       python task.py tasks/build_android.yaml

3.  **Retrieve Compiled Libraries**:
    After the build succeeds, copy the compiled shared libraries from the build server back to your local machine. These libraries are the C++ backend that the Go application will call.

    .. code-block:: bash

       # You run these commands on your local machine to copy the files from the build server.
       # Navigate to your local build artifacts directory
       cd /path/to/your/local_artifacts_dir/

       # Copy the libraries
       scp -P <port> <user>@<build-server-ip>:/your_workspace/your_programname/build-android-arm64-v8a/bin/libMllmRT.so .
       scp -P <port> <user>@<build-server-ip>:/your_workspace/your_programname/build-android-arm64-v8a/bin/libMllmCPUBackend.so .
       scp -P <port> <user>@<build-server-ip>:/your_workspace/your_programname/build-android-arm64-v8a/bin/libMllmSdkC.so .

Step 2: Compile the Go Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, cross-compile the Go server application for Android.

1.  **Sync Code**: Ensure your latest Go code is on the build server. This is only necessary if you've made changes to the Go server files (e.g., in the ``mllm-cli`` directory).

2.  **Run the Build Task**:
    On the build server, execute the server build task. Make sure you have correctly configured the hardcoded paths in this YAML file as described in the "Build Configuration Guide" section.

    .. code-block:: bash

       cd /your_workspace/your_programname/
       python task.py tasks/build_android_mllm_server.yaml

3.  **Retrieve the Executable**:
    Copy the compiled `mllm_web_server` binary from the build server back to your local machine.

    .. code-block:: bash

       # Navigate to your local build artifacts directory
       cd /path/to/your/local_artifacts_dir/

       # Copy the executable
       scp -P <port> <user>@<build-server-ip>:/your_workspace/your_programname/build-android-arm64-v8a/bin/mllm_web_server .

Step 3: Compile the Go Client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using Chatbox or a similar client, you can skip this step.

Similarly, compile the Go client application.

1.  **Sync Code**: Ensure your latest Go client code is on the build server.

2.  **Run the Build Task**:
    On the build server, execute the client build task. This also requires the build YAML to be correctly configured.

    .. code-block:: bash

       cd /your_workspace/your_programname/
       python task.py tasks/build_android_mllm_client.yaml

3.  **Retrieve the Executable**:
    Copy the compiled `mllm_ws_client` binary from the build server to your local machine.

    .. code-block:: bash

       # Navigate to your local build artifacts directory
       cd /path/to/your/local_artifacts_dir/

       # Copy the executable
       scp -P <port> <user>@<build-server-ip>:/your_workspace/your_programname/build-android-arm64-v8a/bin/mllm_ws_client .

Step 4: Deploy to Target Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Push all compiled artifacts (libraries and executables) to your target Android device.

.. code-block:: bash

   # Connect to your device if you haven't already
   adb connect <device-id-or-ip:port>

   # Push the shared libraries from your local artifacts directory
   adb push libMllmRT.so /path/to/your/deployment_dir/
   adb push libMllmCPUBackend.so /path/to/your/deployment_dir/
   adb push libMllmSdkC.so /path/to/your/deployment_dir/

   # Push the server and client executables
   adb push mllm_web_server /path/to/your/deployment_dir/

   # (Optional) Push the Go client if you compiled it in Step 3
   adb push mllm_ws_client /path/to/your/deployment_dir/

Step 5: Running and Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This covers testing with both the Go CLI client and Chatbox.

**A. Testing with the Go CLI Client (On-Device)**

1.  **Terminal 1: Start the Server**:
    Open a shell on the device, navigate to the deployment directory, set the library path, make the server executable, and run it with the required model path.

    .. code-block:: bash

       adb shell
       # Inside the adb shell
       cd /path/to/your/deployment_dir
       chmod +x mllm_web_server
       export LD_LIBRARY_PATH=.
       # Ensure you provide the correct path to your model
       # Option A: Run with Qwen3 LLM only
       ./mllm_web_server --model-path /path/to/your/qwen3_model_dir

       # Option B: Run with both Qwen3 LLM and DeepSeek-OCR
       # Use this if you plan to switch between text chat and OCR tasks
       ./mllm_web_server \
           --model-path /path/to/your/qwen3_model_dir \
           --ocr-model-path /path/to/your/deepseek_ocr_model_dir

    .. warning::
       The `export LD_LIBRARY_PATH=.` command is crucial. It tells the Android dynamic linker to look for the `.so` files in the current directory. Without it, the server will fail to start.

2.  **Terminal 2: Run the Go Client**:
    Open a second terminal and start the client.

    .. code-block:: bash

       adb shell "cd /path/to/your/deployment_dir && chmod +x mllm_ws_client && ./mllm_ws_client"

You should now be able to interact with the model from the client terminal. Type `/exit` to quit the client. Use `Ctrl+C` in the server terminal to stop the server.

**B. Testing with Chatbox (Host Machine)**

1.  **Terminal 1: Start the Server**:
    Follow the same instructions as in **Step 5.A.1** to start the server on the Android device.If you intend to test the OCR functionality, please ensure that you used Option B (which specifies the --ocr-model-path).

2.  **Terminal 2: Set up Port Forwarding**:
    On your host machine (not in the adb shell), run the following command. This maps your local host port (e.g., 8081) to the device's port (e.g., 8080).

    .. code-block:: bash

       # This forwards your host port 8081 to the device's port 8080
       # You can change 8081 to any available port on your host.
       adb forward tcp:8081 tcp:8080

3.  **Open Chatbox**:
    Open the Chatbox application on your host machine and configure it according to the "Alternative Client: Chatbox" section above. You can now chat with the model through the GUI.