MLLM CLI
============

Overview
--------

This document describes the MLLM command-line interface (CLI) tool, which operates within a client-server architecture. The system is designed to provide network access to MLLM's core inference capabilities. The backend service is written in Go and interacts with the core C++ MLLM library through a C API. The frontend can be a Go-based command-line client or a standard GUI client like Chatbox that communicates with the service via an OpenAI-compatible API.

**Currently, the system officially supports the following models:**

* **LLM**: ``mllmTeam/Qwen3-0.6B-w4a32kai,mllmTeam/Qwen3-4B-w4a8-i8mm-kai``
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

Go build tasks (`build_android_mllm_server.yaml` and `build_android_mllm_client.yaml`) now use **dynamic paths** (via `$(pwd)`) to locate project files. This makes the scripts much more portable.

However, there are still specific environment configurations you **must** check and modify to match your local build machine.

Open `tasks/build_android_mllm_server.yaml` and check the following variables:

1.  **``ANDROID_NDK_HOME`` (Critical)**
    The script assumes the NDK is located at `/opt/ndk/android-ndk-r28b`.
    * **Action:** Change this path to the actual location of the Android NDK on your machine.

2.  **``GOPROXY`` (Network)**
    The script uses `https://goproxy.cn` to accelerate downloads in China.
    * **Action:** If you are outside of China, you may remove this line or change it to `https://proxy.golang.org`.


Compilation and Deployment
--------------------------

This section outlines the workflow for compiling the C++ and Go components and deploying them directly to an Android device.

Prerequisites
~~~~~~~~~~~~~

* Android NDK and Go compiler installed and configured.
* An Android device connected via `adb`.

Step 1: Compile C++ Core Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compile the MLLM C++ core to generate the required shared libraries (`.so` files).

1.  **Configure Build Script**:
    Ensure `tasks/build_android.yaml` is configured with your NDK path.

2.  **Run Compilation**:
    Execute the build task from the project root.

    .. code-block:: bash

       python task.py tasks/build_android.yaml

    **Output**: The compiled libraries (`libMllmRT.so`, `libMllmCPUBackend.so`, `libMllmSdkC.so`) will be located in `build-android-arm64-v8a/bin/`.

Step 2: Compile the Go Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cross-compile the Go server application for Android.

1.  **Run Compilation**:
    Execute the server build task.

    .. code-block:: bash

       python task.py tasks/build_android_mllm_server.yaml

    **Output**: The executable `mllm_web_server` will be generated in `build-android-arm64-v8a/bin/`.

Step 3: Compile the Go Client (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need the command-line WebSocket client, compile it using the following task.

1.  **Run Compilation**:

    .. code-block:: bash

       python task.py tasks/build_android_mllm_client.yaml

    **Output**: The executable `mllm_ws_client` will be generated in `build-android-arm64-v8a/bin/`.

Step 4: Deploy to Target Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Push the compiled artifacts directly from the build output directory to your Android device using `adb`.

.. code-block:: bash

   # 1. Connect to device (if not already connected via USB)
   adb connect <device-ip:port>

   # 2. Push libraries and executables
   # Navigate to the build output directory
   cd build-android-arm64-v8a/bin/

   # Define your target directory on Android (e.g., /data/local/tmp/mllm/)
   export ANDROID_DIR=/data/local/tmp/mllm/

   # Push files
   adb push libMllmRT.so $ANDROID_DIR
   adb push libMllmCPUBackend.so $ANDROID_DIR
   adb push libMllmSdkC.so $ANDROID_DIR
   adb push mllm_web_server $ANDROID_DIR

   # (Optional) Push client if compiled
   # adb push mllm_ws_client $ANDROID_DIR
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
