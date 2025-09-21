# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import os
import time
import random
import string
import subprocess
from typing import List, Dict, Optional, Union


class ShellContext:
    """
    Manages a persistent 'adb shell' session for a specific device.
    This allows running a series of commands that maintain state, such as
    environment variables or the current working directory.

    It's recommended to use this class as a context manager (with statement)
    to ensure the shell session is always closed properly.

    Example:
        with adb.get_shell_context(device_id) as shell:
            shell.execute("export MY_VAR=hello")
            output = shell.execute("echo $MY_VAR")
            print(output)  # Should print "hello"

    Note: This class is not thread-safe. A single instance should only be
    used from one thread at a time.
    """

    def __init__(self, adb_path: str, device_id: Optional[str] = None):
        """
        Initializes and starts the persistent shell session.
        :param adb_path: Path to the adb executable.
        :param device_id: The ID of the target device.
        """
        self.adb_path = adb_path
        self.device_id = device_id
        # A unique marker to detect the end of a command's output.
        # This is a reliable way to read output without trying to parse
        # shell prompts, which can change and are unreliable.
        self._end_marker = f"END_OF_COMMAND_{''.join(random.choices(string.ascii_letters + string.digits, k=32))}"
        self._process = self._start_shell()
        # Read and discard any initial startup messages or the first prompt.
        self._consume_initial_prompt()

    def _start_shell(self) -> subprocess.Popen:
        """Starts the underlying 'adb shell' process."""
        cmd = [self.adb_path]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.append("shell")

        try:
            # Start the shell process. Stderr is redirected to stdout to
            # capture all output in one stream. We use a specific encoding
            # and a line-buffered mode.
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except FileNotFoundError:
            raise RuntimeError(f"ADB executable not found at '{self.adb_path}'")
        except Exception as e:
            raise RuntimeError(f"Failed to start adb shell: {e}")

    def _consume_initial_prompt(self):
        """Executes a simple command to clear any initial login messages."""
        # The ':' command is a standard shell no-op (do nothing).
        # We call execute here to read past any initial text from the shell.
        self.execute(":")

    def execute(self, command: str) -> str:
        """
        Executes a command in the persistent shell.
        :param command: The shell command to execute.
        :return: The output (stdout and stderr) of the command.
        """
        if not self.is_alive():
            raise RuntimeError("Shell process is not running. Cannot execute command.")

        # Combine the user's command with our end-marker echo command.
        # The newline is crucial to ensure the command is executed.
        command_to_run = f"{command}; echo {self._end_marker}\n"

        try:
            self._process.stdin.write(command_to_run)
            self._process.stdin.flush()
        except (IOError, BrokenPipeError) as e:
            raise RuntimeError(
                f"Failed to write to shell stdin: {e}. The shell may have crashed."
            )

        output_lines = []
        while True:
            line = self._process.stdout.readline()
            if not line:
                # This indicates the shell has died unexpectedly.
                full_output = "".join(output_lines)
                raise RuntimeError(
                    f"Shell process terminated unexpectedly. Output so far:\n{full_output}"
                )

            # Our end marker will appear on its own line. We strip any whitespace/newlines.
            if line.strip() == self._end_marker:
                break

            output_lines.append(line)

        return "".join(output_lines).strip()

    def close(self):
        """Terminates the shell session and cleans up resources."""
        if self._process and self.is_alive():
            try:
                self._process.stdin.write("exit\n")
                self._process.stdin.flush()
            except (IOError, BrokenPipeError):
                pass  # Pipe might be closed already, which is fine.
            finally:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                except Exception:
                    pass  # Ignore other errors on close
        self._process = None

    def is_alive(self) -> bool:
        """Checks if the shell process is still running."""
        return self._process and self._process.poll() is None

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager, ensuring the shell is closed."""
        self.close()


class ADBToolkit:
    def __init__(self, adb_path: str = "adb"):
        """
        Initialize ADB toolkit
        :param adb_path: Path to adb executable (default: 'adb' from system PATH)
        """
        self.adb_path = adb_path
        self._check_adb_available()

    def _run_command(
        self, command: Union[str, List[str]], device_id: str = None
    ) -> dict:
        """
        Execute ADB command and return results
        Supports both string and list command formats
        """
        if isinstance(command, str):
            cmd_parts = command.split()
        else:
            cmd_parts = command

        cmd = [self.adb_path]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(cmd_parts)

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout.strip(),
                "error": result.stderr.strip(),
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out"}
        except FileNotFoundError:
            return {
                "success": False,
                "error": f"ADB executable not found at '{self.adb_path}'",
            }

    def _check_adb_available(self):
        """Verify ADB is available and working"""
        result = self._run_command("version")
        if not result["success"]:
            raise RuntimeError(
                f"ADB not available: {result.get('error', 'Unknown error')}"
            )

    def get_shell_context(self, device_id: Optional[str] = None) -> ShellContext:
        """
        Get a persistent shell context to a device.
        This allows running commands that maintain state (e.g., environment variables).
        It is highly recommended to use this with a 'with' statement.

        :param device_id: Target device ID. If None, and only one device is
                          connected, it will be used automatically.
        :return: A ShellContext instance.
        """
        target_device = device_id
        if not target_device:
            devices = self.get_devices()
            if len(devices) == 1:
                target_device = devices[0]["id"]
            elif len(devices) == 0:
                raise RuntimeError("Cannot open shell context: No devices connected.")
            else:
                raise RuntimeError(
                    "Cannot open shell context: More than one device connected. Please specify a device_id."
                )

        return ShellContext(self.adb_path, target_device)

    def get_devices(self) -> List[Dict[str, str]]:
        """Get list of connected devices"""
        result = self._run_command("devices -l")
        if not result["success"]:
            return []

        devices = []
        for line in result["output"].splitlines()[1:]:
            if "device" in line and "unauthorized" not in line:
                parts = line.split()
                if len(parts) < 2:
                    continue
                device_id = parts[0]
                info = " ".join(parts[1:])
                devices.append({"id": device_id, "info": info})
        return devices

    def install_apk(self, apk_path: str, device_id: str = None) -> bool:
        """Install APK file on device"""
        result = self._run_command(f'install -r "{apk_path}"', device_id)
        return result["success"] and "Success" in result["output"]

    def uninstall_app(self, package_name: str, device_id: str = None) -> bool:
        """Uninstall application from device"""
        result = self._run_command(f"uninstall {package_name}", device_id)
        return result["success"] and "Success" in result["output"]

    def push_file(
        self, local_path: str, device_path: str, device_id: str = None
    ) -> bool:
        """Push file to device"""
        cmd_args = ["push", local_path, device_path]
        result = self._run_command(cmd_args, device_id)
        return result["success"]

    def pull_file(
        self, device_path: str, local_path: str, device_id: str = None
    ) -> bool:
        """Pull file from device"""
        cmd_args = ["pull", device_path, local_path]
        result = self._run_command(cmd_args, device_id)
        return result["success"]

    def take_screenshot(self, save_path: str, device_id: str = None) -> bool:
        """Capture device screenshot"""
        temp_path = f"/sdcard/screenshot_{int(time.time())}.png"
        if self._run_command(f'shell screencap -p "{temp_path}"', device_id)["success"]:
            if self.pull_file(temp_path, save_path, device_id):
                self._run_command(f'shell rm "{temp_path}"', device_id)
                return True
        return False

    def record_screen(
        self, save_path: str, duration: int = 60, device_id: str = None
    ) -> bool:
        """Record device screen"""
        # This function cannot use the persistent shell context because it involves
        # a long-running command followed by a time.sleep in the Python script.
        # It must be run as a standalone process.
        temp_path = f"/sdcard/record_{int(time.time())}.mp4"
        self._run_command(
            f'shell screenrecord --time-limit {duration} "{temp_path}"', device_id
        )
        time.sleep(duration + 2)
        if self.pull_file(temp_path, save_path, device_id):
            self._run_command(f'shell rm "{temp_path}"', device_id)
            return True
        return False

    def execute_command(self, command: str, device_id: str = None) -> str:
        """
        Execute a single, stateless shell command on device.
        This command will NOT maintain context like environment variables.
        For stateful sessions, use get_shell_context().
        """
        result = self._run_command(f"shell {command}", device_id)
        return result["output"] if result["success"] else f"Error: {result['error']}"

    def get_device_info(self, device_id: str = None) -> Dict[str, str]:
        """Get device information"""
        return {
            "brand": self.execute_command("getprop ro.product.brand", device_id),
            "model": self.execute_command("getprop ro.product.model", device_id),
            "version": self.execute_command(
                "getprop ro.build.version.release", device_id
            ),
            "serial": self.execute_command("getprop ro.serialno", device_id),
        }

    def reboot_device(self, device_id: str = None) -> bool:
        """Reboot device"""
        result = self._run_command("reboot", device_id)
        return result["success"]
