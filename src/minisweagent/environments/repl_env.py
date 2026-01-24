"""REPL Environment for RLM-based SWE-bench agent with Docker-backed tools."""

import io
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from minisweagent import Environment


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={list(self.locals.keys())}, execution_time={self.execution_time})"


class SWEBenchREPLEnv:
    """REPL environment for SWE-bench with tools backed by Docker environment."""

    def __init__(
        self,
        docker_env: Environment,
        issue: str,
        repo_path: str = "/testbed",
        llm_query_func: callable = None,
    ):
        self.docker_env = docker_env
        self.repo_path = repo_path
        self.issue = issue
        self.llm_query_func = llm_query_func

        self.temp_dir = tempfile.mkdtemp(prefix="repl_env_")
        self._lock = threading.Lock()

        # Create safe globals with built-ins
        self.globals = self._create_safe_globals()
        self.locals = {}

        # Extract all files from Docker container
        files_content = self._extract_all_files()

        # Inject context and tools
        self.globals["context"] = {
            "issue": issue,
            "repo_path": repo_path,
            "files": files_content,
        }
        self.globals["get_file"] = self._get_file
        self.globals["write_file"] = self._write_file
        self.globals["bash"] = self._bash
        self.globals["FINAL_VAR"] = self._final_var
        # Only inject llm_query if llm_query_func is provided (depth=1)
        if llm_query_func is not None:
            self.globals["llm_query"] = self._llm_query

    def _create_safe_globals(self) -> dict:
        """Create a safe globals dict with common built-ins."""
        return {
            "__builtins__": {
                # Core functions
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "type": type,
                "isinstance": isinstance,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "round": round,
                "chr": chr,
                "ord": ord,
                "hex": hex,
                "bin": bin,
                "oct": oct,
                "repr": repr,
                "ascii": ascii,
                "format": format,
                "__import__": __import__,
                "open": open,
                "any": any,
                "all": all,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "delattr": delattr,
                "dir": dir,
                "vars": vars,
                "range": range,
                "reversed": reversed,
                "slice": slice,
                "iter": iter,
                "next": next,
                "pow": pow,
                "divmod": divmod,
                "complex": complex,
                "bytes": bytes,
                "bytearray": bytearray,
                "memoryview": memoryview,
                "hash": hash,
                "id": id,
                "callable": callable,
                "issubclass": issubclass,
                "super": super,
                "property": property,
                "staticmethod": staticmethod,
                "classmethod": classmethod,
                "object": object,
                # Exception classes
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "FileNotFoundError": FileNotFoundError,
                "OSError": OSError,
                "IOError": IOError,
                "RuntimeError": RuntimeError,
                "NameError": NameError,
                "ImportError": ImportError,
                "StopIteration": StopIteration,
                "AssertionError": AssertionError,
                "NotImplementedError": NotImplementedError,
                # Disallow dangerous built-ins
                "input": None,
                "eval": None,
                "exec": None,
                "compile": None,
                "globals": None,
                "locals": None,
            }
        }

    def _read_file(self, path: str) -> str:
        """Read a file from the Docker container."""
        if not path.startswith("/"):
            path = f"{self.repo_path}/{path}"
        result = self.docker_env.execute(f"cat {path}")
        if result["returncode"] != 0:
            return f"Error reading file: {result['output']}"
        return result["output"]

    def _write_file(self, path: str, content: str) -> str:
        """Write content to a file in the Docker container."""
        if not path.startswith("/"):
            path = f"{self.repo_path}/{path}"

        # Escape content for bash heredoc
        escaped_content = content.replace("\\", "\\\\").replace("$", "\\$").replace("`", "\\`")

        cmd = f"""cat > {path} << 'REPL_EOF'
{content}
REPL_EOF"""
        result = self.docker_env.execute(cmd)
        if result["returncode"] != 0:
            return f"Error writing file: {result['output']}"
        return f"Successfully wrote to {path}"

    def _bash(self, cmd: str) -> str:
        """Execute a bash command in the Docker container."""
        result = self.docker_env.execute(f"cd {self.repo_path} && {cmd}")
        output = result["output"]
        if result["returncode"] != 0:
            output = f"[Exit code: {result['returncode']}]\n{output}"
        return output

    def _llm_query(self, prompt: str) -> str:
        """Query the sub-LLM with the given prompt."""
        if self.llm_query_func is None:
            return "Error: llm_query function not configured"
        return self.llm_query_func(prompt)

    def _final_var(self, variable_name: str) -> str:
        """Return the value of a variable as the final answer."""
        variable_name = variable_name.strip().strip('"').strip("'").strip("\n").strip("\r")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        return f"Error: Variable '{variable_name}' not found in REPL environment"

    def _extract_all_files(self) -> str:
        """Extract all text files from the Docker container into a single string."""
        # Patterns to exclude from extraction
        exclude_patterns = [
            ".git/", "__pycache__/", ".pyc", ".pyo", ".so", ".o", ".a",
            ".egg-info/", ".eggs/", "node_modules/", ".tox/", ".nox/",
            ".coverage", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
            ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp",
            ".woff", ".woff2", ".ttf", ".eot", ".otf",
            ".pdf", ".doc", ".docx", ".xls", ".xlsx",
            ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
            ".exe", ".dll", ".bin", ".dat", ".db", ".sqlite",
            ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
            ".lock", "package-lock.json", "yarn.lock", "Cargo.lock",
        ]

        # Build find command with exclusions
        find_cmd = f"find {self.repo_path} -type f"
        for pattern in [".git", "__pycache__", "node_modules", ".tox", ".eggs", ".egg-info", ".pytest_cache", ".mypy_cache", ".ruff_cache"]:
            find_cmd += f" -not -path '*/{pattern}/*'"

        result = self.docker_env.execute(find_cmd)
        if result["returncode"] != 0:
            return f"Error listing files: {result['output']}"

        file_paths = [p.strip() for p in result["output"].strip().split("\n") if p.strip()]

        # Filter out binary/unwanted files by extension
        filtered_paths = []
        for path in file_paths:
            skip = False
            for pattern in exclude_patterns:
                if pattern in path:
                    skip = True
                    break
            if not skip:
                filtered_paths.append(path)

        # Extract content from each file
        files_content = []
        for path in filtered_paths:
            read_result = self.docker_env.execute(f"cat '{path}' 2>/dev/null")
            if read_result["returncode"] == 0:
                content = read_result["output"]
                # Skip binary files (files with null bytes or decode errors)
                try:
                    if "\x00" not in content:
                        files_content.append(f"###({path})\n{content}")
                except:
                    pass

        return "\n".join(files_content)

    def _get_file(self, path: str) -> str:
        """Get content of a specific file from the context."""
        if not path.startswith("/"):
            path = f"{self.repo_path}/{path}"
        
        files_str = self.globals["context"].get("files", "")
        marker = f"###({path})\n"
        
        start_idx = files_str.find(marker)
        if start_idx == -1:
            return f"Error: File '{path}' not found in context"
        
        content_start = start_idx + len(marker)
        # Find the next file marker or end of string
        next_marker = files_str.find("\n###(", content_start)
        if next_marker == -1:
            return files_str[content_start:]
        return files_str[content_start:next_marker]

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                yield stdout_buffer, stderr_buffer
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    @contextmanager
    def _temp_working_directory(self):
        """Context manager to temporarily change working directory."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def code_execution(self, code: str) -> REPLResult:
        """Execute Python code in the REPL environment."""
        start_time = time.time()
        with self._capture_output() as (stdout_buffer, stderr_buffer):
            with self._temp_working_directory():
                try:
                    # Split code into import statements and other code
                    lines = code.split("\n")
                    import_lines = []
                    other_lines = []

                    for line in lines:
                        if line.startswith(("import ", "from ")) and not line.startswith("#"):
                            import_lines.append(line)
                        else:
                            other_lines.append(line)

                    # Execute imports first
                    if import_lines:
                        import_code = "\n".join(import_lines)
                        exec(import_code, self.globals, self.globals)

                    # Execute the rest of the code
                    if other_lines:
                        other_code = "\n".join(other_lines)
                        combined_namespace = {**self.globals, **self.locals}

                        # Check if the last non-comment line is an expression
                        non_comment_lines = [line for line in other_lines if line.strip() and not line.strip().startswith("#")]

                        if non_comment_lines:
                            last_line = non_comment_lines[-1]

                            is_expression = (
                                not last_line.strip().startswith(
                                    ("import ", "from ", "def ", "class ", "if ", "for ", "while ", "try:", "with ", "return ", "yield ", "break", "continue", "pass")
                                )
                                and "=" not in last_line.split("#")[0]
                                and not last_line.strip().endswith(":")
                                and not last_line.strip().startswith("print(")
                            )

                            if is_expression:
                                try:
                                    # Execute all lines except the last one
                                    if len(non_comment_lines) > 1:
                                        last_line_start = -1
                                        for i, line in enumerate(other_lines):
                                            if line == last_line:
                                                last_line_start = i
                                                break

                                        if last_line_start > 0:
                                            statements_code = "\n".join(other_lines[:last_line_start])
                                            exec(statements_code, combined_namespace, combined_namespace)

                                    # Evaluate the last line as expression
                                    result = eval(last_line, combined_namespace, combined_namespace)
                                    if result is not None:
                                        print(repr(result))
                                except:
                                    exec(other_code, combined_namespace, combined_namespace)
                            else:
                                exec(other_code, combined_namespace, combined_namespace)
                        else:
                            exec(other_code, combined_namespace, combined_namespace)

                        # Update locals with new variables
                        for key, value in combined_namespace.items():
                            if key not in self.globals:
                                self.locals[key] = value

                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()
                except Exception as e:
                    stderr_content = stderr_buffer.getvalue() + str(e)
                    stdout_content = stdout_buffer.getvalue()

        execution_time = time.time() - start_time
        self.locals["_stdout"] = stdout_content
        self.locals["_stderr"] = stderr_content

        return REPLResult(stdout_content, stderr_content, self.locals.copy(), execution_time)

    def get_patch(self) -> str:
        """Get the git diff patch from the Docker container."""
        result = self.docker_env.execute(f"cd {self.repo_path} && git add -A && git diff --cached")
        return result["output"]

    def cleanup(self):
        """Clean up temporary directory."""
        try:
            import shutil

            shutil.rmtree(self.temp_dir)
        except:
            pass

    def __del__(self):
        self.cleanup()

