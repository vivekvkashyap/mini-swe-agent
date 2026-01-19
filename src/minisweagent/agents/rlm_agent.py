"""RLM Agent for SWE-bench using REPL-based execution."""

import re
import time
from typing import Any

from pydantic import BaseModel

from minisweagent import Environment, Model
from minisweagent.environments.repl_env import SWEBenchREPLEnv


class RLMAgentConfig(BaseModel):
    system_template: str
    instance_template: str
    max_iterations: int = 20
    step_limit: int = 0
    cost_limit: float = 3.0
    max_output_length: int = 100000


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


def find_repl_blocks(text: str) -> list[str]:
    """Find REPL code blocks in text wrapped in triple backticks."""
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)
    return results


def find_final_answer(text: str) -> tuple[str, str] | None:
    """Find FINAL(...) or FINAL_VAR(...) statement in response."""
    # Check for FINAL_VAR pattern first
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL_VAR", match.group(1).strip())

    # Check for FINAL pattern
    final_pattern = r"^\s*FINAL\((.*?)\)"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return ("FINAL", match.group(1).strip())

    return None


def format_execution_result(stdout: str, stderr: str, locals_dict: dict, truncate_length: int = 100) -> str:
    """Format the execution result as a string for display."""
    result_parts = []

    if stdout:
        result_parts.append(f"\n{stdout}")

    if stderr:
        result_parts.append(f"\n{stderr}")

    # Show key variables (excluding internal ones)
    important_vars = {}
    for key, value in locals_dict.items():
        if not key.startswith("_") and key not in ["__builtins__", "__name__", "__doc__"]:
            try:
                if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                    if isinstance(value, str) and len(value) > truncate_length:
                        important_vars[key] = f"'{value[:truncate_length]}...'"
                    else:
                        important_vars[key] = repr(value)
            except:
                important_vars[key] = f"<{type(value).__name__}>"

    if important_vars:
        result_parts.append(f"REPL variables: {list(important_vars.keys())}\n")

    return "\n\n".join(result_parts) if result_parts else "No output"


class RLMAgent:
    """Agent that uses RLM REPL-based execution for SWE-bench tasks."""

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = RLMAgentConfig,
        **kwargs,
    ):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.repl_env: SWEBenchREPLEnv | None = None
        self.extra_template_vars = {}

    def _llm_query_func(self, prompt: str) -> str:
        """Sub-LLM query function for REPL environment."""
        sub_messages = [{"role": "user", "content": prompt}]
        response = self.model.query(sub_messages)
        return response.get("content", "")

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run the RLM agent loop until completion or limits exceeded."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []

        # Initialize REPL environment with Docker-backed tools
        self.repl_env = SWEBenchREPLEnv(
            docker_env=self.env,
            issue=task,
            repo_path="/testbed",
            llm_query_func=self._llm_query_func,
        )

        # Add system and initial user messages
        self.add_message("system", self.config.system_template)
        self.add_message("user", self._format_instance_template(task))

        try:
            for iteration in range(self.config.max_iterations):
                self._check_limits()

                # Query the model
                response = self.model.query(self.messages)
                response_content = response.get("content", "")
                self.add_message("assistant", response_content)

                # Find and execute REPL code blocks
                code_blocks = find_repl_blocks(response_content)

                if code_blocks:
                    for code in code_blocks:
                        result = self.repl_env.code_execution(code)
                        formatted_result = format_execution_result(
                            result.stdout, result.stderr, result.locals
                        )

                        # Truncate if too long
                        if len(formatted_result) > self.config.max_output_length:
                            formatted_result = formatted_result[: self.config.max_output_length] + "..."

                        execution_msg = f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{formatted_result}"
                        self.add_message("user", execution_msg)

                # Check for final answer
                final_result = find_final_answer(response_content)
                if final_result:
                    answer_type, content = final_result

                    if answer_type == "FINAL":
                        # Direct answer - get the patch
                        patch = self.repl_env.get_patch()
                        return "Submitted", patch

                    elif answer_type == "FINAL_VAR":
                        # Variable reference
                        variable_name = content.strip().strip('"').strip("'")
                        if variable_name in self.repl_env.locals:
                            value = self.repl_env.locals[variable_name]
                            # If the variable contains the patch, use it
                            # Otherwise, get the patch from git
                            if isinstance(value, str) and value.startswith("diff "):
                                return "Submitted", value
                            else:
                                patch = self.repl_env.get_patch()
                                return "Submitted", patch
                        else:
                            # Variable not found, continue iteration
                            self.add_message(
                                "user",
                                f"Error: Variable '{variable_name}' not found in REPL environment. Please define it or use a different variable.",
                            )

                # Add continuation prompt if no final answer
                if not code_blocks and not final_result:
                    self.add_message(
                        "user",
                        "Please continue using the REPL environment to solve the task. Write Python code in ```repl``` blocks.",
                    )

            # Max iterations reached - get final patch
            patch = self.repl_env.get_patch()
            return "MaxIterations", patch

        except LimitsExceeded:
            patch = self.repl_env.get_patch() if self.repl_env else ""
            return "LimitsExceeded", patch
        except Submitted as e:
            return "Submitted", str(e)
        finally:
            if self.repl_env:
                self.repl_env.cleanup()

    def _check_limits(self):
        """Check if step or cost limits have been exceeded."""
        if 0 < self.config.step_limit <= self.model.n_calls:
            raise LimitsExceeded("Step limit exceeded")
        if 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded("Cost limit exceeded")

    def _format_instance_template(self, task: str) -> str:
        """Format the instance template with task details."""
        return self.config.instance_template.replace("{{task}}", task)

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump()

