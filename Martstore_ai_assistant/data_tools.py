"""
Tools the agent can choose from.

These are thin, read-only wrappers. They contain NO troubleshooting knowledge:
all knowledge and interpretation comes from the PDF (via search_knowledge_base)
and from the LLM's reasoning.

Safety rules (from the handbook, Chapter 9):
- Allowlist of explicit tools; the LLM never supplies a raw shell command.
- Arguments are validated; no shell is used.
- Timeouts and output-size limits.
- Output is labelled as an OBSERVATION, not a diagnosis.
"""

import ipaddress
import platform
import re
import subprocess

from langchain_core.tools import tool

from backend.rag import doc_to_source, format_docs, retrieve

MAX_OUTPUT_CHARS = 4000
IS_WINDOWS = platform.system() == "Windows"

# Must start with a letter/digit, so a value like "-t" can never act as a flag.
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.\-]{0,251}[A-Za-z0-9])?$")


def _validate_host(value: str) -> str:
    value = value.strip()
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass
    if _HOSTNAME_RE.match(value):
        return value
    raise ValueError(f"'{value}' is not a valid hostname or IP address.")


def _run(command: list[str], label: str, timeout: int) -> str:
    """Run one fixed command without a shell and return a labelled observation."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError:
        return f"OBSERVATION ({label}): this command is not available on this system."
    except subprocess.TimeoutExpired:
        return f"OBSERVATION ({label}): the command timed out after {timeout} seconds."

    output = ((result.stdout or "") + (result.stderr or "")).strip()
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n...[output truncated]"
    return f"OBSERVATION ({label}, exit code {result.returncode}):\n{output}"


# --------------------------------------------------------------------------
# Knowledge tool (lets the agent decide when to consult the handbook)
# --------------------------------------------------------------------------
@tool(response_format="content_and_artifact")
def search_knowledge_base(query: str):
    """Search the networking troubleshooting handbook (PDF).
    Use it to learn which checks suit a problem, how a command works, or how to
    interpret a diagnostic result. The input should be a short, specific search query."""
    docs = retrieve(query)
    if not docs:
        return "No relevant passages were found in the handbook.", []
    return format_docs(docs), [doc_to_source(d) for d in docs]


# --------------------------------------------------------------------------
# Read-only diagnostic tools
# --------------------------------------------------------------------------
@tool
def get_ip_configuration(detailed: bool = False) -> str:
    """Read this computer's network adapter configuration (IP address, subnet mask,
    default gateway). Set detailed=True to also see DHCP status, DHCP server and DNS
    servers. Read-only; it changes nothing."""
    if not IS_WINDOWS:
        return "OBSERVATION (ipconfig): only available on Windows."
    command = ["ipconfig", "/all"] if detailed else ["ipconfig"]
    return _run(command, " ".join(command), timeout=15)


@tool
def ping_host(host: str) -> str:
    """Send 4 ping (ICMP echo) requests to an IP address or hostname to test whether
    it replies. Use an IP address to test connectivity without involving DNS.
    A failed ping is not proof that a host is offline."""
    try:
        host = _validate_host(host)
    except ValueError as error:
        return f"Invalid input: {error}"
    count_flag = "-n" if IS_WINDOWS else "-c"
    return _run(["ping", count_flag, "4", host], f"ping {host}", timeout=30)


@tool
def dns_lookup(hostname: str, dns_server: str = "") -> str:
    """Resolve a hostname with nslookup to check whether DNS name resolution works.
    Optionally give a DNS server IP address to query that server instead of the
    default one."""
    try:
        hostname = _validate_host(hostname)
        command = ["nslookup", hostname]
        if dns_server.strip():
            ipaddress.ip_address(dns_server.strip())
            command.append(dns_server.strip())
    except ValueError as error:
        return f"Invalid input: {error}"
    return _run(command, " ".join(command), timeout=20)


DIAGNOSTIC_TOOLS = [get_ip_configuration, ping_host, dns_lookup]
ALL_TOOLS = [search_knowledge_base] + DIAGNOSTIC_TOOLS