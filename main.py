#!/usr/bin/env python3
"""
mcpo-tester — an interactive console client for MCPO (MCP-to-OpenAPI) servers.

MCPO (https://github.com/open-webui/mcpo) republishes every MCP tool of an MCP
server as a plain REST/OpenAPI endpoint.  This tool talks to that HTTP API — it
never speaks MCP itself:

  1. fetches the OpenAPI schema(s) the MCPO server publishes,
  2. lets you pick a tool from a menu,
  3. walks you through each parameter (type, enum, default, required),
  4. POSTs the flat JSON body and pretty-prints the response.

Usage:
    python mcpo_tester.py http://localhost:8000
    python mcpo_tester.py http://localhost:8000/memory        # one sub-server
    python mcpo_tester.py http://localhost:8000 --list
    python mcpo_tester.py http://localhost:8000 --tool get_current_time --json '{"timezone":"Europe/Paris"}'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

console = Console()

_MISSING = object()
_HTTP_METHODS = ("post", "put", "get", "patch", "delete")
_SKIP_PATHS = {"/", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass
class Tool:
    server: str
    name: str
    path: str                       # path relative to the MCPO base URL
    method: str = "POST"
    description: str = ""
    body_schema: dict = field(default_factory=dict)
    parameters: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.server}/{self.name}" if self.server else self.name

    @property
    def param_count(self) -> int:
        return len(self.body_schema.get("properties") or {})


@dataclass
class Server:
    name: str
    prefix: str                     # "" or "/memory"
    openapi: dict
    tools: list = field(default_factory=list)


class McpoError(Exception):
    """Anything that stops us from talking to the MCPO server."""


# --------------------------------------------------------------------------- #
# tiny JSON-Schema helpers
# --------------------------------------------------------------------------- #
def resolve_ref(ref: str, doc: dict) -> dict:
    """Resolve a local JSON pointer like '#/components/schemas/Foo'."""
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return {}
    node: Any = doc
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return node if isinstance(node, dict) else {}


def first_line(text: str, width: int = 90) -> str:
    flat = " ".join((text or "").split())
    return flat[: width - 1] + "…" if len(flat) > width else flat


# --------------------------------------------------------------------------- #
# MCPO client: discovery + invocation
# --------------------------------------------------------------------------- #
class McpoClient:
    def __init__(self, base_url: str, timeout: float = 60.0, verbose: bool = False):
        self.base = base_url.rstrip("/")
        self.base_path = urlparse(self.base).path.rstrip("/")
        self.verbose = verbose
        self.http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )

    # -- HTTP ------------------------------------------------------------- #
    def _get_json(self, url: str) -> dict:
        if self.verbose:
            console.log(f"GET {url}")
        try:
            response = self.http.get(url)
        except httpx.HTTPError as exc:
            raise McpoError(f"GET {url} failed: {exc}") from exc
        if response.status_code != 200:
            raise McpoError(f"GET {url} -> HTTP {response.status_code}")
        try:
            data = response.json()
        except ValueError as exc:
            raise McpoError(f"GET {url} did not return JSON") from exc
        if not isinstance(data, dict):
            raise McpoError(f"GET {url} did not return an OpenAPI document")
        return data

    # -- discovery -------------------------------------------------------- #
    def discover(self) -> list[Server]:
        """Return every server published by this MCPO instance, with its tools."""
        try:
            root_spec = self._get_json(f"{self.base}/openapi.json")
        except McpoError as exc:
            raise McpoError(
                f"could not fetch {self.base}/openapi.json ({exc}). "
                "Is this really an MCPO server address?"
            ) from exc

        servers: list[Server] = []

        # 1) tools served at the root (single-server mode, or a sub-server URL)
        root_tools = self._tools_from_spec(root_spec, prefix="", server_name="")
        if root_tools:
            servers.append(Server("", "", root_spec, root_tools))

        # 2) mounted sub-servers, advertised as links in the root description
        for name, prefix in self._subservers(root_spec):
            try:
                spec = self._get_json(f"{self.base}{prefix}/openapi.json")
            except McpoError:
                continue
            tools = self._tools_from_spec(spec, prefix=prefix, server_name=name)
            if tools:
                servers.append(Server(name, prefix, spec, tools))

        return servers

    def _subservers(self, root_spec: dict) -> list[tuple[str, str]]:
        """MCPO lists mounted servers as markdown links in the root app description."""
        description = (root_spec.get("info") or {}).get("description") or ""
        found: list[tuple[str, str]] = []
        for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", description):
            label, href = match.group(1).strip(), match.group(2).strip()
            path = urlparse(href).path
            if not path.endswith("/docs"):
                continue
            prefix = path[: -len("/docs")]
            if self.base_path and prefix.startswith(self.base_path):
                prefix = prefix[len(self.base_path):]
            prefix = "/" + prefix.strip("/")
            if prefix != "/":
                found.append((label, prefix))
        seen, unique = set(), []
        for item in found:
            if item[1] not in seen:
                seen.add(item[1])
                unique.append(item)
        return unique

    def _tools_from_spec(self, spec: dict, prefix: str, server_name: str) -> list[Tool]:
        tools: list[Tool] = []
        for path, item in (spec.get("paths") or {}).items():
            if not isinstance(item, dict) or path in _SKIP_PATHS:
                continue
            for method in _HTTP_METHODS:
                op = item.get(method)
                if not isinstance(op, dict):
                    continue
                body_schema: dict = {}
                content = ((op.get("requestBody") or {}).get("content") or {})
                if content:
                    ctype = "application/json" if "application/json" in content else next(iter(content))
                    body_schema = content[ctype].get("schema") or {}
                parameters = [
                    p for p in (item.get("parameters") or [])
                    if p.get("in") in ("query", "path", "header")
                ]
                tools.append(
                    Tool(
                        server=server_name,
                        name=path.rstrip("/").split("/")[-1],
                        path=prefix + path,
                        method=method.upper(),
                        description=(op.get("description") or op.get("summary") or "").strip(),
                        body_schema=body_schema,
                        parameters=parameters,
                    )
                )
        return tools

    # -- invocation ------------------------------------------------------- #
    def call(self, tool: Tool, body: dict, params: Optional[dict] = None) -> bool:
        url = self.base + tool.path
        console.print()
        console.print(
            Panel(
                Syntax(
                    json.dumps(body, indent=2, ensure_ascii=False) if body else "{}",
                    "json",
                    background_color="default",
                    word_wrap=True,
                ),
                title=f"[bold]{tool.method} {url}[/bold]",
                subtitle=first_line(tool.description) or None,
                border_style="cyan",
            )
        )
        try:
            started = time.perf_counter()
            response = self.http.request(
                tool.method,
                url,
                json=body if body else None,
                params=params or None,
                headers={"Content-Type": "application/json"},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        except httpx.HTTPError as exc:
            console.print(f"[bold red]✗ Request failed:[/bold red] {exc}")
            return False
        render_response(response, elapsed_ms)
        return response.status_code < 400


def render_response(response: httpx.Response, elapsed_ms: float) -> None:
    status = response.status_code
    style = "green" if status < 300 else "yellow" if status < 500 else "red"
    icon = "✓" if status < 300 else "✗"
    console.print(
        f"[{style}]{icon} HTTP {status}[/{style}] "
        f"[dim]{response.reason_phrase} · {elapsed_ms:.0f} ms · {len(response.content)} bytes[/dim]"
    )
    console.print(f"[dim]content-type: {response.headers.get('content-type', 'n/a')}[/dim]")

    text = response.text
    if not text.strip():
        console.print("[dim](empty body)[/dim]")
        return

    payload: Any = None
    if "json" in response.headers.get("content-type", ""):
        try:
            payload = response.json()
        except ValueError:
            payload = None
    if payload is None:
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None

    if payload is None:
        console.print(Panel(text, title="response", border_style=style, highlight=True))
        return

    # MCPO error envelope: {"detail": {"message": ..., "data"/"error": ...}}
    if status >= 400 and isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
        detail = payload["detail"]
        message = detail.get("message", json.dumps(detail))
        console.print(Panel(str(message), title="[bold red]error[/bold red]", border_style="red"))
        rest = {k: v for k, v in detail.items() if k != "message"}
        if rest:
            console.print(RichJSON.from_data(rest))
        return

    console.print(Panel(RichJSON.from_data(payload), title="response", border_style=style))


# --------------------------------------------------------------------------- #
# interactive parameter builder
# --------------------------------------------------------------------------- #
class ArgumentBuilder:
    """Turns a tool's JSON Schema into a guided prompt sequence."""

    def __init__(self) -> None:
        self.spec: dict = {}
        self.local_root: dict = {}

    # -- schema plumbing -------------------------------------------------- #
    def deref(self, schema: Any, _depth: int = 0) -> dict:
        if not isinstance(schema, dict) or _depth > 20:
            return schema if isinstance(schema, dict) else {}
        if "$ref" in schema:
            target = resolve_ref(schema["$ref"], self.spec) or resolve_ref(schema["$ref"], self.local_root)
            schema = {**self.deref(target, _depth + 1), **{k: v for k, v in schema.items() if k != "$ref"}}
        if "allOf" in schema:
            merged: dict = {}
            for sub in schema["allOf"]:
                merged.update(self.deref(sub, _depth + 1))
            schema = {**merged, **{k: v for k, v in schema.items() if k != "allOf"}}
        return schema

    def example(self, schema: Any, _depth: int = 0) -> Any:
        schema = self.deref(schema)
        if _depth > 4:
            return None
        stype = schema.get("type")
        if isinstance(stype, list):
            stype = next((t for t in stype if t != "null"), None)
        if schema.get("enum"):
            return schema["enum"][0]
        if schema.get("default") is not None:
            return schema["default"]
        if stype == "object" or (stype is None and schema.get("properties")):
            return {k: self.example(v, _depth + 1) for k, v in (schema.get("properties") or {}).items()}
        if stype == "array":
            return [self.example(schema.get("items") or {}, _depth + 1)]
        if stype == "integer":
            return 0
        if stype == "number":
            return 0.0
        if stype == "boolean":
            return False
        if stype == "string":
            return ""
        return None

    # -- prompting -------------------------------------------------------- #
    def build(self, tool: Tool, spec: dict, show_schema: bool = False) -> Optional[tuple[dict, dict]]:
        self.spec = spec
        body_schema = self.deref(tool.body_schema or {})
        self.local_root = {
            "$defs": body_schema.get("$defs") or {},
            "definitions": body_schema.get("definitions") or {},
        }

        if show_schema:
            console.print(
                Panel(
                    Syntax(json.dumps(tool.body_schema or {}, indent=2), "json", background_color="default"),
                    title=f"input schema — {tool.label}",
                    border_style="dim",
                )
            )

        properties = body_schema.get("properties") or {}
        required = set(body_schema.get("required") or [])
        arguments: dict = {}

        if properties:
            console.print(f"\n[bold]Parameters of[/bold] [cyan]{tool.label}[/cyan] "
                          f"[dim]({len(properties)} — press Enter to accept defaults, "
                          f"type !skip to omit)[/dim]")
            for name, subschema in properties.items():
                value = self.prompt_property(name, subschema, name in required)
                if value is not _MISSING:
                    arguments[name] = value
        else:
            console.print(f"\n[dim]{tool.label} takes no parameters.[/dim]")

        query_params = self.build_query_params(tool)

        console.print()
        console.print(
            Panel(
                Syntax(json.dumps(arguments, indent=2, ensure_ascii=False), "json", background_color="default"),
                title="[bold]request body[/bold]",
                border_style="cyan",
            )
        )
        if Confirm.ask("Send this request?", default=True):
            return arguments, query_params

        raw = Prompt.ask("[yellow]Raw JSON body[/yellow] (empty = cancel)", default="")
        if not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            console.print(f"[bold red]✗ Invalid JSON:[/bold red] {exc}")
            return None
        return (parsed if isinstance(parsed, dict) else {}), query_params

    def build_query_params(self, tool: Tool) -> dict:
        result: dict = {}
        for parameter in tool.parameters:
            parameter = self.deref(parameter)
            location = parameter.get("in")
            name = parameter.get("name") or "param"
            if location == "header":
                console.print(f"[dim]Skipping header parameter '{name}' (not supported).[/dim]")
                continue
            value = self.prompt_property(name, parameter.get("schema") or {}, parameter.get("required", False))
            if value is not _MISSING:
                result[name] = value
        return result

    def prompt_property(self, name: str, schema: Any, required: bool) -> Any:
        schema = self.deref(schema)
        stype = schema.get("type")
        if isinstance(stype, list):
            stype = next((t for t in stype if t != "null"), None)
        description = (schema.get("description") or "").strip()
        enum = schema.get("enum")
        default = schema.get("default", _MISSING)

        meta = [stype or "any", "required" if required else "optional"]
        if default is not _MISSING:
            meta.append(f"default={default!r}")
        if enum:
            meta.append("choices: " + ", ".join(str(e) for e in enum))
        console.print(f"\n[bold cyan]{name}[/bold cyan] [dim]({', '.join(meta)})[/dim]")
        if description:
            console.print(f"  [dim italic]{description}[/dim italic]")

        complex_type = stype in ("object", "array") or (
            stype is None and (schema.get("properties") or schema.get("items"))
        )
        if complex_type:
            console.print("  [dim]shape to fill in:[/dim]")
            console.print(
                Syntax(
                    json.dumps(self.example(schema), indent=2, ensure_ascii=False),
                    "json",
                    background_color="default",
                )
            )

        while True:
            default_text = str(default) if default is not _MISSING else ""
            hint = " [dim](true/false)[/dim]" if stype == "boolean" else ""
            raw = Prompt.ask(f"  [green]{name}[/green]{hint}", default=default_text).strip()

            if raw == "!skip":
                return _MISSING
            if raw == "":
                if required:
                    console.print("  [yellow]! required — enter a value, or type !skip to send nothing[/yellow]")
                    continue
                return _MISSING
            try:
                return self.coerce(raw, stype, enum)
            except (ValueError, json.JSONDecodeError) as exc:
                console.print(f"  [red]✗ {exc}[/red]")

    @staticmethod
    def coerce(raw: str, stype: Optional[str], enum: Optional[list]) -> Any:
        if enum:
            for candidate in enum:
                if str(candidate) == raw:
                    return candidate
            raise ValueError(f"expected one of {[str(e) for e in enum]}")
        if stype == "integer":
            return int(raw)
        if stype == "number":
            return float(raw)
        if stype == "boolean":
            lowered = raw.lower()
            if lowered in ("true", "1", "yes", "y", "on"):
                return True
            if lowered in ("false", "0", "no", "n", "off"):
                return False
            raise ValueError("expected true or false")
        if stype in ("object", "array"):
            return json.loads(raw)
        if stype is None:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw  # string


# --------------------------------------------------------------------------- #
# interactive shell
# --------------------------------------------------------------------------- #
def print_tool_table(tools: list[Tool]) -> None:
    table = Table(title="MCPO tools", header_style="bold", box=None, padding=(0, 1))
    table.add_column("#", justify="right", style="dim")
    table.add_column("server", style="magenta", no_wrap=True)
    table.add_column("tool", style="bold cyan", no_wrap=True)
    table.add_column("args", justify="right", style="dim")
    table.add_column("description", style="white", overflow="fold")
    for index, tool in enumerate(tools, start=1):
        table.add_row(
            str(index),
            tool.server or "(root)",
            tool.name,
            str(tool.param_count),
            first_line(tool.description),
        )
    console.print(table)


def choose_tool(tools: list[Tool]) -> Optional[Tool]:
    print_tool_table(tools)
    while True:
        answer = Prompt.ask("Select a tool number [dim](q = quit)[/dim]", default="q").strip().lower()
        if answer in ("q", "quit", "exit", ""):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(tools):
            return tools[int(answer) - 1]
        console.print(f"[yellow]Enter a number between 1 and {len(tools)}.[/yellow]")


def find_tool(tools: list[Tool], name: str) -> Optional[Tool]:
    wanted = name.strip().lower()
    for tool in tools:
        if wanted in (tool.name.lower(), tool.label.lower()):
            return tool
    return None


def interactive(client: McpoClient, tools: list[Tool], spec: dict, show_schema: bool) -> int:
    builder = ArgumentBuilder()
    while True:
        try:
            tool = choose_tool(tools)
            if tool is None:
                return 0
            built = builder.build(tool, spec, show_schema=show_schema)
            if built is None:
                console.print("[yellow]Cancelled.[/yellow]")
            else:
                body, query = built
                client.call(tool, body, query)
            console.print()
            if not Confirm.ask("Call another tool?", default=False):
                return 0
        except KeyboardInterrupt:
            console.print("\n[dim]bye[/dim]")
            return 130


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mcpo-tester",
        description="Interactive console client for MCPO (MCP-to-OpenAPI) servers.",
    )
    parser.add_argument(
        "url",
        help="base URL of the MCPO server (e.g. http://localhost:8000). "
             "A sub-server URL such as http://localhost:8000/time also works.",
    )
    parser.add_argument("-t", "--timeout", type=float, default=60.0, help="request timeout in seconds (default: 60)")
    parser.add_argument("-l", "--list", action="store_true", help="list the discovered tools and exit")
    parser.add_argument("--tool", metavar="NAME", help="non-interactive: call this tool (use together with --json)")
    parser.add_argument("--json", dest="body", metavar="JSON", help="JSON body to send with --tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every HTTP request")
    parser.add_argument("--show-schema", action="store_true", help="print the raw input schema of the selected tool")
    args = parser.parse_args(argv)

    client = McpoClient(args.url, timeout=args.timeout, verbose=args.verbose)
    try:
        with console.status("[cyan]Fetching OpenAPI schema…[/cyan]"):
            servers = client.discover()
    except McpoError as exc:
        console.print(f"[bold red]✗[/bold red] {exc}")
        return 2

    tools = [tool for server in servers for tool in server.tools]
    if not tools:
        console.print("[bold red]✗ No callable tools found.[/bold red]")
        console.print(f"  Checked [cyan]{client.base}/openapi.json[/cyan]")
        console.print("  If this MCPO instance serves several servers, try a sub-server URL "
                      "such as [cyan]http://host:8000/memory[/cyan].")
        return 2

    console.print(
        f"[green]✓[/green] Connected to [bold]{client.base}[/bold] — "
        f"found [bold]{len(tools)}[/bold] tool(s) in {len(servers)} server(s)."
    )

    if args.list:
        print_tool_table(tools)
        return 0

    if args.tool:
        tool = find_tool(tools, args.tool)
        if tool is None:
            console.print(f"[bold red]✗ Unknown tool '{args.tool}'.[/bold red]")
            print_tool_table(tools)
            return 2
        try:
            body = json.loads(args.body) if args.body else {}
        except json.JSONDecodeError as exc:
            console.print(f"[bold red]✗ --json is not valid JSON:[/bold red] {exc}")
            return 2
        return 0 if client.call(tool, body) else 1

    # merge all servers' specs so $refs resolve regardless of which server a tool came from
    merged_spec: dict = {"components": {"schemas": {}}, "$defs": {}}
    for server in servers:
        merged_spec["components"]["schemas"].update(
            (server.openapi.get("components") or {}).get("schemas") or {}
        )
        merged_spec["$defs"].update(server.openapi.get("$defs") or {})
    return interactive(client, tools, merged_spec, args.show_schema)


if __name__ == "__main__":
    sys.exit(main())



