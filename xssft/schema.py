"""Shared prompt format and output schema.

The same prompt builder is used for training data, evaluation and the scanner,
so the model always sees code in exactly one format: every line is prefixed
with its number. That way the model copies line numbers instead of counting
them, which is where small models usually go wrong.
"""
from __future__ import annotations

import json
import re
from typing import Any

XSS_TYPES = ["reflected", "stored", "dom", "mutation"]
CONTEXTS = ["html_body", "html_attribute", "js_string", "js_code", "url", "css"]
FIX_STRATEGIES = ["contextual_escaping", "framework_safe_api", "sanitizer", "url_validation", "csp"]
SEVERITIES = ["low", "medium", "high", "critical"]

SYSTEM_PROMPT = """You are an application-security analyst specialised in cross-site scripting (XSS).
Given source files (each line prefixed with its line number), trace untrusted input to output sinks and decide whether an XSS vulnerability exists.

Rules:
- Cite exact line numbers from the listing. Quote code verbatim from the cited line.
- Do not report a sink that is correctly neutralised for its output context (framework auto-escaping such as React JSX, Jinja2/Django autoescape, Razor, ERB, Blade {{ }}, Go html/template, Angular interpolation and [innerHTML] sanitisation; correct contextual encoding; sanitisers used as the last step; values that are not attacker-controlled).
- Escaping only helps when it matches the context: HTML escaping does not make a javascript: URL, an unquoted attribute or a JavaScript string inside an event handler safe.
- Proof-of-concept payloads are for authorised testing only.

Answer with one JSON object and nothing else, with keys in this order:
{"sources": [{"file", "line", "code", "kind"}],
 "sinks": [{"file", "line", "code", "kind", "neutralized", "reason"}],
 "vulnerable": bool,
 "findings": [{"id", "type": reflected|stored|dom|mutation,
   "context": html_body|html_attribute|js_string|js_code|url|css, "context_detail",
   "path": [{"file", "line", "code", "note"}], "severity": low|medium|high|critical, "severity_rationale",
   "explanation",
   "fix": {"strategy": contextual_escaping|framework_safe_api|sanitizer|url_validation|csp, "summary", "patch", "alternatives", "defense_in_depth"},
   "poc": {"payload", "delivery", "expected", "scope"}}],
 "safe_patterns": [{"file", "line", "pattern", "why"}]}"""


def number_lines(text: str) -> str:
    lines = text.split("\n")
    width = max(3, len(str(len(lines))))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, 1))


def build_user_prompt(files: list[dict[str, str]], language: str = "", framework: str = "") -> str:
    """files: [{"path": ..., "content": ...}]"""
    head = "Analyze the following code for cross-site scripting (XSS)."
    if language or framework:
        head += f"\nStack: {', '.join(x for x in (language, framework) if x)}"
    parts = [head]
    for f in files:
        parts.append(f"File: {f['path']}\n```\n{number_lines(f['content'])}\n```")
    parts.append("Respond with the JSON report.")
    return "\n\n".join(parts)


def build_messages(files, language="", framework="") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(files, language, framework)},
    ]


def dump_report(report: dict[str, Any]) -> str:
    """Canonical serialisation used for training targets (compact but readable)."""
    return json.dumps(report, ensure_ascii=False, indent=1)


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_report(text: str) -> dict[str, Any] | None:
    """Best-effort parse of a model answer. Returns None if no valid JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) and "vulnerable" in obj else None


def validate_report(report: dict[str, Any], files: list[dict[str, str]]) -> list[str]:
    """Structural + grounding checks. Returns a list of problems (empty = valid).

    Grounding: every cited (file, line) must exist and its `code` must appear on
    that line. Used to filter teacher-distilled data and to score model output.
    """
    problems: list[str] = []
    by_path = {f["path"]: f["content"].split("\n") for f in files}

    def check_loc(loc: dict, where: str):
        path, line, code = loc.get("file"), loc.get("line"), loc.get("code", "")
        if path not in by_path:
            problems.append(f"{where}: unknown file {path!r}")
            return
        if not isinstance(line, int) or not 1 <= line <= len(by_path[path]):
            problems.append(f"{where}: line {line!r} out of range")
            return
        if code and _norm(code) not in _norm(by_path[path][line - 1]):
            problems.append(f"{where}: code {code!r} not on {path}:{line}")

    if not isinstance(report.get("vulnerable"), bool):
        problems.append("vulnerable must be a bool")
    for i, s in enumerate(report.get("sources", [])):
        check_loc(s, f"sources[{i}]")
    for i, s in enumerate(report.get("sinks", [])):
        check_loc(s, f"sinks[{i}]")
    findings = report.get("findings", [])
    if report.get("vulnerable") and not findings:
        problems.append("vulnerable=true but no findings")
    if report.get("vulnerable") is False and findings:
        problems.append("vulnerable=false but findings present")
    for i, f in enumerate(findings):
        if f.get("type") not in XSS_TYPES:
            problems.append(f"findings[{i}].type invalid: {f.get('type')!r}")
        if f.get("context") not in CONTEXTS:
            problems.append(f"findings[{i}].context invalid: {f.get('context')!r}")
        if f.get("severity") not in SEVERITIES:
            problems.append(f"findings[{i}].severity invalid")
        if (f.get("fix") or {}).get("strategy") not in FIX_STRATEGIES:
            problems.append(f"findings[{i}].fix.strategy invalid")
        if not f.get("path"):
            problems.append(f"findings[{i}].path empty")
        for j, step in enumerate(f.get("path", [])):
            check_loc(step, f"findings[{i}].path[{j}]")
    return problems


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
