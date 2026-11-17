"""Primitives shared by every template generator.

A template builds source code line by line and, because it writes the code,
knows exactly which line each source/sink lands on. That line number goes
straight into the report, so every generated example passes
``schema.validate_report`` by construction.
"""
from __future__ import annotations

import random
from typing import Callable


class Code:
    """Accumulates lines of a single file and hands back 1-based line numbers."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, *lines: str) -> int:
        """Append one or more lines; return the 1-based number of the LAST one."""
        for ln in lines:
            self.lines.append(ln)
        return len(self.lines)

    @property
    def n(self) -> int:
        return len(self.lines)

    def text(self) -> str:
        return "\n".join(self.lines)


# ---------------------------------------------------------------------------
# Report fragment helpers
# ---------------------------------------------------------------------------

def src(file: str, line: int, code: str, kind: str) -> dict:
    return {"file": file, "line": line, "code": code, "kind": kind}


def sink(file: str, line: int, code: str, kind: str, neutralized: bool, reason: str) -> dict:
    return {"file": file, "line": line, "code": code, "kind": kind,
            "neutralized": neutralized, "reason": reason}


def step(file: str, line: int, code: str, note: str) -> dict:
    return {"file": file, "line": line, "code": code, "note": note}


def finding(fid: str, xtype: str, context: str, context_detail: str, path: list[dict],
            severity: str, severity_rationale: str, explanation: str,
            fix: dict, poc: dict) -> dict:
    return {"id": fid, "type": xtype, "context": context, "context_detail": context_detail,
            "path": path, "severity": severity, "severity_rationale": severity_rationale,
            "explanation": explanation, "fix": fix, "poc": poc}


def fix(strategy: str, summary: str, patch: str, alternatives: str, defense_in_depth: str) -> dict:
    return {"strategy": strategy, "summary": summary, "patch": patch,
            "alternatives": alternatives, "defense_in_depth": defense_in_depth}


def poc(payload: str, delivery: str, expected: str, scope: str = "authorized testing only") -> dict:
    return {"payload": payload, "delivery": delivery, "expected": expected, "scope": scope}


def vuln_report(sources: list[dict], sinks: list[dict], findings: list[dict]) -> dict:
    return {"sources": sources, "sinks": sinks, "vulnerable": True,
            "findings": findings, "safe_patterns": []}


def safe_report(sinks: list[dict], safe_patterns: list[dict], sources: list[dict] | None = None) -> dict:
    return {"sources": sources or [], "sinks": sinks, "vulnerable": False,
            "findings": [], "safe_patterns": safe_patterns}


def example(files: list[dict], report: dict, language: str, framework: str,
            template: str, label: str) -> dict:
    """`label` in {vuln, safe}; `template` is the generator id for stratified splits."""
    return {"files": files, "report": report, "language": language,
            "framework": framework, "template": template, "label": label}


# ---------------------------------------------------------------------------
# Randomised identifiers so the model learns the data flow, not the names
# ---------------------------------------------------------------------------

_NOUNS = ["bio", "comment", "nickname", "title", "message", "note", "about", "status",
          "headline", "summary", "caption", "tagline", "greeting", "signature", "label",
          "description", "subject", "feedback", "review", "snippet", "remark", "blurb"]
_PARAMS = ["q", "name", "query", "search", "term", "input", "value", "text", "data",
           "keyword", "ref", "msg", "content", "user", "id", "page", "redirect", "next"]
_ROUTES = ["profile", "search", "comment", "post", "user", "page", "view", "item",
           "article", "note", "message", "feedback", "preview", "render", "show"]
_HANDLERS = ["handleView", "render", "show", "display", "load", "mount", "update",
             "onLoad", "init", "build", "draw", "present"]
_CLASSES = ["ProfileController", "SearchController", "CommentController", "PageController",
            "UserController", "ViewController", "ContentController", "MessageController"]
_MODELS = ["Comment", "Post", "Profile", "Note", "Message", "Review", "Article", "Feedback"]


def names(rng: random.Random) -> dict:
    """A bundle of distinct identifiers for one example."""
    return {
        "noun": rng.choice(_NOUNS),
        "param": rng.choice(_PARAMS),
        "param2": rng.choice(_PARAMS),
        "route": rng.choice(_ROUTES),
        "handler": rng.choice(_HANDLERS),
        "cls": rng.choice(_CLASSES),
        "model": rng.choice(_MODELS),
        "var": rng.choice(_NOUNS),
    }


Template = Callable[[random.Random], dict]
