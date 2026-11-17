"""Spring Boot (Java) templates: controllers + Thymeleaf / JSP views.

Sinks: th:utext, @ResponseBody HTML string building, JSP ${param.x} / <%= %>,
ResponseEntity with text/html body. Safe counterparts: th:text, returning JSON,
HtmlUtils.htmlEscape.
"""
from __future__ import annotations

import random

from .builder import (Code, example, finding, fix, names, poc, safe_report, sink,
                      src, step, vuln_report)

LANG = "Java"
FW = "Spring Boot"


# --- vulnerable: Thymeleaf th:utext (reflected), two files ------------------
def sb_thymeleaf_utext(rng: random.Random) -> dict:
    nm = names(rng)
    cpath = f"src/main/java/com/app/{nm['cls']}.java"
    tpath = f"src/main/resources/templates/{nm['route']}.html"
    c = Code()
    c.add("package com.app;", "")
    c.add("import org.springframework.stereotype.Controller;",
          "import org.springframework.ui.Model;",
          "import org.springframework.web.bind.annotation.GetMapping;",
          "import org.springframework.web.bind.annotation.RequestParam;", "")
    c.add("@Controller", f"public class {nm['cls']} {{", "")
    c.add(f"    @GetMapping(\"/{nm['route']}\")")
    src_line = c.add(f"    public String {nm['handler']}(@RequestParam String {nm['param']}, Model model) {{")
    c.add(f"        model.addAttribute(\"{nm['noun']}\", {nm['param']});",
          f"        return \"{nm['route']}\";", "    }", "}")
    t = Code()
    t.add("<!DOCTYPE html>", "<html xmlns:th=\"http://www.thymeleaf.org\">", "<body>")
    sink_line = t.add(f"  <div th:utext=\"${{{nm['noun']}}}\"></div>")
    t.add("</body>", "</html>")
    files = [{"path": cpath, "content": c.text()}, {"path": tpath, "content": t.text()}]
    report = vuln_report(
        sources=[src(cpath, src_line, f"@RequestParam String {nm['param']}", "request_param")],
        sinks=[sink(tpath, sink_line, f"th:utext=\"${{{nm['noun']}}}\"", "thymeleaf_utext", False,
                    "th:utext emits unescaped HTML (u = unescaped)")],
        findings=[finding(
            "xss-1", "reflected", "html_body",
            "A request parameter is rendered through Thymeleaf th:utext without escaping.",
            [step(cpath, src_line, f"@RequestParam String {nm['param']}", "attacker-controlled request parameter"),
             step(cpath, src_line + 1, f"model.addAttribute(\"{nm['noun']}\", {nm['param']})", "passed to the view"),
             step(tpath, sink_line, f"th:utext=\"${{{nm['noun']}}}\"", "rendered as raw HTML")],
            "high", "Reflected XSS from a crafted link; th:utext performs no escaping.",
            "th:utext is the unescaped counterpart of th:text and outputs the model value as raw HTML.",
            fix("framework_safe_api",
                "Use th:text, which HTML-escapes the value.",
                f"<div th:text=\"${{{nm['noun']}}}\"></div>",
                "If HTML is required, sanitize server-side with the OWASP Java HTML Sanitizer.",
                "Add a Content-Security-Policy header via Spring Security."),
            poc("<script>alert(1)</script>",
                f"GET /{nm['route']}?{nm['param']}=<script>alert(1)</script>",
                "Script executes in the rendered page"))])
    return example(files, report, LANG, FW, "sb_thymeleaf_utext", "vuln")


# --- vulnerable: @ResponseBody HTML concat ----------------------------------
def sb_responsebody_concat(rng: random.Random) -> dict:
    nm = names(rng)
    cpath = f"src/main/java/com/app/{nm['cls']}.java"
    c = Code()
    c.add("package com.app;", "")
    c.add("import org.springframework.web.bind.annotation.GetMapping;",
          "import org.springframework.web.bind.annotation.RequestParam;",
          "import org.springframework.web.bind.annotation.ResponseBody;",
          "import org.springframework.web.bind.annotation.RestController;", "")
    c.add("@RestController", f"public class {nm['cls']} {{", "")
    c.add(f"    @GetMapping(value = \"/{nm['route']}\", produces = \"text/html\")")
    src_line = c.add(f"    public String {nm['handler']}(@RequestParam String {nm['param']}) {{")
    sink_line = c.add(f"        return \"<h1>Hello \" + {nm['param']} + \"</h1>\";")
    c.add("    }", "}")
    files = [{"path": cpath, "content": c.text()}]
    report = vuln_report(
        sources=[src(cpath, src_line, f"@RequestParam String {nm['param']}", "request_param")],
        sinks=[sink(cpath, sink_line, f"return \"<h1>Hello \" + {nm['param']} + \"</h1>\";",
                    "responsebody_html", False, "String returned as a text/html body with no escaping")],
        findings=[finding(
            "xss-1", "reflected", "html_body",
            "A request parameter is concatenated into an HTML response body.",
            [step(cpath, src_line, f"@RequestParam String {nm['param']}", "attacker-controlled parameter"),
             step(cpath, sink_line, f"\"<h1>Hello \" + {nm['param']} + \"</h1>\"",
                  "interpolated into raw HTML returned as text/html")],
            "high", "Reflected XSS; a controller returning hand-built HTML has no templating defense.",
            "Returning concatenated HTML as text/html skips all view-layer escaping.",
            fix("framework_safe_api",
                "Render via a Thymeleaf view using th:text, or escape the value.",
                f"return \"<h1>Hello \" + HtmlUtils.htmlEscape({nm['param']}) + \"</h1>\";",
                "Prefer returning a template name over building HTML in the controller.",
                "CSP header as defense in depth."),
            poc("<script>alert(1)</script>",
                f"GET /{nm['route']}?{nm['param']}=<script>alert(1)</script>",
                "Script runs in the response"))])
    return example(files, report, LANG, FW, "sb_responsebody_concat", "vuln")


# --- vulnerable: JSP expression ---------------------------------------------
def sb_jsp_param(rng: random.Random) -> dict:
    nm = names(rng)
    tpath = f"src/main/webapp/WEB-INF/views/{nm['route']}.jsp"
    t = Code()
    t.add("<%@ page contentType=\"text/html\" %>", "<html>", "<body>", "  <p>")
    sink_line = t.add(f"    Welcome ${{param.{nm['param']}}}")
    t.add("  </p>", "</body>", "</html>")
    files = [{"path": tpath, "content": t.text()}]
    report = vuln_report(
        sources=[src(tpath, sink_line, f"param.{nm['param']}", "request_param")],
        sinks=[sink(tpath, sink_line, f"${{param.{nm['param']}}}", "jsp_el", False,
                    "JSP EL ${} writes the value to the page without HTML escaping")],
        findings=[finding(
            "xss-1", "reflected", "html_body",
            "A JSP EL expression prints a request parameter unescaped.",
            [step(tpath, sink_line, f"${{param.{nm['param']}}}", "request parameter rendered raw")],
            "high", "Reflected XSS; plain JSP EL does not escape output.",
            "${param.x} resolves the request parameter and emits it verbatim into the HTML body.",
            fix("contextual_escaping",
                "Use JSTL c:out, which escapes by default.",
                f"<c:out value=\"${{param.{nm['param']}}}\"/>",
                "Or enable the fn:escapeXml() function on the value.",
                "Set response CSP headers."),
            poc("<script>alert(1)</script>",
                f"GET the JSP with ?{nm['param']}=<script>alert(1)</script>",
                "Script executes"))])
    return example(files, report, LANG, FW, "sb_jsp_param", "vuln")


# --- vulnerable: ResponseEntity text/html -----------------------------------
def sb_responseentity(rng: random.Random) -> dict:
    nm = names(rng)
    cpath = f"src/main/java/com/app/{nm['cls']}.java"
    c = Code()
    c.add("package com.app;", "")
    c.add("import org.springframework.http.MediaType;",
          "import org.springframework.http.ResponseEntity;",
          "import org.springframework.web.bind.annotation.GetMapping;",
          "import org.springframework.web.bind.annotation.RequestParam;",
          "import org.springframework.web.bind.annotation.RestController;", "")
    c.add("@RestController", f"public class {nm['cls']} {{", "")
    c.add(f"    @GetMapping(\"/{nm['route']}\")")
    src_line = c.add(f"    public ResponseEntity<String> {nm['handler']}(@RequestParam String {nm['param']}) {{")
    sink_line = c.add(f"        return ResponseEntity.ok().contentType(MediaType.TEXT_HTML)")
    c.add(f"            .body(\"<div>\" + {nm['param']} + \"</div>\");")
    c.add("    }", "}")
    files = [{"path": cpath, "content": c.text()}]
    report = vuln_report(
        sources=[src(cpath, src_line, f"@RequestParam String {nm['param']}", "request_param")],
        sinks=[sink(cpath, sink_line + 1, f".body(\"<div>\" + {nm['param']} + \"</div>\")",
                    "responseentity_html", False, "User value placed in a text/html response body")],
        findings=[finding(
            "xss-1", "reflected", "html_body",
            "A request parameter is written into a text/html ResponseEntity body.",
            [step(cpath, src_line, f"@RequestParam String {nm['param']}", "attacker-controlled parameter"),
             step(cpath, sink_line + 1, f".body(\"<div>\" + {nm['param']} + \"</div>\")",
                  "raw HTML body with the value interpolated")],
            "high", "Reflected XSS; the handcrafted HTML body is served as text/html.",
            "Setting MediaType.TEXT_HTML and concatenating user input returns unescaped markup.",
            fix("framework_safe_api",
                "Return JSON, or escape the value for the HTML context.",
                f".body(\"<div>\" + HtmlUtils.htmlEscape({nm['param']}) + \"</div>\")",
                "Better: return a DTO as JSON and render on a framework-escaped client.",
                "CSP header."),
            poc("<script>alert(1)</script>",
                f"GET /{nm['route']}?{nm['param']}=<script>alert(1)</script>",
                "Script runs in the HTML response"))])
    return example(files, report, LANG, FW, "sb_responseentity", "vuln")


# --- safe: Thymeleaf th:text ------------------------------------------------
def sb_safe_thtext(rng: random.Random) -> dict:
    nm = names(rng)
    cpath = f"src/main/java/com/app/{nm['cls']}.java"
    tpath = f"src/main/resources/templates/{nm['route']}.html"
    c = Code()
    c.add("package com.app;", "")
    c.add("import org.springframework.stereotype.Controller;",
          "import org.springframework.ui.Model;",
          "import org.springframework.web.bind.annotation.GetMapping;",
          "import org.springframework.web.bind.annotation.RequestParam;", "")
    c.add("@Controller", f"public class {nm['cls']} {{", "")
    c.add(f"    @GetMapping(\"/{nm['route']}\")")
    c.add(f"    public String {nm['handler']}(@RequestParam String {nm['param']}, Model model) {{",
          f"        model.addAttribute(\"{nm['noun']}\", {nm['param']});",
          f"        return \"{nm['route']}\";", "    }", "}")
    t = Code()
    t.add("<!DOCTYPE html>", "<html xmlns:th=\"http://www.thymeleaf.org\">", "<body>")
    safe_line = t.add(f"  <div th:text=\"${{{nm['noun']}}}\"></div>")
    t.add("</body>", "</html>")
    files = [{"path": cpath, "content": c.text()}, {"path": tpath, "content": t.text()}]
    report = safe_report(
        sinks=[sink(tpath, safe_line, f"th:text=\"${{{nm['noun']}}}\"", "thymeleaf_text", True,
                    "th:text HTML-escapes the model value")],
        safe_patterns=[{"file": tpath, "line": safe_line, "pattern": f"th:text=\"${{{nm['noun']}}}\"",
                        "why": "th:text escapes HTML metacharacters, so the request parameter is inert in body context"}])
    return example(files, report, LANG, FW, "sb_safe_thtext", "safe")


# --- safe: return JSON (no HTML context) ------------------------------------
def sb_safe_json(rng: random.Random) -> dict:
    nm = names(rng)
    cpath = f"src/main/java/com/app/{nm['cls']}.java"
    c = Code()
    c.add("package com.app;", "")
    c.add("import java.util.Map;",
          "import org.springframework.web.bind.annotation.GetMapping;",
          "import org.springframework.web.bind.annotation.RequestParam;",
          "import org.springframework.web.bind.annotation.RestController;", "")
    c.add("@RestController", f"public class {nm['cls']} {{", "")
    c.add(f"    @GetMapping(\"/{nm['route']}\")")
    c.add(f"    public Map<String, String> {nm['handler']}(@RequestParam String {nm['param']}) {{")
    safe_line = c.add(f"        return Map.of(\"{nm['noun']}\", {nm['param']});")
    c.add("    }", "}")
    files = [{"path": cpath, "content": c.text()}]
    report = safe_report(
        sinks=[sink(cpath, safe_line, f"return Map.of(\"{nm['noun']}\", {nm['param']});",
                    "json_body", True,
                    "Returned as an application/json body; the Jackson converter JSON-encodes the value")],
        safe_patterns=[{"file": cpath, "line": safe_line, "pattern": "return Map.of(...)",
                        "why": "A @RestController returning a Map serializes to JSON, not HTML, so there is no HTML sink"}],
        sources=[src(cpath, safe_line - 1, f"@RequestParam String {nm['param']}", "request_param")])
    return example(files, report, LANG, FW, "sb_safe_json", "safe")


ALL = [sb_thymeleaf_utext, sb_responsebody_concat, sb_jsp_param, sb_responseentity,
       sb_safe_thtext, sb_safe_json]
