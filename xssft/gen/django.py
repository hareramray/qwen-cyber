"""Django templates (Python views + Django templating).

Sinks: HttpResponse string building, mark_safe, |safe filter, {% autoescape off %},
format_html misuse, and the subtle javascript:-in-href case where autoescaping is
on but the URL context still allows script. Safe counterparts: {{ }} autoescaping,
render() with escaped context, django.utils.html.escape.
"""
from __future__ import annotations

import random

from .builder import (Code, example, finding, fix, names, poc, safe_report, sink,
                      src, step, vuln_report)

LANG = "Python"
FW = "Django"


# --- vulnerable: HttpResponse string concat (reflected) ---------------------
def dj_httpresponse_concat(rng: random.Random) -> dict:
    nm = names(rng)
    path = f"{nm['route']}/views.py"
    c = Code()
    c.add("from django.http import HttpResponse", "")
    c.add(f"def {nm['handler']}(request):")
    src_line = c.add(f"    {nm['var']} = request.GET.get('{nm['param']}', '')")
    sink_line = c.add(f"    return HttpResponse('<h1>Results for ' + {nm['var']} + '</h1>')")
    files = [{"path": path, "content": c.text()}]
    report = vuln_report(
        sources=[src(path, src_line, f"request.GET.get('{nm['param']}', '')", "http_query")],
        sinks=[sink(path, sink_line, f"HttpResponse('<h1>Results for ' + {nm['var']} + '</h1>')",
                    "httpresponse_concat", False, "Raw string is returned as the HTML body")],
        findings=[finding(
            "xss-1", "reflected", "html_body",
            "A GET parameter is concatenated into an HTML response without escaping.",
            [step(path, src_line, f"request.GET.get('{nm['param']}', '')", "attacker-controlled query parameter"),
             step(path, sink_line, f"HttpResponse('<h1>Results for ' + {nm['var']} + '</h1>')",
                  "value interpolated into raw HTML")],
            "high", "Reflected XSS from a crafted link; no template escaping applies.",
            "Building HTML by string concatenation bypasses Django's template autoescaping entirely.",
            fix("framework_safe_api",
                "Render through a template so autoescaping applies.",
                f"return render(request, '{nm['route']}.html', {{'{nm['param']}': {nm['var']}}})",
                "Or escape explicitly with django.utils.html.escape before concatenation.",
                "Set a Content-Security-Policy header."),
            poc("<script>alert(1)</script>",
                f"Request the view with ?{nm['param']}=<script>alert(1)</script>",
                "Script executes in the response page"))])
    return example(files, report, LANG, FW, "dj_httpresponse_concat", "vuln")


# --- vulnerable: mark_safe on user input ------------------------------------
def dj_mark_safe(rng: random.Random) -> dict:
    nm = names(rng)
    path = f"{nm['route']}/views.py"
    c = Code()
    c.add("from django.shortcuts import render", "from django.utils.safestring import mark_safe", "")
    c.add(f"def {nm['handler']}(request):")
    src_line = c.add(f"    {nm['var']} = request.GET.get('{nm['param']}', '')")
    sink_line = c.add(f"    {nm['noun']} = mark_safe({nm['var']})")
    c.add(f"    return render(request, '{nm['route']}.html', {{'{nm['noun']}': {nm['noun']}}})")
    files = [{"path": path, "content": c.text()}]
    report = vuln_report(
        sources=[src(path, src_line, f"request.GET.get('{nm['param']}', '')", "http_query")],
        sinks=[sink(path, sink_line, f"mark_safe({nm['var']})", "mark_safe", False,
                    "mark_safe flags the string as safe, disabling autoescape in the template")],
        findings=[finding(
            "xss-1", "reflected", "html_body",
            "User input is wrapped in mark_safe, so the template will not escape it.",
            [step(path, src_line, f"request.GET.get('{nm['param']}', '')", "attacker-controlled input"),
             step(path, sink_line, f"mark_safe({nm['var']})", "escaping disabled for this value")],
            "high", "mark_safe on untrusted input defeats Django's main XSS defense.",
            "mark_safe should only wrap trusted, already-safe HTML; applied to request data it enables injection.",
            fix("framework_safe_api",
                "Drop mark_safe and let the template autoescape the value.",
                f"return render(request, '{nm['route']}.html', {{'{nm['noun']}': {nm['var']}}})",
                "If limited HTML is needed, run bleach.clean() with an allow-list first.",
                "CSP as defense in depth."),
            poc("<script>alert(1)</script>",
                f"Request with ?{nm['param']}=<script>alert(1)</script>",
                "Unescaped script renders in the template"))])
    return example(files, report, LANG, FW, "dj_mark_safe", "vuln")


# --- vulnerable: |safe filter in template (stored), two files ---------------
def dj_safe_filter(rng: random.Random) -> dict:
    nm = names(rng)
    vpath = f"{nm['route']}/views.py"
    tpath = f"{nm['route']}/templates/{nm['route']}.html"
    v = Code()
    v.add("from django.shortcuts import render", f"from .models import {nm['model']}", "")
    v.add(f"def {nm['handler']}(request):")
    src_line = v.add(f"    {nm['var']} = {nm['model']}.objects.latest('id')")
    v.add(f"    return render(request, '{nm['route']}.html', {{'{nm['noun']}': {nm['var']}.{nm['noun']}}})")
    t = Code()
    t.add("{% extends 'base.html' %}", "{% block content %}", "  <article>")
    sink_line = t.add(f"    {{{{ {nm['noun']}|safe }}}}")
    t.add("  </article>", "{% endblock %}")
    files = [{"path": vpath, "content": v.text()}, {"path": tpath, "content": t.text()}]
    report = vuln_report(
        sources=[src(vpath, src_line, f"{nm['model']}.objects.latest('id')", "database_record")],
        sinks=[sink(tpath, sink_line, f"{{{{ {nm['noun']}|safe }}}}", "safe_filter", False,
                    "The |safe filter turns off autoescaping for this variable")],
        findings=[finding(
            "xss-1", "stored", "html_body",
            "A stored model field is printed through the |safe filter.",
            [step(vpath, src_line, f"{nm['model']}.objects.latest('id')", "user-submitted record from the database"),
             step(tpath, sink_line, f"{{{{ {nm['noun']}|safe }}}}", "autoescaping disabled by |safe")],
            "high", "Persistent XSS: payload stored once, runs for every viewer.",
            "The |safe filter marks the value as trusted HTML; on stored user content this is stored XSS.",
            fix("framework_safe_api",
                "Remove the |safe filter so the value is autoescaped.",
                f"{{{{ {nm['noun']} }}}}",
                "If some HTML is required, sanitize with bleach before saving.",
                "CSP and output validation on write."),
            poc("<script>alert(1)</script>",
                f"Submit the payload so it is stored in {nm['model']}.{nm['noun']}",
                "Stored script runs when the page is viewed"))])
    return example(files, report, LANG, FW, "dj_safe_filter", "vuln")


# --- vulnerable: autoescape off ---------------------------------------------
def dj_autoescape_off(rng: random.Random) -> dict:
    nm = names(rng)
    tpath = f"{nm['route']}/templates/{nm['route']}.html"
    t = Code()
    t.add("<div class=\"content\">", "{% autoescape off %}")
    sink_line = t.add(f"  {{{{ {nm['noun']} }}}}")
    t.add("{% endautoescape %}", "</div>")
    files = [{"path": tpath, "content": t.text()}]
    report = vuln_report(
        sources=[],
        sinks=[sink(tpath, sink_line, f"{{{{ {nm['noun']} }}}}", "autoescape_off", False,
                    "Inside {% autoescape off %} the variable is not escaped")],
        findings=[finding(
            "xss-1", "stored", "html_body",
            "A variable is rendered inside an {% autoescape off %} block.",
            [step(tpath, sink_line, f"{{{{ {nm['noun']} }}}}", "rendered with escaping disabled")],
            "high", "Any user-controlled value reaching this block is injectable.",
            "Disabling autoescape for a region means every variable in it is emitted as raw HTML.",
            fix("framework_safe_api",
                "Remove the autoescape off wrapper.",
                f"{{{{ {nm['noun']} }}}}",
                "Scope escaping per value with the escape filter if a few trusted fragments need raw HTML.",
                "CSP as defense in depth."),
            poc("<script>alert(1)</script>",
                f"Provide the payload as the {nm['noun']} context value",
                "Raw script renders"))])
    return example(files, report, LANG, FW, "dj_autoescape_off", "vuln")


# --- vulnerable context trap: autoescape ON but javascript: URL -------------
def dj_href_url_trap(rng: random.Random) -> dict:
    nm = names(rng)
    vpath = f"{nm['route']}/views.py"
    tpath = f"{nm['route']}/templates/{nm['route']}.html"
    v = Code()
    v.add("from django.shortcuts import render", "")
    v.add(f"def {nm['handler']}(request):")
    src_line = v.add(f"    {nm['var']} = request.GET.get('{nm['param']}', '')")
    v.add(f"    return render(request, '{nm['route']}.html', {{'target': {nm['var']}}})")
    t = Code()
    t.add("<p>")
    sink_line = t.add("  <a href=\"{{ target }}\">continue</a>")
    t.add("</p>")
    files = [{"path": vpath, "content": v.text()}, {"path": tpath, "content": t.text()}]
    report = vuln_report(
        sources=[src(vpath, src_line, f"request.GET.get('{nm['param']}', '')", "http_query")],
        sinks=[sink(tpath, sink_line, "<a href=\"{{ target }}\">", "href_attribute", False,
                    "Autoescaping protects HTML syntax but not the javascript: URL scheme")],
        findings=[finding(
            "xss-1", "reflected", "url",
            "User input is placed in an href; autoescaping does not neutralise a javascript: URL.",
            [step(vpath, src_line, f"request.GET.get('{nm['param']}', '')", "attacker-controlled URL"),
             step(tpath, sink_line, "<a href=\"{{ target }}\">", "javascript: scheme survives HTML escaping")],
            "medium", "Requires a click but executes script despite autoescaping being on.",
            "Django autoescaping encodes <, >, \" and & but leaves the URL scheme intact, so a "
            "javascript: value in an href still runs. HTML escaping is the wrong defense for a URL context.",
            fix("url_validation",
                "Validate the scheme; allow only http/https or relative URLs.",
                "if not target.startswith(('http://', 'https://', '/')): target = '#'",
                "Use django.utils.http.url_has_allowed_host_and_scheme for redirect-style URLs.",
                "CSP to block inline script execution."),
            poc("javascript:alert(1)",
                f"Request with ?{nm['param']}=javascript:alert(1) then click the link",
                "Script runs on click even though the template autoescapes"))])
    return example(files, report, LANG, FW, "dj_href_url_trap", "vuln")


# --- safe: autoescaped interpolation ----------------------------------------
def dj_safe_autoescape(rng: random.Random) -> dict:
    nm = names(rng)
    vpath = f"{nm['route']}/views.py"
    tpath = f"{nm['route']}/templates/{nm['route']}.html"
    v = Code()
    v.add("from django.shortcuts import render", "")
    v.add(f"def {nm['handler']}(request):")
    v.add(f"    {nm['var']} = request.GET.get('{nm['param']}', '')")
    v.add(f"    return render(request, '{nm['route']}.html', {{'{nm['noun']}': {nm['var']}}})")
    t = Code()
    t.add("<section>")
    safe_line = t.add(f"  <h2>{{{{ {nm['noun']} }}}}</h2>")
    t.add("</section>")
    files = [{"path": vpath, "content": v.text()}, {"path": tpath, "content": t.text()}]
    report = safe_report(
        sinks=[sink(tpath, safe_line, f"{{{{ {nm['noun']} }}}}", "template_variable", True,
                    "Django autoescapes variables in HTML body context by default")],
        safe_patterns=[{"file": tpath, "line": safe_line, "pattern": f"{{{{ {nm['noun']} }}}}",
                        "why": "Default autoescaping encodes HTML metacharacters, so query input is inert in body context"}])
    return example(files, report, LANG, FW, "dj_safe_autoescape", "safe")


# --- safe: escape() helper --------------------------------------------------
def dj_safe_escape(rng: random.Random) -> dict:
    nm = names(rng)
    path = f"{nm['route']}/views.py"
    c = Code()
    c.add("from django.http import HttpResponse", "from django.utils.html import escape", "")
    c.add(f"def {nm['handler']}(request):")
    src_line = c.add(f"    {nm['var']} = request.GET.get('{nm['param']}', '')")
    safe_line = c.add(f"    return HttpResponse('<h1>' + escape({nm['var']}) + '</h1>')")
    files = [{"path": path, "content": c.text()}]
    report = safe_report(
        sinks=[sink(path, safe_line, f"escape({nm['var']})", "httpresponse_concat", True,
                    "Value is HTML-escaped for the body context before concatenation")],
        safe_patterns=[{"file": path, "line": safe_line, "pattern": f"escape({nm['var']})",
                        "why": "django.utils.html.escape encodes the value for HTML body, matching the output context"}],
        sources=[src(path, src_line, f"request.GET.get('{nm['param']}', '')", "http_query")])
    return example(files, report, LANG, FW, "dj_safe_escape", "safe")


ALL = [dj_httpresponse_concat, dj_mark_safe, dj_safe_filter, dj_autoescape_off,
       dj_href_url_trap, dj_safe_autoescape, dj_safe_escape]
