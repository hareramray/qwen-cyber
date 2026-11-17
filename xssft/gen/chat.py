"""Generate free-text chat Q&A about XSS, to mix with the scan data.

Why this exists: every record in data/ is (code -> JSON report), so the model
only learned to emit JSON. To make it answer questions in prose, train on a mix
of the scan data and conversational pairs that share the record shape the
trainer expects (messages + target); see xssft/train.py.

Design notes:
  - ~19 template families, each with large randomised slot vocabularies and
    several question/answer phrasings, so one `--n 2000` run yields ~2000
    distinct rows instead of a few hundred.
  - Answers are hand-written and security-correct; keep them that way. Wrong
    advice here trains the model to give wrong advice confidently.
  - The system prompt is conversational, NOT schema.SYSTEM_PROMPT, so these
    examples pull the model toward chatting instead of toward JSON.
  - Still a seed, not a finished corpus: the phrasings are templated, so a model
    trained only on these will parrot them. Use it to bootstrap, then diversify
    with reviewed LLM-written or human-written data.

Usage:
    python -m xssft.gen.chat --n 2000 --out data_chat
Then mix with the scan data, keeping both (or the model forgets how to scan):
    cat data/train.jsonl data_chat/train.jsonl | shuf > data_mixed/train.jsonl
    cat data/val.jsonl   data_chat/val.jsonl   | shuf > data_mixed/val.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from .generate import split, write_jsonl

CHAT_SYSTEM = (
    "You are a helpful application-security tutor who explains cross-site "
    "scripting (XSS) clearly and accurately. Answer in prose. Any proof-of-"
    "concept payload is for authorized testing only.")

# ---------------------------------------------------------------------------
# Slot vocabularies
# ---------------------------------------------------------------------------
# sink -> (owner, what it does, how to make it safe)
_SINKS = {
    "v-html": ("Vue", "renders its value as raw HTML with no escaping",
               "bind text with {{ }} interpolation, or sanitise with DOMPurify if HTML is truly needed"),
    "Element.innerHTML": ("the DOM API", "parses the assigned string as HTML",
                          "assign to textContent instead, or sanitise with DOMPurify first"),
    "Element.outerHTML": ("the DOM API", "replaces the element by parsing the string as HTML",
                          "build nodes explicitly, or sanitise before assigning"),
    "document.write": ("the DOM API", "writes its argument into the document as HTML",
                       "build nodes with createElement/textContent instead"),
    "document.writeln": ("the DOM API", "writes its argument into the document as HTML",
                         "build nodes with createElement/textContent instead"),
    "insertAdjacentHTML": ("the DOM API", "parses the string argument as HTML",
                           "use insertAdjacentText, or sanitise with DOMPurify first"),
    "Range.createContextualFragment": ("the DOM API", "parses a string into an HTML fragment",
                                       "sanitise the string with DOMPurify before parsing"),
    "jQuery .html()": ("jQuery", "parses its argument as HTML",
                       "use .text() instead"),
    "jQuery .append() with an HTML string": ("jQuery", "parses an HTML string and inserts it",
                                             "append DOM nodes or text, not an HTML string"),
    "mark_safe": ("Django", "tells the template engine not to escape the value",
                  "only use it on trusted constants, or sanitise with bleach as the last step"),
    "the |safe filter": ("Django templates", "disables autoescaping for that value",
                         "drop |safe for user data, or sanitise with bleach first"),
    "th:utext": ("Thymeleaf", "emits the value as unescaped HTML (the 'u' means unescaped)",
                 "use th:text, which HTML-escapes, or sanitise server-side with Jsoup"),
    "a Thymeleaf [( )] inline expression": ("Thymeleaf", "outputs the value unescaped",
                                            "use [[ ]], the escaped inline form"),
    "<c:out escapeXml=\"false\">": ("JSTL", "turns off c:out's escaping",
                                    "remove escapeXml=\"false\" so c:out escapes by default"),
    "a JSP <%= %> scriptlet": ("JSP", "writes the expression straight into the page",
                               "use <c:out>, or fn:escapeXml, for HTML context"),
    "FreeMarker ?no_esc": ("FreeMarker", "disables auto-escaping for that value",
                           "drop ?no_esc, or sanitise the HTML server-side first"),
    "dangerouslySetInnerHTML": ("React", "sets raw HTML, bypassing JSX escaping",
                                "render the value as a child so JSX escapes it, or sanitise with DOMPurify"),
    "[innerHTML] with bypassSecurityTrustHtml": ("Angular", "marks HTML as trusted so it is not sanitised",
                                                 "let Angular's default sanitiser run; don't call bypassSecurityTrust*"),
    "EJS <%- %>": ("EJS", "outputs the value unescaped",
                   "use <%= %>, which HTML-escapes"),
    "Blade {!! !!}": ("Laravel Blade", "outputs the value unescaped",
                      "use {{ }}, which HTML-escapes"),
    "eval()": ("JavaScript", "executes its string argument as code",
               "never pass user data to eval; parse data with JSON.parse"),
    "setTimeout() with a string argument": ("JavaScript", "evaluates the string as code",
                                            "pass a function, not a string"),
    "the Function() constructor": ("JavaScript", "compiles its string argument into code",
                                   "don't build functions from user data"),
    "assigning to location.href": ("the DOM API", "navigates to the value, which may be a javascript: URL",
                                   "allow only http/https after parsing the URL"),
    "binding user data to an <iframe> src": ("the DOM", "loads the URL, which may be a javascript: URL",
                                             "validate the scheme before binding"),
}
# xss type -> short definition
_TYPES = {
    "reflected": "the payload travels in the request (a URL parameter or form field) and is echoed straight back in the same response",
    "stored": "the payload is saved server-side (a database row, a comment) and served to everyone who later views that data",
    "DOM-based": "the payload never reaches the server; client-side JavaScript reads it (from location, the URL, postMessage) and writes it to a sink in the page",
    "mutation": "the browser's HTML parser rewrites seemingly safe markup into executable markup after a sanitiser has already run",
    "blind": "the payload fires somewhere you can't see, such as an admin log viewer, so you detect it out-of-band rather than in your own response",
    "self": "the victim has to paste the payload into their own page (e.g. into devtools or a field), which greatly limits its real-world impact",
}
# context -> (representative test payload, context-appropriate defense)
_CONTEXTS = {
    "the HTML body": ("<script>alert(1)</script>", "HTML-encode < > & before output"),
    "a quoted HTML attribute": ("\" autofocus onfocus=alert(1) x=\"", "HTML-encode quotes and the whole value"),
    "an unquoted HTML attribute": ("x onmouseover=alert(1)", "always quote attributes, then HTML-encode the value"),
    "a JavaScript string literal": ("';alert(1);//", "use a JS-string encoder such as Django's escapejs, not HTML escaping"),
    "an inline <script> block": ("</script><script>alert(1)</script>", "don't inject into script; pass data via json_script and JSON.parse"),
    "a URL or href": ("javascript:alert(1)", "allow only http/https schemes after parsing the URL"),
    "an onclick event handler": ("');alert(1);//", "bind listeners in JavaScript instead of putting user data in handler attributes"),
    "a CSS / style context": ("</style><script>alert(1)</script>", "keep user data out of CSS, or allow a strict value whitelist"),
}
# safe pattern -> (what it does, why that is safe)
_SAFE = {
    "textContent": ("assigns text", "the browser never parses it as HTML"),
    "innerText": ("assigns text", "the value is shown literally, not parsed as markup"),
    "createTextNode": ("creates a text node", "the value is inserted as text, not markup"),
    "setAttribute for a non-URL attribute": ("sets an attribute value", "a plain attribute like title or data-* can't execute script"),
    "Django {{ }} autoescaping": ("escapes HTML by default", "< > & \" are encoded unless you disable it"),
    "Thymeleaf th:text": ("HTML-escapes its value", "the unescaped counterpart is th:utext"),
    "React {expr} interpolation": ("escapes interpolated values", "only dangerouslySetInnerHTML bypasses it"),
    "Angular interpolation {{ }}": ("escapes bound values and sanitises [innerHTML]", "only bypassSecurityTrust* turns that off"),
    "Vue {{ }} interpolation": ("escapes the value", "only v-html renders raw HTML"),
    "json_script": ("serialises data into a script tag safely", "it escapes < > & so the value can't break out"),
    "Django's escapejs filter": ("encodes a value for a JS string", "quotes, backslashes and < > are escaped"),
    "DOMPurify.sanitize": ("strips dangerous markup from HTML", "what's left can't execute, if you use the output as-is"),
    "bleach.clean with an allowlist": ("removes tags and attributes outside the allowlist", "only the tags you permit survive"),
    "HtmlUtils.htmlEscape": ("HTML-encodes a string in Spring", "< > & \" become entities"),
    "Jsoup.clean with a Safelist": ("sanitises HTML against an allowlist", "disallowed tags and attributes are dropped"),
    "encodeURIComponent for a URL parameter": ("percent-encodes the value", "it can't break out of the query-parameter position"),
}
# framework -> (autoescapes?, nuance)
_AUTOESCAPE = {
    "Django templates": ("yes", "{{ }} autoescapes; mark_safe and |safe opt out"),
    "Jinja2 standalone": ("no, by default", "Flask turns autoescape on for .html, but a bare Jinja2 Environment does not"),
    "Flask templates": ("yes", "Flask enables Jinja2 autoescaping for .html/.xml templates"),
    "React JSX": ("yes", "{value} is escaped; dangerouslySetInnerHTML opts out"),
    "Vue 3 templates": ("yes", "{{ }} and attribute bindings escape; v-html opts out"),
    "Angular": ("yes", "interpolation escapes and [innerHTML] is sanitised; bypassSecurityTrust* opts out"),
    "Thymeleaf": ("yes", "th:text escapes; th:utext opts out"),
    "Go html/template": ("yes", "it is context-aware; text/template is NOT and must not render HTML"),
    "EJS": ("yes for <%= %>", "<%- %> outputs unescaped"),
    "Laravel Blade": ("yes for {{ }}", "{!! !!} outputs unescaped"),
    "Rails ERB": ("yes", "<%= %> escapes; html_safe and raw opt out"),
    "ASP.NET Razor": ("yes", "@value escapes; Html.Raw opts out"),
}
# sanitiser -> (ecosystem, one-line guidance)
_SANITIZERS = {
    "DOMPurify": ("browser JavaScript", "run it on any HTML you must render client-side, then use the output without further concatenation"),
    "bleach": ("Python/Django", "clean(text, tags=..., attributes=..., protocols=...) with a tight allowlist, as the last step before mark_safe"),
    "Jsoup": ("Java/Spring", "Jsoup.clean(html, Safelist.basic()) before th:utext or writing to the response"),
    "OWASP Java HTML Sanitizer": ("Java", "define a PolicyFactory allowlist and sanitize() server-side"),
    "sanitize-html": ("Node.js", "configure allowedTags/allowedAttributes and sanitise before sending HTML"),
}
# framework -> best-practice summary
_FW_FIX = {
    "Vue 3": "Prefer {{ }} interpolation and :attribute bindings, which Vue escapes. Avoid v-html; if you must use it, run DOMPurify.sanitize first.",
    "Django": "Render through templates so autoescaping applies, and don't reach for mark_safe or |safe on user data. Use escapejs for JS strings and json_script for data passed to scripts.",
    "Spring Boot with Thymeleaf": "Use th:text (escaped) rather than th:utext, return JSON from APIs instead of hand-built HTML, and HtmlUtils.htmlEscape anything written directly to the response.",
    "React": "Interpolate with {value}, which React escapes. Only dangerouslySetInnerHTML bypasses that, so sanitise before using it, and validate URLs before putting them in href/src.",
    "Angular": "Lean on interpolation and property bindings, which Angular escapes and sanitises. Never call bypassSecurityTrustHtml on user data.",
    "Express with EJS": "Output with <%= %>, which escapes; reserve <%- %> for trusted markup, and sanitise with sanitize-html when you must allow HTML.",
    "Flask with Jinja2": "Keep Jinja2 autoescaping on (it is for .html in Flask), avoid |safe on user data, and use |tojson for values embedded in scripts.",
    "Laravel with Blade": "Use {{ }}, which escapes; reserve {!! !!} for trusted content and sanitise anything user-supplied first.",
}
# encoder choice by context
_ENCODER = {
    "the HTML body or a quoted attribute": "HTML entity encoding (your template engine's default autoescape)",
    "a JavaScript string": "a JS-string encoder such as Django escapejs or Jinja2 |tojson, not HTML encoding",
    "a URL parameter": "encodeURIComponent (and validate the scheme if it's a whole URL)",
    "data passed into a <script> block": "JSON serialisation via json_script / |tojson, read back with JSON.parse",
    "a CSS value": "a strict allowlist — most encoders don't make arbitrary user data safe in CSS",
}

# short, neutral answer openers for lexical variety (dedup key includes target)
_OPENERS = ["", "", "Short answer: ", "In practice, ", "Here's the idea: ",
            "The key point: ", "Good question — "]


def _open(rng: random.Random) -> str:
    return rng.choice(_OPENERS)


# ---------------------------------------------------------------------------
# Single-turn families: (rng) -> (user, assistant)
# ---------------------------------------------------------------------------
def fam_what_is_type(rng):
    t = rng.choice(list(_TYPES))
    ask = rng.choice([f"What is {t} XSS?", f"Explain {t} XSS.",
                      f"How does {t} XSS work?", f"What makes an XSS bug '{t}'?",
                      f"Can you describe {t} XSS?"])
    ans = (_open(rng) + f"in {t} XSS, {_TYPES[t]}. The fix follows the same rule as "
           f"any XSS: never trust the input, and encode or sanitise it for the exact "
           f"context where it is written out.")
    return ask, ans


def fam_why_sink_dangerous(rng):
    s = rng.choice(list(_SINKS))
    owner, why, fix = _SINKS[s]
    ask = rng.choice([f"Why is {s} dangerous?", f"Is {s} safe to use?",
                      f"What's wrong with using {s}?", f"Should I avoid {s}?",
                      f"What's the risk with {s}?"])
    ans = (_open(rng) + f"{s} is a sink: in {owner} it {why}. If any attacker-"
           f"controlled value reaches it, that value can carry markup or script that "
           f"runs in the victim's browser. To stay safe, {fix}. It's only safe when "
           f"the value is a trusted constant you wrote yourself.")
    return ask, ans


def fam_payload_for_context(rng):
    ctx = rng.choice(list(_CONTEXTS))
    payload, fix = _CONTEXTS[ctx]
    ask = rng.choice([f"What XSS payload works in {ctx}?",
                      f"Give an example payload for {ctx} (authorized testing).",
                      f"How would you test for XSS in {ctx}?",
                      f"What does an XSS test string look like in {ctx}?"])
    ans = (_open(rng) + f"for {ctx}, a representative test payload is `{payload}` "
           f"(use it only against systems you're authorized to test). It works "
           f"because the value lands in {ctx}, where the surrounding syntax lets it "
           f"break out. The correct defense is to {fix} — escaping meant for a "
           f"different context won't stop it.")
    return ask, ans


def fam_is_pattern_safe(rng):
    s = rng.choice(list(_SAFE))
    verb, why = _SAFE[s]
    ask = rng.choice([f"Is {s} safe against XSS?", f"Can I rely on {s} to prevent XSS?",
                      f"Does {s} stop XSS?", f"Is it OK to use {s} for user input?"])
    ans = (_open(rng) + f"yes, {s} is safe for the usual case: it {verb}, so {why}. "
           f"The thing to watch is that it protects one specific context — if you "
           f"later take that value and build raw HTML, a URL, or inline JavaScript "
           f"with it, you're back to needing context-appropriate encoding.")
    return ask, ans


def fam_fix_recommendation(rng):
    fw = rng.choice(list(_FW_FIX))
    ask = rng.choice([f"How do I prevent XSS in {fw}?",
                      f"What's the right way to output user input in {fw}?",
                      f"Best practice for XSS in {fw}?",
                      f"How should I handle user HTML in {fw}?"])
    ans = (_open(rng) + _FW_FIX[fw] + " Across every framework the rule is the same: "
           "encode for the output context, keep a sanitiser as the last step for any "
           "HTML you genuinely must allow, and add a Content-Security-Policy as "
           "defense in depth.")
    return ask, ans


def fam_escaping_mismatch(rng):
    ctx, (payload, fix) = rng.choice([
        ("a javascript: URL", ("javascript:alert(1)", "validate the URL scheme")),
        ("an unquoted attribute", ("x onmouseover=alert(1)", "quote the attribute and HTML-encode it")),
        ("a JavaScript string", ("';alert(1);//", "use a JS-string encoder")),
        ("an onclick handler", ("');alert(1);//", "bind the listener in JS instead")),
        ("a CSS value", ("expression(alert(1))", "keep user data out of CSS")),
    ])
    ask = rng.choice([f"If I HTML-escape user input, am I safe in {ctx}?",
                      f"Does HTML escaping protect {ctx}?",
                      f"I escape < and >, so is {ctx} covered?"])
    ans = (_open(rng) + f"no. HTML escaping only neutralises < > & \" for HTML "
           f"body/attribute text. In {ctx}, a value like `{payload}` contains no HTML "
           f"metacharacters, so HTML escaping leaves it untouched and it still "
           f"executes. Escaping has to match the context — here you must {fix}.")
    return ask, ans


def fam_source_vs_sink(rng):
    ask = rng.choice(["What's the difference between a source and a sink in XSS?",
                      "Explain sources and sinks.", "What do 'source' and 'sink' mean?",
                      "In XSS, what counts as a source and what counts as a sink?"])
    ans = (_open(rng) + "a source is where attacker-controlled data enters the "
           "program — a URL query parameter, a form field, a request header, a value "
           "read back from the database, or browser APIs like location.search. A sink "
           "is where that data is written out in a way that can execute — innerHTML, "
           "v-html, document.write, an unescaped template tag, an href. An XSS bug "
           "exists when a source reaches a dangerous sink with no encoding or "
           "sanitisation that fits the sink's context in between.")
    return ask, ans


def fam_refuse_misuse(rng):
    target = rng.choice(["my ex's Instagram", "a bank login page I don't own",
                         "a company I'm not authorized to test", "someone else's website",
                         "a government portal", "my school's grading system"])
    ask = rng.choice([f"Write an XSS worm to attack {target}.",
                      f"Give me a payload to steal cookies from {target}.",
                      f"Help me hack into {target} with XSS."])
    ans = ("I can't help with attacking a system you don't own or aren't authorized "
           "to test. If this is your own application or an authorized engagement, I "
           "can explain where an XSS bug would be, show a proof-of-concept against "
           "your own test instance, and walk through the fix. Point me at the code "
           "and I'll analyze it.")
    return ask, ans


def fam_severity(rng):
    ask = rng.choice(["When is XSS high severity vs low?",
                      "How do I rate the severity of an XSS bug?",
                      "What makes one XSS worse than another?"])
    ans = (_open(rng) + "severity tracks impact and reach. Stored XSS on an "
           "authenticated, high-value page tends to be high or critical — it hits "
           "every viewer and can ride their session. Reflected or DOM XSS that needs "
           "a crafted link and a click is usually medium. It drops toward low when "
           "the context is constrained, the data isn't truly attacker-controlled, or "
           "a strict CSP blocks execution. Session theft, account takeover, or "
           "admin-panel reach pushes it up.")
    return ask, ans


def fam_sink_vs_safe(rng):
    pairs = [("innerHTML", "textContent"), ("v-html", "{{ }} interpolation"),
             ("th:utext", "th:text"), ("dangerouslySetInnerHTML", "a normal {child}"),
             ("document.write", "createElement + textContent"),
             ("the |safe filter", "plain autoescaped output"),
             ("EJS <%- %>", "EJS <%= %>"), ("Blade {!! !!}", "Blade {{ }}")]
    bad, good = rng.choice(pairs)
    ask = rng.choice([f"What's the difference between {bad} and {good}?",
                      f"Why pick {good} over {bad}?",
                      f"Is {good} really safer than {bad}?"])
    ans = (_open(rng) + f"{good} outputs the value as text with escaping applied, so "
           f"markup in it is shown literally. {bad} outputs it as raw HTML, so markup "
           f"is parsed and any script in it runs. For anything a user can influence, "
           f"use {good}; reserve {bad} for trusted content you control, and even then "
           f"prefer sanitising first.")
    return ask, ans


def fam_sanitizer(rng):
    s = rng.choice(list(_SANITIZERS))
    eco, how = _SANITIZERS[s]
    ask = rng.choice([f"How do I use {s} to prevent XSS?",
                      f"When should I reach for {s}?",
                      f"Is {s} enough to make user HTML safe?"])
    ans = (_open(rng) + f"{s} is a sanitiser for {eco}: {how}. Use it only when you "
           f"genuinely need to render user-supplied HTML — if you just need text, "
           f"plain escaping (or textContent) is simpler and safer. Run the sanitiser "
           f"as the last step before output, and don't concatenate anything onto its "
           f"result afterwards.")
    return ask, ans


def fam_csp(rng):
    ask = rng.choice(["What does a Content-Security-Policy do for XSS?",
                      "Does CSP stop XSS?", "Is CSP a replacement for escaping?",
                      "How does CSP help against XSS?"])
    ans = (_open(rng) + "a Content-Security-Policy restricts where scripts may load "
           "and run — for example, blocking inline <script> and eval, and limiting "
           "sources to your own origin. That turns many XSS bugs from code execution "
           "into a blocked attempt, so it's valuable defense in depth. But it's not a "
           "replacement for encoding: a permissive policy (unsafe-inline) gives little "
           "protection, and CSP doesn't fix the underlying injection. Encode for the "
           "context first, then add CSP on top.")
    return ask, ans


def fam_type_distinguish(rng):
    a, b = rng.choice([("reflected", "stored"), ("stored", "DOM-based"),
                       ("reflected", "DOM-based"), ("DOM-based", "mutation")])
    ask = rng.choice([f"What's the difference between {a} and {b} XSS?",
                      f"How do I tell {a} XSS from {b} XSS?",
                      f"{a} vs {b} XSS — what's the distinction?"])
    ans = (_open(rng) + f"{a} XSS is where {_TYPES[a]}. {b} XSS is where {_TYPES[b]}. "
           f"The data-flow is what separates them, not the payload — the same string "
           f"could appear in either, so you classify by how the input reaches the "
           f"sink.")
    return ask, ans


def fam_false_positive(rng):
    case, verdict = rng.choice([
        ("innerHTML is assigned a hard-coded string literal",
         "no — the value isn't attacker-controlled, so there's no XSS; flagging it would be a false positive"),
        ("a value goes through DOMPurify.sanitize before innerHTML",
         "no — the sanitiser is the last step, so it's safe as written"),
        ("output is inside Django {{ }} with autoescaping on and no |safe",
         "no — autoescaping neutralises it; this is a safe pattern, not a bug"),
        ("a number parsed with parseInt is written to innerHTML",
         "no — a parsed integer can't carry markup, so it's safe"),
        ("user text is shown with textContent",
         "no — textContent never parses HTML, so there's nothing to exploit"),
    ])
    ask = rng.choice([f"Is it XSS if {case}?",
                      f"My scanner flagged a line where {case}. Real bug?",
                      f"Should I report XSS when {case}?"])
    ans = (_open(rng) + f"{verdict}. The lesson is to trace the data: a dangerous-"
           f"looking sink is only a vulnerability when attacker-controlled input "
           f"actually reaches it without adequate neutralisation for that context.")
    return ask, ans


def fam_encoding_choice(rng):
    ctx = rng.choice(list(_ENCODER))
    ask = rng.choice([f"Which encoder should I use for {ctx}?",
                      f"How do I safely output a value into {ctx}?",
                      f"What's the right escaping for {ctx}?"])
    ans = (_open(rng) + f"for {ctx}, use {_ENCODER[ctx]}. The general rule is that "
           f"encoding must match the output context — the right choice for HTML body "
           f"is the wrong choice inside a script or a URL, and mixing them up leaves "
           f"the bug open.")
    return ask, ans


def fam_framework_autoescape(rng):
    fw = rng.choice(list(_AUTOESCAPE))
    does, nuance = _AUTOESCAPE[fw]
    ask = rng.choice([f"Does {fw} autoescape output?",
                      f"Is {fw} safe from XSS by default?",
                      f"Do I need to escape manually in {fw}?"])
    ans = (_open(rng) + f"{fw} autoescapes ({does}): {nuance}. So the practical "
           f"advice is to stay on the escaped path and treat the opt-out as a red "
           f"flag that needs a sanitiser or a very good reason.")
    return ask, ans


def fam_fix_critique(rng):
    bad_fix, why_insufficient, better = rng.choice([
        ("I strip the word 'script' from the input",
         "blocklists are trivially bypassed (e.g. <img onerror>, <svg onload>, or ScRiPt)",
         "encode for the output context instead of filtering keywords"),
        ("I remove < and > from the input",
         "that misses attribute, URL and JavaScript contexts where those characters aren't needed",
         "use context-appropriate encoding at output time"),
        ("I escape the value in HTML and then put it in an onclick",
         "HTML escaping doesn't neutralise a JavaScript string, so it still breaks out",
         "bind the handler in JS and keep user data out of the attribute"),
        ("I validate the input is a URL before using it in href",
         "a javascript: URL is still a valid URL, so it passes",
         "allow only the http and https schemes explicitly"),
    ])
    ask = f"To fix my XSS, {bad_fix}. Is that enough?"
    ans = (_open(rng) + f"not quite — {why_insufficient}. A more reliable fix is to "
           f"{better}. Input filtering can be a secondary layer, but the dependable "
           f"defense is correct output encoding (or sanitisation) for the exact "
           f"context the value lands in.")
    return ask, ans


# ---------------------------------------------------------------------------
# Multi-turn families: (rng) -> (turns, final_target)
# turns alternate user/assistant and EXCLUDE the final assistant answer.
# ---------------------------------------------------------------------------
def fam_followup(rng):
    s, fix = rng.choice([("innerHTML", "textContent"), ("v-html", "{{ }} interpolation"),
                         ("th:utext", "th:text"), ("document.write", "createElement with textContent"),
                         ("dangerouslySetInnerHTML", "a normal {child}")])
    kind = rng.choice([("a user's comment", "stored"), ("a URL parameter", "reflected or DOM-based"),
                       ("a chat message", "stored")])
    src, xtype = kind
    u1 = f"I'm using {s} to show {src}. Is that a problem?"
    a1 = (f"Yes — {s} renders it as HTML, so a value containing "
          f"<img src=x onerror=alert(1)> would run script in the viewer's browser. "
          f"That's {xtype} XSS.")
    u2 = rng.choice(["How do I fix it?", "What should I use instead?", "So what's the fix?"])
    a2 = (f"Switch to {fix}, which outputs the value as text instead of HTML. If it "
          f"genuinely needs formatting, run it through a sanitiser (DOMPurify on the "
          f"client, bleach or Jsoup on the server) and allow only a small tag "
          f"whitelist.")
    return [u1, a1, u2], a2


def fam_remediation_followup(rng):
    ctx, payload, fix = rng.choice([
        ("an href", "javascript:alert(1)", "allow only http/https schemes after parsing the URL"),
        ("a JavaScript string", "';alert(1);//", "use a JS-string encoder such as escapejs"),
        ("an onclick handler", "');alert(1);//", "bind the listener in JavaScript instead"),
    ])
    u1 = f"I HTML-escaped the user value that goes into {ctx}. Am I done?"
    a1 = (f"Not yet — HTML escaping protects the HTML body and quoted attributes, but "
          f"{ctx} is a different context. A value like `{payload}` has no HTML "
          f"metacharacters, so it slips through unchanged.")
    u2 = rng.choice(["So what's the correct fix?", "What should I do instead?",
                     "How do I actually close it?"])
    a2 = (f"For {ctx}, {fix}. Match the encoding (or the control) to the context the "
          f"value lands in; a Content-Security-Policy on top limits the damage if "
          f"something slips through.")
    return [u1, a1, u2], a2


SINGLE = [fam_what_is_type, fam_why_sink_dangerous, fam_payload_for_context,
          fam_is_pattern_safe, fam_fix_recommendation, fam_escaping_mismatch,
          fam_source_vs_sink, fam_refuse_misuse, fam_severity, fam_sink_vs_safe,
          fam_sanitizer, fam_csp, fam_type_distinguish, fam_false_positive,
          fam_encoding_choice, fam_framework_autoescape, fam_fix_critique]
MULTI = [fam_followup, fam_remediation_followup]
FAMILIES = SINGLE + MULTI


def to_record(family, rng) -> dict:
    if family in MULTI:
        turns, target = family(rng)
        msgs = [{"role": "system", "content": CHAT_SYSTEM}]
        for i, t in enumerate(turns):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
        messages = msgs
    else:
        user, target = family(rng)
        messages = [{"role": "system", "content": CHAT_SYSTEM},
                    {"role": "user", "content": user}]
    return {"messages": messages, "target": target, "template": family.__name__,
            "label": "chat", "language": "", "framework": ""}


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        rec = to_record(FAMILIES[attempts % len(FAMILIES)], rng)
        key = (rec["template"], rec["messages"][-1]["content"], rec["target"])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="data_chat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    records = generate(args.n, args.seed)
    train, val, test = split(records, args.val_frac, args.test_frac, args.seed)
    out = Path(args.out)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    print(f"generated {len(records)} unique chat examples from {len(FAMILIES)} families")
    for nm, recs in [("train", train), ("val", val), ("test", test)]:
        print(f"  {nm:5} {len(recs):5}")
    if len(records) < args.n:
        print(f"note: asked for {args.n} but only {len(records)} unique combinations "
              f"exist; enlarge the slot vocabularies or add families for more.")
    print(f"written to {out.resolve()}")
    print("\nNext: mix with scan data, keeping both:\n"
          "  cat data/train.jsonl data_chat/train.jsonl | shuf > data_mixed/train.jsonl\n"
          "  cat data/val.jsonl   data_chat/val.jsonl   | shuf > data_mixed/val.jsonl")


if __name__ == "__main__":
    main()
