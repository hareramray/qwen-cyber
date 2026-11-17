"""Vue 3 single-file-component templates (JavaScript / TypeScript).

Sinks covered: v-html, Element.innerHTML, :href="javascript:", render-function
domProps.innerHTML. Safe counterparts: {{ }} interpolation, textContent,
DOMPurify, validated URLs.
"""
from __future__ import annotations

import random

from .builder import (Code, example, finding, fix, names, poc, safe_report, sink,
                      src, step, vuln_report)

LANG = "JavaScript"
FW = "Vue 3"


def _comp_name(nm: dict) -> str:
    return nm["route"].capitalize() + "View"


# --- vulnerable: v-html from route query (reflected) ------------------------
def vue_vhtml_route(rng: random.Random) -> dict:
    nm = names(rng)
    comp = _comp_name(nm)
    path = f"src/views/{comp}.vue"
    c = Code()
    c.add("<template>", "  <section class=\"page\">", "    <h1>Profile</h1>")
    sink_line = c.add(f"    <div class=\"{nm['noun']}\" v-html=\"{nm['noun']}\"></div>")
    c.add("  </section>", "</template>", "", "<script setup>")
    c.add("import { ref } from 'vue'", "import { useRoute } from 'vue-router'", "")
    c.add("const route = useRoute()")
    src_line = c.add(f"const {nm['noun']} = ref(route.query.{nm['param']} || '')")
    c.add("</script>")
    files = [{"path": path, "content": c.text()}]
    report = vuln_report(
        sources=[src(path, src_line, f"route.query.{nm['param']}", "url_query")],
        sinks=[sink(path, sink_line, f"v-html=\"{nm['noun']}\"", "vue_v_html", False,
                    "v-html renders its value as raw HTML with no escaping")],
        findings=[finding(
            "xss-1", "reflected", "html_body",
            "Value from the URL query string is bound into the DOM through v-html.",
            [step(path, src_line, f"route.query.{nm['param']}", "attacker-controlled query parameter"),
             step(path, sink_line, f"v-html=\"{nm['noun']}\"", "rendered as raw HTML")],
            "high", "Reflected script execution in the victim's session from a crafted link.",
            "v-html bypasses Vue's template escaping, so a query parameter reflected through it "
            "executes attacker HTML/JS in the page.",
            fix("framework_safe_api",
                "Render the value as text, not HTML.",
                f"<div class=\"{nm['noun']}\">{{{{ {nm['noun']} }}}}</div>",
                "If HTML really is required, sanitize with DOMPurify before binding to v-html.",
                "Content-Security-Policy without unsafe-inline limits the impact."),
            poc("<img src=x onerror=alert(1)>",
                f"Open the page with ?{nm['param']}=<img src=x onerror=alert(1)>",
                "Injected markup runs script in the victim's browser"))])
    return example(files, report, LANG, FW, "vue_vhtml_route", "vuln")


# --- vulnerable: innerHTML in mounted (DOM) ---------------------------------
def vue_innerhtml_dom(rng: random.Random) -> dict:
    nm = names(rng)
    comp = _comp_name(nm)
    path = f"src/components/{comp}.vue"
    c = Code()
    c.add("<template>", f"  <div ref=\"box\" class=\"{nm['noun']}\"></div>", "</template>", "")
    c.add("<script setup>", "import { ref, onMounted } from 'vue'", "")
    c.add("const box = ref(null)")
    c.add("onMounted(() => {")
    src_line = c.add(f"  const {nm['var']} = new URLSearchParams(location.search).get('{nm['param']}')")
    sink_line = c.add(f"  box.value.innerHTML = {nm['var']}")
    c.add("})", "</script>")
    files = [{"path": path, "content": c.text()}]
    report = vuln_report(
        sources=[src(path, src_line, f"location.search", "dom_location")],
        sinks=[sink(path, sink_line, f"box.value.innerHTML = {nm['var']}", "innerHTML", False,
                    "Assigning to innerHTML parses the string as HTML")],
        findings=[finding(
            "xss-1", "dom", "html_body",
            "location.search is read in the browser and written to innerHTML.",
            [step(path, src_line, f"location.search", "attacker controls the query string"),
             step(path, sink_line, f"box.value.innerHTML = {nm['var']}", "raw HTML assignment")],
            "high", "DOM-based XSS executing entirely client-side.",
            "The value never reaches the server; the sink parses attacker HTML directly in the DOM.",
            fix("framework_safe_api",
                "Use textContent instead of innerHTML.",
                f"box.value.textContent = {nm['var']}",
                "Or sanitize with DOMPurify.sanitize() before assignment.",
                "A strict CSP reduces exploitability of DOM sinks."),
            poc("<img src=x onerror=alert(1)>",
                f"Navigate to the component URL with ?{nm['param']}=<img src=x onerror=alert(1)>",
                "Script runs when the element is populated"))])
    return example(files, report, LANG, FW, "vue_innerhtml_dom", "vuln")


# --- vulnerable: :href javascript: URL (url context) ------------------------
def vue_href_url(rng: random.Random) -> dict:
    nm = names(rng)
    comp = _comp_name(nm)
    path = f"src/views/{comp}.vue"
    c = Code()
    c.add("<template>", "  <nav>")
    sink_line = c.add(f"    <a :href=\"{nm['var']}\">visit</a>")
    c.add("  </nav>", "</template>", "", "<script setup>")
    c.add("import { ref } from 'vue'", "import { useRoute } from 'vue-router'")
    c.add("const route = useRoute()")
    src_line = c.add(f"const {nm['var']} = ref(route.query.{nm['param']})")
    c.add("</script>")
    files = [{"path": path, "content": c.text()}]
    report = vuln_report(
        sources=[src(path, src_line, f"route.query.{nm['param']}", "url_query")],
        sinks=[sink(path, sink_line, f":href=\"{nm['var']}\"", "href_binding", False,
                    "A bound href accepts the javascript: scheme")],
        findings=[finding(
            "xss-1", "reflected", "url",
            "User-supplied URL is bound to href without scheme validation.",
            [step(path, src_line, f"route.query.{nm['param']}", "attacker-controlled URL"),
             step(path, sink_line, f":href=\"{nm['var']}\"", "javascript: scheme executes on click")],
            "medium", "Requires a user click but runs arbitrary script.",
            "Vue escapes the attribute value for HTML but does not block the javascript: scheme, "
            "so a javascript: URL executes when the link is activated.",
            fix("url_validation",
                "Allow only http/https (or relative) URLs before binding.",
                f"const safe = /^https?:\\/\\//.test({nm['var']}.value) ? {nm['var']}.value : '#'",
                "Use the URL constructor and check protocol against an allow-list.",
                "Add rel=noopener and a CSP as defense in depth."),
            poc("javascript:alert(1)",
                f"Open the page with ?{nm['param']}=javascript:alert(1) and click the link",
                "Script runs on click"))])
    return example(files, report, LANG, FW, "vue_href_url", "vuln")


# --- vulnerable: render function domProps.innerHTML -------------------------
def vue_render_domprops(rng: random.Random) -> dict:
    nm = names(rng)
    comp = _comp_name(nm)
    path = f"src/components/{comp}.js"
    c = Code()
    c.add("import { h } from 'vue'", "")
    c.add("export default {", f"  props: ['{nm['noun']}'],", "  render() {")
    src_line = c.add(f"    const html = this.{nm['noun']}")
    sink_line = c.add(f"    return h('div', {{ domProps: {{ innerHTML: html }} }})")
    c.add("  }", "}")
    files = [{"path": path, "content": c.text()}]
    report = vuln_report(
        sources=[src(path, src_line, f"this.{nm['noun']}", "component_prop")],
        sinks=[sink(path, sink_line, "domProps: { innerHTML: html }", "innerHTML", False,
                    "domProps.innerHTML renders raw HTML like v-html")],
        findings=[finding(
            "xss-1", "stored", "html_body",
            "A prop value is rendered as raw HTML through domProps.innerHTML.",
            [step(path, src_line, f"this.{nm['noun']}", "prop forwarded from parent/API data"),
             step(path, sink_line, "innerHTML: html", "raw HTML rendering")],
            "high", "If the prop carries stored user content, this is persistent XSS.",
            "domProps.innerHTML is the render-function equivalent of v-html and performs no escaping.",
            fix("framework_safe_api",
                "Return a text vnode instead of innerHTML.",
                f"return h('div', this.{nm['noun']})",
                "Sanitize with DOMPurify if HTML output is a requirement.",
                "CSP as an additional layer."),
            poc("<img src=x onerror=alert(1)>",
                "Supply the payload as the component prop via stored data",
                "Stored markup executes for every viewer"))])
    return example(files, report, LANG, FW, "vue_render_domprops", "vuln")


# --- safe: interpolation ----------------------------------------------------
def vue_safe_interp(rng: random.Random) -> dict:
    nm = names(rng)
    comp = _comp_name(nm)
    path = f"src/views/{comp}.vue"
    c = Code()
    c.add("<template>", "  <section>")
    safe_line = c.add(f"    <p class=\"{nm['noun']}\">{{{{ {nm['noun']} }}}}</p>")
    c.add("  </section>", "</template>", "", "<script setup>")
    c.add("import { ref } from 'vue'", "import { useRoute } from 'vue-router'")
    c.add("const route = useRoute()")
    c.add(f"const {nm['noun']} = ref(route.query.{nm['param']} || '')")
    c.add("</script>")
    files = [{"path": path, "content": c.text()}]
    report = safe_report(
        sinks=[sink(path, safe_line, f"{{{{ {nm['noun']} }}}}", "vue_interpolation", True,
                    "Vue mustache interpolation HTML-escapes its value")],
        safe_patterns=[{"file": path, "line": safe_line, "pattern": f"{{{{ {nm['noun']} }}}}",
                        "why": "Text interpolation escapes <, >, & and quotes, so query input is inert"}])
    return example(files, report, LANG, FW, "vue_safe_interp", "safe")


# --- safe: textContent ------------------------------------------------------
def vue_safe_textcontent(rng: random.Random) -> dict:
    nm = names(rng)
    comp = _comp_name(nm)
    path = f"src/components/{comp}.vue"
    c = Code()
    c.add("<template>", f"  <div ref=\"box\"></div>", "</template>", "")
    c.add("<script setup>", "import { ref, onMounted } from 'vue'")
    c.add("const box = ref(null)")
    c.add("onMounted(() => {")
    c.add(f"  const {nm['var']} = new URLSearchParams(location.search).get('{nm['param']}')")
    safe_line = c.add(f"  box.value.textContent = {nm['var']}")
    c.add("})", "</script>")
    files = [{"path": path, "content": c.text()}]
    report = safe_report(
        sinks=[sink(path, safe_line, f"box.value.textContent = {nm['var']}", "textContent", True,
                    "textContent sets text, never parses HTML")],
        safe_patterns=[{"file": path, "line": safe_line, "pattern": "textContent",
                        "why": "textContent assigns a string as plain text, so no markup is parsed"}])
    return example(files, report, LANG, FW, "vue_safe_textcontent", "safe")


# --- safe: DOMPurify + v-html ----------------------------------------------
def vue_safe_dompurify(rng: random.Random) -> dict:
    nm = names(rng)
    comp = _comp_name(nm)
    path = f"src/views/{comp}.vue"
    c = Code()
    c.add("<template>")
    safe_line = c.add(f"  <article v-html=\"clean{nm['noun'].capitalize()}\"></article>")
    c.add("</template>", "", "<script setup>")
    c.add("import { computed } from 'vue'", "import DOMPurify from 'dompurify'")
    c.add(f"const props = defineProps(['{nm['noun']}'])")
    san_line = c.add(f"const clean{nm['noun'].capitalize()} = computed(() => DOMPurify.sanitize(props.{nm['noun']}))")
    c.add("</script>")
    files = [{"path": path, "content": c.text()}]
    report = safe_report(
        sinks=[sink(path, safe_line, f"v-html=\"clean{nm['noun'].capitalize()}\"", "vue_v_html", True,
                    "Value was sanitized by DOMPurify before binding")],
        safe_patterns=[{"file": path, "line": san_line, "pattern": "DOMPurify.sanitize",
                        "why": "DOMPurify strips scripts and event handlers before v-html renders the value"}])
    return example(files, report, LANG, FW, "vue_safe_dompurify", "safe")


ALL = [vue_vhtml_route, vue_innerhtml_dom, vue_href_url, vue_render_domprops,
       vue_safe_interp, vue_safe_textcontent, vue_safe_dompurify]
