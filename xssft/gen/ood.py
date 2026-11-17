"""Hand-written out-of-distribution (OOD) test set.

Every example here uses a sink, source or trap that none of the 20 generator
templates in vue.py / django.py / spring.py produce (document.write,
insertAdjacentHTML, setTimeout strings, iframe src, unquoted attributes,
escaped-but-wrong-context event handlers, Thymeleaf [( )] inlining, FreeMarker
?no_esc, JSTL escapeXml="false", sanitiser-then-raw, text/plain responses, ...).
It measures whether the fine-tuned model learned XSS reasoning or just the
templates. Never train on it.

Usage:
    python -m xssft.gen.ood --out data_ood
    python -m xssft.eval --data data_ood --adapter runs/xss-qlora-1.5b --dump runs/ood_finetuned_1.5b_preds.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .. import schema
from .builder import example, finding, fix, poc, safe_report, sink, src, step, vuln_report
from .generate import to_record, write_jsonl

STACK = {"vue": ("JavaScript", "Vue 3"), "django": ("Python", "Django"),
         "spring": ("Java", "Spring Boot")}

# Each spec: id, stack, label, files {path: code}, sources [(path, snippet, kind)],
# sinks [(path, snippet, kind)], and for vuln: type, context, why.
# A snippet must occur on exactly one line of its file; that fixes the gold line.
SPECS: list[dict] = []


def add(**spec) -> None:
    SPECS.append(spec)


# =============================================================================
# Vue 3: vulnerable
# =============================================================================
add(id="ood_vue_document_write", stack="vue", label="vuln", type="dom", context="html_body",
    why="location.hash is decoded and passed to document.write, which parses it as HTML.",
    files={"src/components/PromoBanner.vue": """<template>
  <div class="banner"></div>
</template>

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const promo = decodeURIComponent(location.hash.slice(1))
  document.write('<h2>' + promo + '</h2>')
})
</script>"""},
    sources=[("src/components/PromoBanner.vue", "location.hash", "dom_location")],
    sinks=[("src/components/PromoBanner.vue", "document.write('<h2>' + promo + '</h2>')", "document_write")])

add(id="ood_vue_insert_adjacent_api", stack="vue", label="vuln", type="stored", context="html_body",
    why="Comment fields from the API are interpolated into a template literal and inserted with insertAdjacentHTML.",
    files={"src/components/CommentList.vue": """<template>
  <ul ref="list" class="comments"></ul>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const list = ref(null)

onMounted(async () => {
  const { data } = await axios.get('/api/comments')
  for (const c of data) {
    list.value.insertAdjacentHTML('beforeend', `<li><b>${c.author}</b>: ${c.text}</li>`)
  }
})
</script>"""},
    sources=[("src/components/CommentList.vue", "axios.get('/api/comments')", "api_response")],
    sinks=[("src/components/CommentList.vue", "list.value.insertAdjacentHTML", "insertAdjacentHTML")])

add(id="ood_vue_settimeout_string", stack="vue", label="vuln", type="dom", context="js_code",
    why="A query parameter is concatenated into a string passed to setTimeout, which evaluates it as code.",
    files={"src/components/Widget.vue": """<template>
  <div id="widget">Loading...</div>
</template>

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  const cb = new URLSearchParams(location.search).get('callback') || 'init'
  setTimeout(cb + '()', 100)
})
</script>"""},
    sources=[("src/components/Widget.vue", "new URLSearchParams(location.search).get('callback')", "dom_location")],
    sinks=[("src/components/Widget.vue", "setTimeout(cb + '()', 100)", "settimeout_string")])

add(id="ood_vue_iframe_src", stack="vue", label="vuln", type="reflected", context="url",
    why="A route query value is bound to iframe src with no scheme check, so javascript: URLs execute.",
    files={"src/views/PreviewView.vue": """<template>
  <div class="preview">
    <iframe :src="embedUrl" width="600" height="400"></iframe>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const embedUrl = computed(() => route.query.embed)
</script>"""},
    sources=[("src/views/PreviewView.vue", "route.query.embed", "url_query")],
    sinks=[("src/views/PreviewView.vue", ':src="embedUrl"', "src_binding")])

add(id="ood_vue_location_assign", stack="vue", label="vuln", type="dom", context="url",
    why="The next parameter is assigned to window.location.href; a javascript: value runs script.",
    files={"src/views/OnboardingView.vue": """<template>
  <button @click="finish">Continue</button>
</template>

<script setup>
const params = new URLSearchParams(window.location.search)
const next = params.get('next') || '/'

function finish() {
  window.location.href = next
}
</script>"""},
    sources=[("src/views/OnboardingView.vue", "params.get('next')", "dom_location")],
    sinks=[("src/views/OnboardingView.vue", "window.location.href = next", "location_assign")])

add(id="ood_vue_jquery_html", stack="vue", label="vuln", type="dom", context="html_body",
    why="A query parameter is concatenated into jQuery .html(), which parses it as HTML.",
    files={"src/components/Greeting.vue": """<template>
  <div id="greeting"></div>
</template>

<script setup>
import { onMounted } from 'vue'
import $ from 'jquery'

onMounted(() => {
  const who = new URLSearchParams(location.search).get('who')
  $('#greeting').html('Welcome back, ' + who)
})
</script>"""},
    sources=[("src/components/Greeting.vue", "new URLSearchParams(location.search).get('who')", "dom_location")],
    sinks=[("src/components/Greeting.vue", "$('#greeting').html('Welcome back, ' + who)", "jquery_html")])

# =============================================================================
# Vue 3: safe
# =============================================================================
add(id="ood_vue_safe_innertext", stack="vue", label="safe",
    files={"src/components/TabTitle.vue": """<template>
  <h2 ref="title"></h2>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const title = ref(null)

onMounted(() => {
  const tab = decodeURIComponent(location.hash.slice(1))
  title.value.innerText = tab
})
</script>"""},
    sources=[("src/components/TabTitle.vue", "location.hash", "dom_location")],
    sinks=[("src/components/TabTitle.vue", "title.value.innerText = tab", "innerText")],
    why="innerText assigns text; the browser never parses it as HTML.")

add(id="ood_vue_safe_href_validated", stack="vue", label="safe",
    files={"src/views/LinkView.vue": """<template>
  <a :href="safeLink" rel="noopener">Open link</a>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const safeLink = computed(() => {
  try {
    const u = new URL(route.query.url, window.location.origin)
    return ['http:', 'https:'].includes(u.protocol) ? u.href : '#'
  } catch {
    return '#'
  }
})
</script>"""},
    sources=[("src/views/LinkView.vue", "route.query.url", "url_query")],
    sinks=[("src/views/LinkView.vue", ':href="safeLink"', "href_binding")],
    why="The URL is parsed and only http/https schemes are allowed before binding to href.")

add(id="ood_vue_safe_vhtml_constant", stack="vue", label="safe",
    files={"src/components/SiteFooter.vue": """<template>
  <footer v-html="legalNotice"></footer>
</template>

<script setup>
const legalNotice = '<small>&copy; 2024 Example Corp. <a href="/terms">Terms</a></small>'
</script>"""},
    sources=[],
    sinks=[("src/components/SiteFooter.vue", 'v-html="legalNotice"', "vue_v_html")],
    why="v-html only renders a hard-coded constant; nothing attacker-controlled reaches it.")

add(id="ood_vue_safe_textnode", stack="vue", label="safe",
    files={"src/components/MessageList.vue": """<template>
  <ul ref="list"></ul>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const list = ref(null)

onMounted(async () => {
  const res = await fetch('/api/messages')
  const msgs = await res.json()
  for (const m of msgs) {
    const li = document.createElement('li')
    li.appendChild(document.createTextNode(m.body))
    list.value.appendChild(li)
  }
})
</script>"""},
    sources=[("src/components/MessageList.vue", "fetch('/api/messages')", "api_response")],
    sinks=[("src/components/MessageList.vue", "document.createTextNode(m.body)", "text_node")],
    why="createTextNode inserts the stored value as text, never as markup.")

add(id="ood_vue_safe_title_binding", stack="vue", label="safe",
    files={"src/views/HelpView.vue": """<template>
  <span :title="hint" class="tooltip">?</span>
</template>

<script setup>
import { ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const hint = ref(route.query.hint || '')
</script>"""},
    sources=[("src/views/HelpView.vue", "route.query.hint", "url_query")],
    sinks=[("src/views/HelpView.vue", ':title="hint"', "attribute_binding")],
    why="Vue sets title via setAttribute; a plain text attribute cannot execute script.")

add(id="ood_vue_safe_settimeout_fn", stack="vue", label="safe",
    files={"src/components/Loader.vue": """<template>
  <div v-if="ready">Ready</div>
</template>

<script setup>
import { ref } from 'vue'

const ready = ref(false)
const delay = Number(new URLSearchParams(location.search).get('delay')) || 500
setTimeout(() => { ready.value = true }, delay)
</script>"""},
    sources=[("src/components/Loader.vue", "new URLSearchParams(location.search).get('delay')", "dom_location")],
    sinks=[("src/components/Loader.vue", "setTimeout(() => { ready.value = true }, delay)", "settimeout")],
    why="setTimeout receives a function; the user value is only a numeric delay.")

# =============================================================================
# Django: vulnerable
# =============================================================================
add(id="ood_dj_unquoted_attr", stack="django", label="vuln", type="reflected", context="html_attribute",
    why="Autoescaping does not escape spaces, so an unquoted attribute lets the value add new attributes such as onfocus.",
    files={"search/views.py": """from django.shortcuts import render


def search(request):
    q = request.GET.get('q', '')
    return render(request, 'search.html', {'q': q})""",
           "search/templates/search.html": """<form action="/search">
  <input name="q" value={{ q }}>
  <button type="submit">Search</button>
</form>"""},
    sources=[("search/views.py", "request.GET.get('q', '')", "http_query")],
    sinks=[("search/templates/search.html", "value={{ q }}", "template_unquoted_attr")])

add(id="ood_dj_onclick_js_string", stack="django", label="vuln", type="reflected", context="js_string",
    why="HTML escaping turns ' into &#x27;, which the browser decodes inside the onclick attribute before running the JS string.",
    files={"greet/views.py": """from django.shortcuts import render


def greet(request):
    name = request.GET.get('name', 'friend')
    return render(request, 'greet.html', {'name': name})""",
           "greet/templates/greet.html": """<p>Hello!</p>
<button onclick="greet('{{ name }}')">Say hi</button>
<script src="/static/greet.js"></script>"""},
    sources=[("greet/views.py", "request.GET.get('name', 'friend')", "http_query")],
    sinks=[("greet/templates/greet.html", "onclick=\"greet('{{ name }}')\"", "event_handler_attr")])

add(id="ood_dj_template_from_db", stack="django", label="vuln", type="stored", context="html_body",
    why="The stored post body is concatenated into the template source itself, so it is output raw.",
    files={"blog/views.py": """from django.http import HttpResponse
from django.template import Context, Template

from .models import Post


def show_post(request, pk):
    post = Post.objects.get(pk=pk)
    tpl = Template("<article><h1>{{ title }}</h1>" + post.body + "</article>")
    return HttpResponse(tpl.render(Context({'title': post.title})))"""},
    sources=[("blog/views.py", "Post.objects.get(pk=pk)", "database")],
    sinks=[("blog/views.py", 'Template("<article><h1>{{ title }}</h1>" + post.body + "</article>")', "template_source_concat")])

add(id="ood_dj_format_html_href", stack="django", label="vuln", type="reflected", context="url",
    why="format_html HTML-escapes the value but does not block a javascript: URL in href.",
    files={"redirects/views.py": """from django.http import HttpResponse
from django.utils.html import format_html


def go(request):
    target = request.GET.get('next', '/')
    link = format_html('<a href="{}">Continue</a>', target)
    return HttpResponse(link)"""},
    sources=[("redirects/views.py", "request.GET.get('next', '/')", "http_query")],
    sinks=[("redirects/views.py", "format_html('<a href=\"{}\">Continue</a>', target)", "format_html_href")])

add(id="ood_dj_format_html_fstring", stack="django", label="vuln", type="reflected", context="html_body",
    why="The f-string interpolates the value before format_html runs, so nothing is escaped.",
    files={"accounts/views.py": """from django.http import HttpResponse
from django.utils.html import format_html


def welcome(request):
    name = request.GET.get('name', 'guest')
    html = format_html(f'<h1>Welcome, {name}!</h1>')
    return HttpResponse(html)"""},
    sources=[("accounts/views.py", "request.GET.get('name', 'guest')", "http_query")],
    sinks=[("accounts/views.py", "format_html(f'<h1>Welcome, {name}!</h1>')", "format_html_fstring")])

add(id="ood_dj_jinja_no_autoescape", stack="django", label="vuln", type="stored", context="html_body",
    why="A jinja2 Environment has autoescape off by default, so the stored bio is rendered raw.",
    files={"profiles/views.py": """from django.http import HttpResponse
from jinja2 import Environment

from .models import Profile

env = Environment()


def profile(request, username):
    user = Profile.objects.get(username=username)
    tpl = env.from_string('<div class="bio">{{ bio }}</div>')
    return HttpResponse(tpl.render(bio=user.bio))"""},
    sources=[("profiles/views.py", "Profile.objects.get(username=username)", "database")],
    sinks=[("profiles/views.py", "env.from_string('<div class=\"bio\">{{ bio }}</div>')", "jinja_no_autoescape"),
           ("profiles/views.py", "tpl.render(bio=user.bio)", "jinja_no_autoescape")])

# =============================================================================
# Django: safe
# =============================================================================
add(id="ood_dj_safe_json_script", stack="django", label="safe",
    files={"dash/views.py": """from django.shortcuts import render


def dashboard(request):
    prefs = {'theme': request.GET.get('theme', 'light')}
    return render(request, 'dash.html', {'prefs': prefs})""",
           "dash/templates/dash.html": """{{ prefs|json_script:"prefs-data" }}
<script>
  const prefs = JSON.parse(document.getElementById('prefs-data').textContent);
  document.body.dataset.theme = prefs.theme;
</script>"""},
    sources=[("dash/views.py", "request.GET.get('theme', 'light')", "http_query")],
    sinks=[("dash/templates/dash.html", '{{ prefs|json_script:"prefs-data" }}', "json_script")],
    why="json_script escapes <, > and & so the data cannot break out of its script tag.")

add(id="ood_dj_safe_format_html", stack="django", label="safe",
    files={"accounts/views.py": """from django.http import HttpResponse
from django.utils.html import format_html


def welcome(request):
    name = request.GET.get('name', 'guest')
    html = format_html('<h1>Welcome, {}!</h1>', name)
    return HttpResponse(html)"""},
    sources=[("accounts/views.py", "request.GET.get('name', 'guest')", "http_query")],
    sinks=[("accounts/views.py", "format_html('<h1>Welcome, {}!</h1>', name)", "format_html")],
    why="format_html escapes each argument placed in HTML body text.")

add(id="ood_dj_safe_quoted_attr", stack="django", label="safe",
    files={"search/views.py": """from django.shortcuts import render


def search(request):
    q = request.GET.get('q', '')
    return render(request, 'search.html', {'q': q})""",
           "search/templates/search.html": """<form action="/search">
  <input name="q" value="{{ q }}">
  <button type="submit">Search</button>
</form>"""},
    sources=[("search/views.py", "request.GET.get('q', '')", "http_query")],
    sinks=[("search/templates/search.html", 'value="{{ q }}"', "template_quoted_attr")],
    why="Autoescaping encodes quotes, so a quoted attribute value cannot be escaped from.")

add(id="ood_dj_safe_bleach", stack="django", label="safe",
    files={"forum/views.py": """import bleach
from django.shortcuts import get_object_or_404, render
from django.utils.safestring import mark_safe

from .models import Comment

ALLOWED_TAGS = ['b', 'i', 'em', 'strong', 'a', 'p']


def comment(request, pk):
    c = get_object_or_404(Comment, pk=pk)
    clean = bleach.clean(c.body, tags=ALLOWED_TAGS, attributes={'a': ['href']},
                         protocols=['http', 'https'], strip=True)
    return render(request, 'comment.html', {'body': mark_safe(clean)})"""},
    sources=[("forum/views.py", "get_object_or_404(Comment, pk=pk)", "database")],
    sinks=[("forum/views.py", "mark_safe(clean)", "mark_safe")],
    why="bleach.clean with a tag/protocol allow-list runs as the last step before mark_safe.")

add(id="ood_dj_safe_mark_safe_constant", stack="django", label="safe",
    files={"home/views.py": """from django.shortcuts import render
from django.utils.safestring import mark_safe

BANNER = '<strong>Scheduled maintenance on Sunday 02:00 UTC.</strong>'


def home(request):
    ctx = {'banner': mark_safe(BANNER), 'user_name': request.user.get_full_name()}
    return render(request, 'home.html', ctx)""",
           "home/templates/home.html": """<div class="alert">{{ banner }}</div>
<p>Hi {{ user_name }}</p>"""},
    sources=[],
    sinks=[("home/views.py", "mark_safe(BANNER)", "mark_safe")],
    why="mark_safe wraps a hard-coded constant; the user's name goes through autoescaping.")

add(id="ood_dj_safe_escapejs", stack="django", label="safe",
    files={"search/views.py": """from django.shortcuts import render


def results(request):
    return render(request, 'results.html', {'q': request.GET.get('q', '')})""",
           "search/templates/results.html": """<h2>Results for <span id="term"></span></h2>
<script>
  var searchTerm = "{{ q|escapejs }}";
  document.getElementById('term').textContent = searchTerm;
</script>"""},
    sources=[("search/views.py", "request.GET.get('q', '')", "http_query")],
    sinks=[("search/templates/results.html", 'var searchTerm = "{{ q|escapejs }}";', "escapejs")],
    why="escapejs encodes quotes, backslashes and < > for a JS string, and the DOM write uses textContent.")

# =============================================================================
# Spring Boot: vulnerable
# =============================================================================
_SB_IMPORTS = """package com.app;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
"""

add(id="ood_sb_th_href", stack="spring", label="vuln", type="reflected", context="url",
    why="th:href escapes the value for HTML but does not reject a javascript: URL.",
    files={"src/main/java/com/app/LinkController.java": _SB_IMPORTS + """
@Controller
public class LinkController {

    @GetMapping("/out")
    public String out(@RequestParam("url") String url, Model model) {
        model.addAttribute("target", url);
        return "out";
    }
}""",
           "src/main/resources/templates/out.html": """<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<body>
  <p>You are leaving the site.</p>
  <a th:href="${target}">Continue</a>
</body>
</html>"""},
    sources=[("src/main/java/com/app/LinkController.java", '@RequestParam("url") String url', "request_param")],
    sinks=[("src/main/resources/templates/out.html", 'th:href="${target}"', "thymeleaf_href")])

add(id="ood_sb_th_inline_unescaped", stack="spring", label="vuln", type="reflected", context="html_body",
    why="Thymeleaf [( )] inline expressions are unescaped output.",
    files={"src/main/java/com/app/ResultController.java": _SB_IMPORTS + """
@Controller
public class ResultController {

    @GetMapping("/result")
    public String result(@RequestParam String msg, Model model) {
        model.addAttribute("msg", msg);
        return "result";
    }
}""",
           "src/main/resources/templates/result.html": """<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<body>
  <p th:inline="text">Result: [(${msg})]</p>
</body>
</html>"""},
    sources=[("src/main/java/com/app/ResultController.java", "@RequestParam String msg", "request_param")],
    sinks=[("src/main/resources/templates/result.html", "[(${msg})]", "thymeleaf_inline_unescaped")])

add(id="ood_sb_servlet_writer", stack="spring", label="vuln", type="reflected", context="html_body",
    why="The parameter is concatenated into HTML written straight to the servlet response.",
    files={"src/main/java/com/app/GreetController.java": """package com.app;

import java.io.IOException;

import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class GreetController {

    @GetMapping("/greet")
    public void greet(@RequestParam String name, HttpServletResponse response) throws IOException {
        response.setContentType("text/html;charset=UTF-8");
        response.getWriter().write("<h1>Hello " + name + "</h1>");
    }
}"""},
    sources=[("src/main/java/com/app/GreetController.java", "@RequestParam String name", "request_param")],
    sinks=[("src/main/java/com/app/GreetController.java", 'response.getWriter().write("<h1>Hello " + name + "</h1>")', "servlet_writer")])

add(id="ood_sb_cout_escapexml_false", stack="spring", label="vuln", type="reflected", context="html_body",
    why='escapeXml="false" turns off c:out escaping, so the request parameter is output raw.',
    files={"src/main/webapp/WEB-INF/jsp/search.jsp": """<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<html>
<body>
  <h2>Results for <c:out value="${param.q}" escapeXml="false"/></h2>
</body>
</html>"""},
    sources=[("src/main/webapp/WEB-INF/jsp/search.jsp", "${param.q}", "request_param")],
    sinks=[("src/main/webapp/WEB-INF/jsp/search.jsp", '<c:out value="${param.q}" escapeXml="false"/>', "jstl_cout_unescaped")])

add(id="ood_sb_freemarker_no_esc", stack="spring", label="vuln", type="stored", context="html_body",
    why="?no_esc disables FreeMarker's auto-escaping for the stored review body.",
    files={"src/main/java/com/app/ReviewController.java": """package com.app;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

@Controller
public class ReviewController {

    private final ReviewRepository reviews;

    public ReviewController(ReviewRepository reviews) {
        this.reviews = reviews;
    }

    @GetMapping("/reviews/{id}")
    public String show(@PathVariable Long id, Model model) {
        model.addAttribute("review", reviews.findById(id).orElseThrow());
        return "review";
    }
}""",
           "src/main/resources/templates/review.ftlh": """<div class="review">
  <h3>${review.title}</h3>
  <div class="body">${review.body?no_esc}</div>
</div>"""},
    sources=[("src/main/java/com/app/ReviewController.java", "reviews.findById(id)", "database")],
    sinks=[("src/main/resources/templates/review.ftlh", "${review.body?no_esc}", "freemarker_no_esc")])

add(id="ood_sb_jsp_onclick_escapexml", stack="spring", label="vuln", type="reflected", context="js_string",
    why="fn:escapeXml makes ' into &#039;, which the browser decodes inside onclick before the JS string is parsed.",
    files={"src/main/webapp/WEB-INF/jsp/item.jsp": """<%@ taglib prefix="fn" uri="http://java.sun.com/jsp/jstl/functions" %>
<html>
<body>
  <button onclick="loadItem('${fn:escapeXml(param.id)}')">Load</button>
  <script src="/js/items.js"></script>
</body>
</html>"""},
    sources=[("src/main/webapp/WEB-INF/jsp/item.jsp", "param.id", "request_param")],
    sinks=[("src/main/webapp/WEB-INF/jsp/item.jsp", "onclick=\"loadItem('${fn:escapeXml(param.id)}')\"", "event_handler_attr")])

# =============================================================================
# Spring Boot: safe
# =============================================================================
add(id="ood_sb_safe_th_inline_escaped", stack="spring", label="safe",
    files={"src/main/java/com/app/ResultController.java": _SB_IMPORTS + """
@Controller
public class ResultController {

    @GetMapping("/result")
    public String result(@RequestParam String msg, Model model) {
        model.addAttribute("msg", msg);
        return "result";
    }
}""",
           "src/main/resources/templates/result.html": """<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<body>
  <p th:inline="text">Result: [[${msg}]]</p>
</body>
</html>"""},
    sources=[("src/main/java/com/app/ResultController.java", "@RequestParam String msg", "request_param")],
    sinks=[("src/main/resources/templates/result.html", "[[${msg}]]", "thymeleaf_inline")],
    why="[[ ]] inline expressions are HTML-escaped, like th:text.")

add(id="ood_sb_safe_cout", stack="spring", label="safe",
    files={"src/main/webapp/WEB-INF/jsp/search.jsp": """<%@ taglib prefix="c" uri="http://java.sun.com/jsp/jstl/core" %>
<html>
<body>
  <h2>Results for <c:out value="${param.q}"/></h2>
</body>
</html>"""},
    sources=[("src/main/webapp/WEB-INF/jsp/search.jsp", "${param.q}", "request_param")],
    sinks=[("src/main/webapp/WEB-INF/jsp/search.jsp", '<c:out value="${param.q}"/>', "jstl_cout")],
    why="c:out escapes XML/HTML special characters by default.")

add(id="ood_sb_safe_writer_escaped", stack="spring", label="safe",
    files={"src/main/java/com/app/GreetController.java": """package com.app;

import java.io.IOException;

import jakarta.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.util.HtmlUtils;

@Controller
public class GreetController {

    @GetMapping("/greet")
    public void greet(@RequestParam String name, HttpServletResponse response) throws IOException {
        response.setContentType("text/html;charset=UTF-8");
        response.getWriter().write("<h1>Hello " + HtmlUtils.htmlEscape(name) + "</h1>");
    }
}"""},
    sources=[("src/main/java/com/app/GreetController.java", "@RequestParam String name", "request_param")],
    sinks=[("src/main/java/com/app/GreetController.java", "response.getWriter().write(", "servlet_writer")],
    why="HtmlUtils.htmlEscape encodes the value for the HTML body context it is written into.")

add(id="ood_sb_safe_jsoup_utext", stack="spring", label="safe",
    files={"src/main/java/com/app/PostController.java": """package com.app;

import org.jsoup.Jsoup;
import org.jsoup.safety.Safelist;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;

@Controller
public class PostController {

    private final PostRepository posts;

    public PostController(PostRepository posts) {
        this.posts = posts;
    }

    @GetMapping("/posts/{id}")
    public String show(@PathVariable Long id, Model model) {
        Post post = posts.findById(id).orElseThrow();
        String clean = Jsoup.clean(post.getBody(), Safelist.basic());
        model.addAttribute("bodyHtml", clean);
        return "post";
    }
}""",
           "src/main/resources/templates/post.html": """<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<body>
  <article th:utext="${bodyHtml}"></article>
</body>
</html>"""},
    sources=[("src/main/java/com/app/PostController.java", "posts.findById(id)", "database")],
    sinks=[("src/main/resources/templates/post.html", 'th:utext="${bodyHtml}"', "thymeleaf_utext")],
    why="Jsoup.clean with Safelist.basic() runs as the last step before th:utext.")

add(id="ood_sb_safe_href_validated", stack="spring", label="safe",
    files={"src/main/java/com/app/LinkController.java": _SB_IMPORTS + """
import java.net.URI;

@Controller
public class LinkController {

    @GetMapping("/out")
    public String out(@RequestParam("url") String url, Model model) {
        String target = "/";
        try {
            String scheme = URI.create(url).getScheme();
            if ("https".equalsIgnoreCase(scheme) || "http".equalsIgnoreCase(scheme)) {
                target = url;
            }
        } catch (IllegalArgumentException ignored) {
        }
        model.addAttribute("target", target);
        return "out";
    }
}""",
           "src/main/resources/templates/out.html": """<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<body>
  <a th:href="${target}">Continue</a>
</body>
</html>"""},
    sources=[("src/main/java/com/app/LinkController.java", '@RequestParam("url") String url', "request_param")],
    sinks=[("src/main/resources/templates/out.html", 'th:href="${target}"', "thymeleaf_href")],
    why="Only http/https URLs reach th:href; anything else falls back to /.")

add(id="ood_sb_safe_text_plain", stack="spring", label="safe",
    files={"src/main/java/com/app/EchoController.java": """package com.app;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class EchoController {

    @GetMapping(value = "/echo", produces = MediaType.TEXT_PLAIN_VALUE)
    public String echo(@RequestParam String text) {
        return "You said: " + text;
    }
}"""},
    sources=[("src/main/java/com/app/EchoController.java", "@RequestParam String text", "request_param")],
    sinks=[("src/main/java/com/app/EchoController.java", 'return "You said: " + text;', "text_plain_response")],
    why="The response is text/plain, so the browser shows it as text and never parses HTML.")


# =============================================================================
def _line(files: dict, path: str, snippet: str) -> int:
    hits = [i for i, ln in enumerate(files[path].split("\n"), 1) if snippet in ln]
    if len(hits) != 1:
        raise AssertionError(f"snippet {snippet!r} matches {len(hits)} lines in {path}")
    return hits[0]


_FIX = {"html_body": "framework_safe_api", "html_attribute": "contextual_escaping",
        "js_string": "contextual_escaping", "js_code": "framework_safe_api",
        "url": "url_validation", "css": "contextual_escaping"}


def build(spec: dict) -> dict:
    files = spec["files"]
    srcs = [src(p, _line(files, p, s), s, k) for p, s, k in spec["sources"]]
    if spec["label"] == "vuln":
        sinks = [sink(p, _line(files, p, s), s, k, False, spec["why"]) for p, s, k in spec["sinks"]]
        path = ([step(x["file"], x["line"], x["code"], "attacker-controlled input") for x in srcs]
                + [step(x["file"], x["line"], x["code"], "unsafe sink") for x in sinks])
        report = vuln_report(srcs, sinks, [finding(
            "xss-1", spec["type"], spec["context"], spec["why"], path, "high",
            "Attacker-controlled script runs in the victim's session.", spec["why"],
            fix(_FIX[spec["context"]], "Neutralise the value for its output context.", "",
                "", "Content-Security-Policy"),
            poc("<img src=x onerror=alert(1)>", "", "script executes"))])
    else:
        sinks = [sink(p, _line(files, p, s), s, k, True, spec["why"]) for p, s, k in spec["sinks"]]
        report = safe_report(sinks, [{"file": x["file"], "line": x["line"], "pattern": x["kind"],
                                      "why": spec["why"]} for x in sinks], srcs)
    fl = [{"path": p, "content": c} for p, c in files.items()]
    problems = schema.validate_report(report, fl)
    if problems:
        raise AssertionError(f"{spec['id']}: {problems}")
    lang, fw = STACK[spec["stack"]]
    return example(fl, report, lang, fw, spec["id"], spec["label"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_ood")
    args = ap.parse_args()
    records = [to_record(build(s)) for s in SPECS]
    write_jsonl(Path(args.out) / "test.jsonl", records)
    n_v = sum(r["label"] == "vuln" for r in records)
    print(f"wrote {len(records)} OOD examples ({n_v} vuln, {len(records) - n_v} safe) "
          f"to {Path(args.out).resolve() / 'test.jsonl'}")


if __name__ == "__main__":
    main()
