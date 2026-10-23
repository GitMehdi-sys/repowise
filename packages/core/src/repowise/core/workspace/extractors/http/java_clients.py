"""Java HTTP consumer dialect.

Three call shapes, each with its own source for the verb:

* Feign declarative clients: ``@RequestLine("POST /v1/{entityType}/images")``.
  Verb and path both sit inside one annotation string and the placeholders are
  already ``{param}`` shaped, so this is the cleanest Java shape there is.
* ``java.net.http``: ``HttpRequest.newBuilder(URI.create(url))`` or
  ``HttpRequest.newBuilder().uri(URI.create(url))``, with the verb read from the
  builder terminal (``.GET()``, ``.POST(...)``, ``.method("PATCH", ...)``) found
  by walking the chain up to its ``;`` or its ``.build()``.
* Spring ``RestTemplate``: ``getForObject`` / ``postForEntity`` and friends name
  the verb in the callee, ``exchange(uri, HttpMethod.GET, ...)`` names it in the
  second argument.

Spring's declarative clients, Java or Kotlin, are read by their own dialect:
an ``@FeignClient`` interface written with the controller annotations, and a
Spring 6 ``@HttpExchange`` interface. Both are parsed by the Spring provider
reader in ``ingestion.framework_routes``, so a client and the controller it
calls read one grammar.

A chain with no terminal verb is refused rather than defaulted to GET, and a URL
that is a bare variable is refused unless the file assigns it a string literal
exactly once. OkHttp is left out: the corpus builds its ``Request.Builder`` from
a URL validated upstream, so no site carries a path this layer could read.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from repowise.core.ingestion.framework_routes import (
    SPRING_EXCHANGE,
    RouteMatch,
    micronaut_annotations,
    spring_class_paths,
    spring_feign_clients,
    spring_route_prefixes,
    spring_routes,
)
from repowise.core.workspace.origins import ContractOrigin

from ..langs import JAVA, KOTLIN
from ..strings import JAVA_SYNTAX, call_arguments, match_paren, string_constants
from .client_calls import VERBS, ClientCallMatch, consumer_contracts, method_from_argument
from .mounts import compose_prefix

if TYPE_CHECKING:
    from repowise.core.workspace.contracts import Contract

    from ..base import ScanContext

# @RequestLine("POST /v1/{entityType}/images"): verb then path, one literal.
_REQUEST_LINE_RE = re.compile(r"@RequestLine\s*\(\s*\"([A-Za-z]+)[ \t]+([^\"\n]*)\"")

# The head of a java.net.http builder chain.
_NEW_BUILDER_RE = re.compile(r"\bHttpRequest\s*\.\s*newBuilder\s*\(")

# One `.name(` step of a fluent chain.
_CHAIN_STEP_RE = re.compile(r"\.\s*([A-Za-z_]\w*)\s*\(")

# How far past the chain head a terminal may sit. A java.net.http request is
# built in a handful of lines; beyond that the scan is reading unrelated code.
_CHAIN_LIMIT = 2000

# RestTemplate calls. The verb-suffixed names are unmistakable; bare `put` and
# `delete` are not, which is why the file is gated on the class name and the
# rows go through `path_only`.
_REST_TEMPLATE_RE = re.compile(
    r"\b(?:[A-Za-z_]\w*\s*\.\s*"
    r"(getForObject|getForEntity|postForObject|postForEntity|patchForObject|exchange)"
    r"|\w*[Tt]emplate\w*\s*\.\s*(put|delete))"
    r"\s*\("
)



def feign_calls(content: str) -> Iterator[ClientCallMatch]:
    for m in _REQUEST_LINE_RE.finditer(content):
        verb = m.group(1)
        if verb.lower() not in VERBS:
            continue
        yield ClientCallMatch(
            client="feign",
            # The annotation holds the path unquoted; the resolver reads a
            # literal, so hand it one.
            url=f'"{m.group(2).strip()}"',
            offset=m.start(),
            method=verb.upper(),
            origin=ContractOrigin.CLIENT_CALL,
        )


def _chain_steps(content: str, start: int) -> Iterator[tuple[str, int]]:
    """``(step name, its '(' offset)`` for each ``.name(...)`` from *start*.

    Stops at the first character that is not part of the chain (the statement's
    ``;``, a comma, a closing paren) and after ``.build()``, so a verb belonging
    to a neighbouring call is never read as this request's terminal.
    """
    i = start
    limit = min(len(content), start + _CHAIN_LIMIT)
    while i < limit:
        if content[i].isspace():
            i += 1
            continue
        m = _CHAIN_STEP_RE.match(content, i)
        if m is None:
            return
        close = match_paren(content, m.end() - 1)
        if close < 0:
            return
        yield m.group(1), m.end() - 1
        if m.group(1) == "build":
            return
        i = close + 1


def java_net_http_calls(content: str) -> Iterator[ClientCallMatch]:
    for m in _NEW_BUILDER_RE.finditer(content):
        paren = m.end() - 1
        close = match_paren(content, paren)
        args = call_arguments(content, paren, close) if close >= 0 else None
        if args is None:
            continue
        # newBuilder(uri); the two-argument copy form carries a request, not a URL.
        url = args[0] if len(args) == 1 else None
        method: str | None = None
        for name, step_paren in _chain_steps(content, close + 1):
            if name == "uri" and url is None:
                inner = call_arguments(content, step_paren)
                if inner is not None and len(inner) == 1:
                    url = inner[0]
            elif name.isupper() and name.lower() in VERBS:
                method = name
            elif name == "method":
                inner = call_arguments(content, step_paren)
                if inner:
                    method = method_from_argument(inner[0])
        if url is None or method is None:
            continue  # no URL, or a verb only the caller knows
        yield ClientCallMatch(
            client="java-net-http",
            url=url,
            offset=m.start(),
            method=method,
            origin=ContractOrigin.CLIENT_CALL,
        )


def resttemplate_calls(content: str) -> Iterator[ClientCallMatch]:
    if "RestTemplate" not in content:
        return
    for m in _REST_TEMPLATE_RE.finditer(content):
        callee = m.group(1) or m.group(2)
        paren = m.end() - 1
        args = call_arguments(content, paren)
        if not args:
            continue
        method = None
        if callee == "exchange":
            if len(args) < 2:
                continue
            method = method_from_argument(args[1])
            if method is None:
                continue  # a verb held in a variable is not one we know
        yield ClientCallMatch(
            client="resttemplate",
            url=args[0],
            offset=m.start(),
            callee=callee,
            method=method,
            origin=ContractOrigin.NAME_HEURISTIC,
        )


class JavaClientsDialect:
    name = "java-clients"
    extensions = JAVA
    requires = ("@RequestLine", "HttpRequest", "RestTemplate")

    def extract(self, ctx: ScanContext) -> list[Contract]:
        content = ctx.content
        named = [*feign_calls(content), *java_net_http_calls(content)]
        # `.put`/`.delete` on any receiver could be a collection, so a URL with
        # no slash in it is a key rather than a route.
        ambiguous = list(resttemplate_calls(content))
        if not named and not ambiguous:
            return []
        constants = string_constants(content, JAVA_SYNTAX)
        return [
            *consumer_contracts(ctx, named, JAVA_SYNTAX, constants=constants),
            *consumer_contracts(ctx, ambiguous, JAVA_SYNTAX, constants=constants, path_only=True),
        ]


# A literal argument of a Spring annotation: `key = "value"`, or the positional
# first argument, which is `value`.
_ANNOTATION_STRING_RE = re.compile(
    r'(?:,\s*(?P<key>\w+)\s*=\s*|\(\s*(?:(?P<first>\w+)\s*=\s*)?)"(?P<value>[^"\\\n]*)"\s*(?=[,)])'
)

# A controller in the same file: `@HttpExchange` there declares a route served.
# The guard is per file; a shared interface a controller elsewhere implements
# still reads as a client.
_CONTROLLER_RE = re.compile(r"@(?:Rest)?Controller\b")

# A host the service-name map can look up: a bare service id, no placeholder.
_SERVICE_ID_RE = re.compile(r"[\w.-]+")


def _top_level(args: str) -> str:
    """*args* with every nested ``(...)``, ``{...}`` and ``[...]`` blanked, offsets kept.

    A nested annotation's ``path`` or ``url`` (``configuration = @Cfg(path = ...)``)
    is not the client's.
    """
    out = list(args)
    depth = 0
    quoted = False
    for i, ch in enumerate(args):
        if ch == '"' and (i == 0 or args[i - 1] != "\\"):
            quoted = not quoted
        elif not quoted and ch in "([{":
            depth += 1
        elif not quoted and ch in ")]}":
            depth -= 1
            if depth >= 1:
                out[i] = " "
            continue
        if depth > 1:
            out[i] = " "
    return "".join(out)


def _annotation_strings(args: str) -> dict[str, str]:
    """The literal arguments of an annotation's *args*, by key."""
    out: dict[str, str] = {}
    for m in _ANNOTATION_STRING_RE.finditer(args):
        out.setdefault(m.group("key") or m.group("first") or "value", m.group("value"))
    return out


def _names_key(args: str, key: str) -> bool:
    """Whether *args* passes *key* at all, literal or not."""
    return re.search(rf"[(,]\s*{key}\s*=", args) is not None


def _feign_base(args: str) -> tuple[str, str] | None:
    """``(base URL, path prefix)`` a ``@FeignClient`` declares, ``None`` when unreadable.

    ``url`` is where the client sends, literal or a ``${property}`` placeholder
    the base resolver reads like any other. With no ``url`` the client calls
    the service its ``name`` (or ``value``) registers under, which becomes the
    host so the service-name map can place it; a placeholder name, or a ``url``
    held in a constant, leaves the call a bare path. A ``path`` this cannot
    read leaves every route under the client unknown.
    """
    args = _top_level(args)
    strings = _annotation_strings(args)
    prefix = strings.get("path", "")
    if "${" in prefix or ("path" not in strings and _names_key(args, "path")):
        return None
    return _feign_host(args, strings), prefix


def _feign_host(args: str, strings: dict[str, str]) -> str:
    """The base URL a Feign client sends to: its ``url``, else its service name, else ``""``."""
    if "url" in strings:
        return strings["url"].rstrip("/")
    name = strings.get("name") or strings.get("value") or ""
    if _names_key(args, "url") or not _SERVICE_ID_RE.fullmatch(name):
        return ""
    return f"http://{name.lower()}"


def _composed_paths(
    class_paths: list[tuple[int, int, tuple[str, ...] | None]], route: RouteMatch, prefix: str
) -> Iterator[str]:
    """Each concrete path *route* reaches under its type prefixes and the client *prefix*."""
    for type_prefix in spring_route_prefixes(class_paths, route.offset) or ():
        path = compose_prefix(prefix, compose_prefix(type_prefix, route.path or ""))
        if path.strip("/"):
            yield path


def _interface_rows(
    routes: Iterator[RouteMatch],
    class_paths: list[tuple[int, int, tuple[str, ...] | None]],
    base_at: Callable[[int], tuple[str, str] | None],
    *,
    client: str,
    unrestricted: str | None,
) -> Iterator[ClientCallMatch]:
    """Rows for a client interface's *routes*, each composed onto its type prefix and base.

    *base_at* gives the ``(base URL, path prefix)`` of the client around an
    offset. *unrestricted* is the verb the client sends for a mapping that
    names none; ``None`` refuses such a mapping.
    """
    for route in routes:
        base = None if route.path is None else base_at(route.offset)
        verb = unrestricted if route.verb == "ALL" else route.verb
        if base is None or verb is None or verb.lower() not in VERBS:
            continue
        url_base, prefix = base
        for path in _composed_paths(class_paths, route, prefix):
            yield ClientCallMatch(
                client=client,
                # The resolver reads a literal, so hand it one.
                url=f'"{url_base}/{path.lstrip("/")}"',
                offset=route.offset,
                method=verb,
                origin=ContractOrigin.CLIENT_CALL,
            )


def feign_interface_calls(content: str, annotations: set[int]) -> Iterator[ClientCallMatch]:
    """Calls an ``@FeignClient`` interface declares with Spring MVC annotations."""
    clients = spring_feign_clients(content, annotations)
    if not clients:
        return

    def base_at(offset: int) -> tuple[str, str] | None:
        inner = [args for start, end, args in clients if start < offset < end]
        return _feign_base(inner[-1]) if inner else None

    yield from _interface_rows(
        spring_routes(content, annotations, feign=True),
        spring_class_paths(content, annotations),
        base_at,
        client="feign",
        unrestricted="GET",  # Feign sends a mapping that names no verb as a GET
    )


def http_exchange_calls(content: str, annotations: set[int]) -> Iterator[ClientCallMatch]:
    """Calls a Spring 6 ``@HttpExchange`` interface declares.

    The base URL is set where the proxy is built, out of this file's sight, so
    each row is a path. A file holding a controller is skipped: there the same
    annotations declare routes served, not calls.
    """
    if _CONTROLLER_RE.search(content):
        return
    yield from _interface_rows(
        spring_routes(content, annotations, SPRING_EXCHANGE),
        spring_class_paths(content, annotations, SPRING_EXCHANGE),
        lambda _offset: ("", ""),
        client="http-exchange",
        unrestricted=None,
    )


class SpringClientsDialect:
    name = "spring-clients"
    extensions = JAVA | KOTLIN
    requires = (
        "@FeignClient",
        "@HttpExchange",
        "@GetExchange",
        "@PostExchange",
        "@PutExchange",
        "@DeleteExchange",
        "@PatchExchange",
    )

    def extract(self, ctx: ScanContext) -> list[Contract]:
        content = ctx.content
        annotations = micronaut_annotations(content)
        rows = [
            *feign_interface_calls(content, annotations),
            *http_exchange_calls(content, annotations),
        ]
        if not rows:
            return []
        # The URL text is built here, not read from the file, so it resolves as
        # a plain literal whatever the file's language.
        return consumer_contracts(ctx, rows, JAVA_SYNTAX)
