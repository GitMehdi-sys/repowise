"""A queue named through the result of the call that declared it.

Go code binds and consumes ``q.Name`` after ``q, err := ch.QueueDeclare("orders",
...)``; amqplib ``q.queue`` after ``const q = await ch.assertQueue('orders')``;
pika ``result.method.queue`` after ``result = ch.queue_declare(queue='orders')``.
The queue is read from the file's declarations of that receiver when every one
names the same queue and the file settles it. A server-named queue (``""``),
a receiver declared with two names, or one the file also binds to anything
else, stays unread.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from ..strings import Arg, call_arguments, select_argument

if TYPE_CHECKING:
    from ..base import ScanContext
    from ..calls import FileStrings

_RECEIVER_RE = re.compile(r"[A-Za-z_$][\w$]*")
# Where every supported declaring call names its queue: `QueueDeclare("q")`,
# `assertQueue('q')`, `queue_declare(queue='q')`.
_DECLARED_NAME = Arg(keys=("queue",), pos=0)


@lru_cache(maxsize=256)
def _assignment_re(receiver: str, call: str) -> re.Pattern[str]:
    # Every assignment of *receiver*; ``decl`` is set when it is the declaring
    # call (`q, err := ch.QueueDeclare(`, `result = channel.queue_declare(`,
    # `const q = await ch.assertQueue(`).
    return re.compile(
        rf"(?<![\w$.]){re.escape(receiver)}\s*(?:,\s*[A-Za-z_]\w*)?\s*:?=(?![=>])\s*"
        rf"(?P<decl>(?:await\s+)?(?:[\w$]+\s*\.\s*)+{re.escape(call)}\s*\()?"
    )


@dataclass(frozen=True)
class QueueDeclaration:
    """*call* declares a queue, named at *name*; its result's *member* names it after."""

    call: str
    member: str
    name: Arg = _DECLARED_NAME

    def queues(
        self, ctx: ScanContext, args: list[str], arg: Arg, strings: FileStrings
    ) -> list[str]:
        """The queue the value *arg* selects (``q.Name``) names, or ``[]``."""
        raw = select_argument(args, arg)
        if len(raw) != 1:
            return []
        receiver, dot, member = raw[0].partition(".")
        if not dot or member != self.member or not _RECEIVER_RE.fullmatch(receiver):
            return []
        found: set[str] = set()
        for m in _assignment_re(receiver, self.call).finditer(ctx.content):
            if ctx.in_comment(m.start()):
                continue
            if m.group("decl") is None:
                return []  # the name is bound to something else too (file scope)
            declared = call_arguments(ctx.content, m.end() - 1) or []
            names, refused = strings.resolve(declared, self.name)
            if refused or not names or not all(names):
                return []  # unread, or server-named (`""`)
            found.update(names)
        return list(found) if len(found) == 1 else []


__all__ = ["QueueDeclaration"]
