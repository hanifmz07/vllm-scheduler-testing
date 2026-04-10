"""Custom scheduler queue variants based on prompt-length ordering.

This module defines a heap-backed request queue that orders waiting requests by
prompt length and two AsyncScheduler subclasses that install that queue for both
`waiting` and `skipped_waiting` flows.
"""

import heapq
from collections.abc import Iterable, Iterator

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request


class PromptLengthRequestQueue(RequestQueue):
    """A heap-backed queue that orders requests by prompt length.

    Ordering is deterministic and uses `(prompt_length, arrival_time,
    request_id)` as the sort key. For longest-first mode, prompt length is
    negated so larger prompts are popped first.

    The queue uses lazy deletion: removed requests are dropped from the active
    set immediately and later discarded from the heap when encountered.
    """

    def __init__(self, longest_first: bool) -> None:
        """Initialize the queue.

        Args:
            longest_first: If True, pop longest prompts first. If False, pop
                shortest prompts first.
        """
        self._heap: list[tuple[tuple[int, float, str], Request]] = []
        self._active: set[Request] = set()
        self._longest_first = longest_first

    def _key(self, request: Request) -> tuple[int, float, str]:
        """Return the deterministic heap key for a request."""
        # Tie-break by arrival time and request id for deterministic behavior.
        length_key = -request.num_prompt_tokens if self._longest_first else request.num_prompt_tokens
        return (length_key, request.arrival_time, request.request_id)

    def _trim_stale(self) -> None:
        """Discard stale heap entries that are no longer active."""
        while self._heap and self._heap[0][1] not in self._active:
            heapq.heappop(self._heap)

    def add_request(self, request: Request) -> None:
        """Insert a request into the queue according to prompt-length policy."""
        self._active.add(request)
        heapq.heappush(self._heap, (self._key(request), request))

    def pop_request(self) -> Request:
        """Remove and return the next request by queue ordering."""
        self._trim_stale()
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, request = heapq.heappop(self._heap)
        self._active.remove(request)
        return request

    def peek_request(self) -> Request:
        """Return the next request without removing it."""
        self._trim_stale()
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0][1]

    def prepend_request(self, request: Request) -> None:
        """Add one request using the same ordering semantics as add_request."""
        # Prepend follows policy ordering semantics for heap-backed queue.
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add multiple requests using this queue's ordering policy."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Mark a request as removed.

        The heap entry is removed lazily by `_trim_stale`.
        """
        self._active.discard(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Mark multiple requests as removed using lazy deletion."""
        for request in requests:
            self._active.discard(request)

    def __bool__(self) -> bool:
        """Return True when at least one active request exists."""
        return bool(self._active)

    def __len__(self) -> int:
        """Return the number of active requests."""
        return len(self._active)

    def __iter__(self) -> Iterator[Request]:
        """Iterate active requests in queue order."""
        return iter(sorted(self._active, key=self._key))


class LongestPromptFirstScheduler(AsyncScheduler):
    """AsyncScheduler variant with longest-prompt-first waiting queues.

    This preserves AsyncScheduler runtime behavior and only swaps queue policy
    for `waiting` and `skipped_waiting`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize scheduler and install longest-first prompt-length queues."""
        super().__init__(*args, **kwargs)
        self.waiting = PromptLengthRequestQueue(longest_first=True)
        self.skipped_waiting = PromptLengthRequestQueue(longest_first=True)


class ShortestPromptFirstScheduler(AsyncScheduler):
    """AsyncScheduler variant with shortest-prompt-first waiting queues.

    This preserves AsyncScheduler runtime behavior and only swaps queue policy
    for `waiting` and `skipped_waiting`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize scheduler and install shortest-first prompt-length queues."""
        super().__init__(*args, **kwargs)
        self.waiting = PromptLengthRequestQueue(longest_first=False)
        self.skipped_waiting = PromptLengthRequestQueue(longest_first=False)
