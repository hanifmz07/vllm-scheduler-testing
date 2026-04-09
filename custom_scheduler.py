import heapq
from collections.abc import Iterable, Iterator

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


class PromptLengthRequestQueue(RequestQueue):
    """Queue that pops waiting requests by prompt length."""

    def __init__(self, longest_first: bool) -> None:
        self._heap: list[tuple[tuple[int, float, str], Request]] = []
        self._active: set[Request] = set()
        self._longest_first = longest_first

    def _key(self, request: Request) -> tuple[int, float, str]:
        # Tie-break by arrival time and request id for deterministic behavior.
        length_key = -request.num_prompt_tokens if self._longest_first else request.num_prompt_tokens
        return (length_key, request.arrival_time, request.request_id)

    def _trim_stale(self) -> None:
        while self._heap and self._heap[0][1] not in self._active:
            heapq.heappop(self._heap)

    def add_request(self, request: Request) -> None:
        self._active.add(request)
        heapq.heappush(self._heap, (self._key(request), request))

    def pop_request(self) -> Request:
        self._trim_stale()
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, request = heapq.heappop(self._heap)
        self._active.remove(request)
        return request

    def peek_request(self) -> Request:
        self._trim_stale()
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0][1]

    def prepend_request(self, request: Request) -> None:
        # Prepend follows policy ordering semantics for heap-backed queue.
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        self._active.discard(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            self._active.discard(request)

    def __bool__(self) -> bool:
        return bool(self._active)

    def __len__(self) -> int:
        return len(self._active)

    def __iter__(self) -> Iterator[Request]:
        return iter(sorted(self._active, key=self._key))


class LongestPromptFirstScheduler(Scheduler):
    """vLLM scheduler variant using longest-prompt-first for waiting queues."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.waiting = PromptLengthRequestQueue(longest_first=True)
        self.skipped_waiting = PromptLengthRequestQueue(longest_first=True)


class ShortestPromptFirstScheduler(Scheduler):
    """vLLM scheduler variant using shortest-prompt-first for waiting queues."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.waiting = PromptLengthRequestQueue(longest_first=False)
        self.skipped_waiting = PromptLengthRequestQueue(longest_first=False)
