// jsdom has no ResizeObserver or WebSocket; the console only needs them not to throw.
class NoopObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = NoopObserver;
