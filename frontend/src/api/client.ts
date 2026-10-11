import type {
  AskAnswer,
  BulletinResponse,
  ConfigResponse,
  Frame,
  HealthResponse,
  LogEntry,
  Proposal,
  ReliabilityArtifact,
  ReplayResponse,
  SimulationArtifact,
  SimulationJob,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  if (!response.ok) {
    // FastAPI puts the reason in `detail`; surfacing it verbatim is the whole point
    // of the 409s and 422s the backend raises.
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* not JSON; the status text will do */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body?: unknown): Promise<T> =>
  call<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => call<HealthResponse>("/api/health"),
  config: () => call<ConfigResponse>("/api/config"),
  session: () => call<Frame>("/api/session"),
  log: (limit = 200) => call<{ entries: LogEntry[] }>(`/api/session/log?limit=${limit}`),
  approvals: () =>
    call<{ pending: Proposal[]; all: Proposal[]; overrides: unknown[] }>(
      "/api/session/approvals",
    ),
  replay: (interval: number) =>
    call<ReplayResponse>(`/api/session/replay/${interval}`),

  step: (intervals = 1) => post<Frame>("/api/session/step", { intervals }),
  run: (speed_x: number) => post<Frame>("/api/session/run", { speed_x }),
  pause: () => post<Frame>("/api/session/pause"),
  reset: (body: { preset?: string; seed?: number }) => post<Frame>("/api/session/reset", body),

  inject: (body: Record<string, unknown>) => post<Frame>("/api/session/inject", body),
  bulletin: (text: string) => post<BulletinResponse>("/api/session/bulletin", { text }),
  confirmBulletin: (accept: boolean) =>
    post<{ accepted: boolean; frame: Frame }>("/api/session/bulletin/confirm", { accept }),
  weights: (body: { weights?: Record<string, number>; risk_lambda?: number; note?: string }) =>
    post<{ elapsed_ms: number; before_index: number | null; after_index: number | null; changed: boolean; frame: Frame }>(
      "/api/session/weights",
      body,
    ),
  ask: (question: string) => post<AskAnswer>("/api/session/ask", { question }),
  decideApproval: (id: string, approve: boolean) =>
    post<Proposal & { frame: Frame }>(`/api/session/approvals/${encodeURIComponent(id)}`, {
      approve,
    }),
  override: (note: string) => post<{ note: string; frame: Frame }>("/api/session/override", { note }),

  reliability: () => call<ReliabilityArtifact>("/api/reliability"),
  simulation: () => call<SimulationArtifact>("/api/simulation"),
  simulate: (body: { preset: string; paths: number; intervals?: number; strategies?: string[] }) =>
    post<SimulationJob>("/api/simulate", body),
  simulateStatus: (id: string) => call<SimulationJob>(`/api/simulate/${id}`),
};

export type SocketMessage =
  | { type: "frame"; frame: Frame }
  | { type: "error"; message: string };

/** The tick stream. Reconnects with backoff, because a paused demo should survive a
 *  backend restart without the operator reloading the page. */
export function openSocket(
  onMessage: (message: SocketMessage) => void,
  onStatus: (connected: boolean) => void,
): () => void {
  let socket: WebSocket | null = null;
  let closed = false;
  let attempt = 0;
  let timer: number | undefined;

  const connect = () => {
    if (closed) return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

    socket.onopen = () => {
      attempt = 0;
      onStatus(true);
    };
    socket.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data as string) as SocketMessage);
      } catch {
        /* a malformed frame is dropped rather than taking the console down */
      }
    };
    socket.onclose = () => {
      onStatus(false);
      if (closed) return;
      attempt = Math.min(attempt + 1, 6);
      timer = window.setTimeout(connect, Math.min(500 * 2 ** (attempt - 1), 10_000));
    };
    socket.onerror = () => socket?.close();
  };

  connect();
  return () => {
    closed = true;
    if (timer !== undefined) window.clearTimeout(timer);
    socket?.close();
  };
}
