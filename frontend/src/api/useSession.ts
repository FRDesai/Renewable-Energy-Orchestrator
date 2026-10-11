import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, openSocket, type SocketMessage } from "./client";
import type { ConfigResponse, Frame, LogEntry } from "./types";

export interface SessionState {
  frame: Frame | null;
  config: ConfigResponse | null;
  log: LogEntry[];
  connected: boolean;
  /** The last action's error, shown as a dismissable banner rather than swallowed. */
  error: string | null;
  busy: boolean;
}

/** Owns the one live session the whole console reads from.
 *
 * Every mutation goes through `act`, which keeps exactly one request in flight, applies
 * the frame the backend returns, and refreshes the log. The WebSocket then delivers the
 * same frame again to every other connected console, which is harmless: a frame is a
 * complete snapshot, so applying it twice is applying it once.
 */
export function useSession() {
  const [state, setState] = useState<SessionState>({
    frame: null,
    config: null,
    log: [],
    connected: false,
    error: null,
    busy: false,
  });
  const inFlight = useRef(false);

  const refreshLog = useCallback(async () => {
    try {
      const { entries } = await api.log(200);
      setState((s) => ({ ...s, log: entries }));
    } catch {
      /* the log is supporting detail; losing it must not break the view */
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [config, frame] = await Promise.all([api.config(), api.session()]);
        if (!cancelled) setState((s) => ({ ...s, config, frame }));
        await refreshLog();
      } catch (err) {
        if (!cancelled) {
          setState((s) => ({
            ...s,
            error:
              err instanceof Error
                ? `Cannot reach the decision service: ${err.message}`
                : "Cannot reach the decision service",
          }));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshLog]);

  useEffect(() => {
    const close = openSocket(
      (message: SocketMessage) => {
        if (message.type === "frame") {
          setState((s) => ({ ...s, frame: message.frame }));
        } else {
          setState((s) => ({ ...s, error: message.message }));
        }
      },
      (connected) => setState((s) => ({ ...s, connected })),
    );
    return close;
  }, []);

  // A frame arriving over the socket may carry a new log entry, so the log is pulled
  // again whenever the interval moves or a re-plan happens.
  const stamp = state.frame
    ? `${state.frame.clock.interval}:${state.frame.status.reason}`
    : "";
  useEffect(() => {
    if (stamp) void refreshLog();
  }, [stamp, refreshLog]);

  /** Run one session mutation, apply its frame, and surface any refusal. */
  const act = useCallback(
    async <T>(fn: () => Promise<T>, pick?: (result: T) => Frame | undefined): Promise<T | null> => {
      if (inFlight.current) return null;
      inFlight.current = true;
      setState((s) => ({ ...s, busy: true, error: null }));
      try {
        const result = await fn();
        const frame = pick
          ? pick(result)
          : ((result as unknown as { clock?: unknown }).clock
              ? (result as unknown as Frame)
              : undefined);
        setState((s) => ({ ...s, frame: frame ?? s.frame, busy: false }));
        return result;
      } catch (err) {
        const message =
          err instanceof ApiError
            ? `${err.message} (${err.status})`
            : err instanceof Error
              ? err.message
              : "request failed";
        setState((s) => ({ ...s, busy: false, error: message }));
        return null;
      } finally {
        inFlight.current = false;
      }
    },
    [],
  );

  const dismissError = useCallback(() => setState((s) => ({ ...s, error: null })), []);

  return { ...state, act, dismissError, refreshLog };
}
