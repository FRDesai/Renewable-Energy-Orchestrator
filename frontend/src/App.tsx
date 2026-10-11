import { useState } from "react";
import { useSession } from "./api/useSession";
import { Header, type TabKey } from "./components/Header";
import { LiveOperations } from "./tabs/LiveOperations";
import { ScenarioSimulation } from "./tabs/ScenarioSimulation";
import { ReliabilityTests } from "./tabs/ReliabilityTests";
import { HowItDecides } from "./tabs/HowItDecides";

export function App() {
  const [tab, setTab] = useState<TabKey>("live");
  const { frame, config, log, connected, error, busy, act, dismissError } = useSession();

  return (
    <div className="app">
      <Header
        frame={frame}
        config={config}
        tab={tab}
        onTab={setTab}
        connected={connected}
        busy={busy}
        act={act}
      />
      <main>
        {error && (
          <div className="banner critical" style={{ marginBottom: 12 }}>
            <div className="grow">
              <strong>{error}</strong>
            </div>
            <button className="btn small" onClick={dismissError}>
              Dismiss
            </button>
          </div>
        )}

        {tab === "live" &&
          (frame ? (
            <LiveOperations
              frame={frame}
              config={config}
              log={log}
              busy={busy}
              act={act}
            />
          ) : (
            <p className="empty">
              Waiting for the decision service on :8000. Start it with <code>make api</code>.
            </p>
          ))}
        {tab === "simulation" && <ScenarioSimulation config={config} />}
        {tab === "reliability" && <ReliabilityTests />}
        {tab === "how" && <HowItDecides config={config} />}
      </main>
    </div>
  );
}
