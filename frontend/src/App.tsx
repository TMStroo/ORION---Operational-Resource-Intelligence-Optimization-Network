/**
 * App shell: navigation, health, and the reducer store.
 *
 * Health is polled on mount and the banner shows the real backend status. If
 * the backend is down the banner says so rather than letting every page render
 * an empty state that looks like "no data yet".
 */

import React, { useEffect, useReducer } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { api } from "./api/client";
import { ErrorBanner } from "./components/ui";
import { Comparison } from "./pages/Comparison";
import { Disruptions } from "./pages/Disruptions";
import { Experiments } from "./pages/Experiments";
import { Optimization } from "./pages/Optimization";
import { Overview } from "./pages/Overview";
import { Planning } from "./pages/Planning";
import { Replanning } from "./pages/Replanning";
import { Scenarios } from "./pages/Scenarios";
import { Simulation } from "./pages/Simulation";
import { WhatIf } from "./pages/WhatIf";
import { StoreContext, initialState, reducer } from "./state/store";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/scenarios", label: "Scenarios" },
  { to: "/optimization", label: "Optimization" },
  { to: "/planning", label: "Planning" },
  { to: "/simulation", label: "Simulation" },
  { to: "/disruptions", label: "Disruptions" },
  { to: "/replanning", label: "Replanning" },
  { to: "/what-if", label: "What-if" },
  { to: "/comparison", label: "Comparison" },
  { to: "/experiments", label: "Experiments" },
];

export default function App() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const store = React.useMemo(() => ({ state, dispatch }), [state]);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const h = await api.health();
        if (!cancelled) dispatch({ type: "health", payload: h });
      } catch (e) {
        if (!cancelled) {
          dispatch({ type: "health", payload: null });
          dispatch({ type: "error", message: (e as Error).message });
        }
      }
    };
    void check();
    const id = window.setInterval(check, 30000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const unreachable = state.health === null && state.error !== null;

  return (
    <StoreContext.Provider value={store}>
      <div className="shell">
        <header className="topbar">
          <div className="brand">
            <strong>ORION</strong>
            <span className="brand-sub">Operational Resource Intelligence &amp; Optimization Network</span>
          </div>
          <div className="health">
            {state.health ? (
              <>
                <span className={`dot ${state.health.status === "ok" ? "ok" : "warn"}`} />
                <span>
                  backend {state.health.status} · {state.health.live_experiments} live /{" "}
                  {state.health.superseded_experiments} superseded experiments
                </span>
              </>
            ) : (
              <>
                <span className="dot bad" />
                <span>backend unreachable</span>
              </>
            )}
          </div>
        </header>

        <nav className="nav">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
            >
              {n.label}
            </NavLink>
          ))}
        </nav>

        <main className="main">
          {unreachable && (
            <ErrorBanner
              message={`${state.error} — start the backend with: uvicorn orion.api.asgi:app`}
            />
          )}
          <Routes>
            <Route path="/" element={<Overview state={state} />} />
            <Route path="/scenarios" element={<Scenarios state={state} dispatch={dispatch} />} />
            <Route path="/optimization" element={<Optimization state={state} dispatch={dispatch} />} />
            <Route path="/planning" element={<Planning state={state} dispatch={dispatch} />} />
            <Route path="/simulation" element={<Simulation state={state} dispatch={dispatch} />} />
            <Route path="/disruptions" element={<Disruptions state={state} dispatch={dispatch} />} />
            <Route path="/replanning" element={<Replanning state={state} dispatch={dispatch} />} />
            <Route path="/what-if" element={<WhatIf state={state} dispatch={dispatch} />} />
            <Route path="/comparison" element={<Comparison state={state} dispatch={dispatch} />} />
            <Route path="/experiments" element={<Experiments state={state} dispatch={dispatch} />} />
          </Routes>
        </main>

        <footer className="footer">
          <span>
            Every figure on this page is computed by the ORION engine. A relaxation is labelled as
            one; TIME_LIMIT and INFEASIBLE are reported, not hidden.
          </span>
        </footer>
      </div>
    </StoreContext.Provider>
  );
}
