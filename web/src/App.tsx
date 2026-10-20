import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { analyzeFile, analyzeGallery, galleryImageUrl, getGallery, getHealth } from "./api";
import type { Analysis, Component, Gallery, GalleryItem, Provenance } from "./types";

const TRAD_LABEL: Record<string, string> = { Nagara: "Nagara", Dravida: "Dravida", Deccan_Vesara: "Deccan / Vesara" };
const COLORS = ["#b45309", "#0f766e", "#7c3aed", "#be123c", "#1d4ed8", "#4d7c0f", "#c2410c"];

function Caveat({ text }: { text: string }) {
  return (
    <div role="note" className="rounded-md border border-amber-300 bg-amber-50 px-4 py-2 text-sm text-amber-900">
      <b>Operational taxonomy.</b> {text}
    </div>
  );
}

const MODE_TEXT: Record<string, Record<string, string>> = {
  label_mode: {
    RULE_BASED: "Style labels come from a documented rule over Wikidata, Wikipedia and Commons categories; conflicting temples are excluded.",
    HUMAN_VERIFIED: "Style labels were verified by people.",
  },
  knowledge_mode: {
    MACHINE_COMPILED: "Definitions and links were compiled from fetched open sources; every claim cites a sentence.",
    PARTLY_HUMAN_REVIEWED: "Some knowledge entries were reviewed by a person; the rest are machine-compiled.",
    HUMAN_REVIEWED: "A person reviewed the knowledge entries.",
  },
  detector_mode: {
    DISTILLED: "The component detector was trained on boxes from a zero-shot detector (Grounding DINO), not on hand-drawn boxes.",
    HUMAN_CORRECTED: "The component detector was trained on human-corrected boxes.",
  },
  gold_eval: {
    UNAVAILABLE: "No hand-drawn gold boxes yet, so box-accuracy metrics are shown as “available with the gold-set add-on”.",
    AVAILABLE: "Gold boxes exist; box-accuracy metrics are computed.",
  },
};

function MethodPanel({ prov }: { prov: Provenance | null }) {
  if (!prov) return null;
  return (
    <details className="rounded-md border border-stone-300 bg-white p-4 text-sm" open>
      <summary className="cursor-pointer font-semibold">Method — how this result was produced</summary>
      <ul className="mt-3 space-y-2">
        {(["label_mode", "knowledge_mode", "detector_mode", "gold_eval"] as const).map((k) => (
          <li key={k}>
            <code className="rounded bg-stone-100 px-1">{k}={prov[k]}</code> — {MODE_TEXT[k][prov[k]] ?? ""}
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-stone-500">
        Models: {Object.values(prov.models).join(" · ")} · code {prov.git_commit}
      </p>
    </details>
  );
}

function ProbBars({ a }: { a: Analysis }) {
  return (
    <div className="space-y-2">
      {Object.entries(a.prediction.probabilities)
        .sort((x, y) => y[1] - x[1])
        .map(([t, p]) => (
          <div key={t}>
            <div className="flex justify-between text-sm">
              <span className={t === a.prediction.tradition ? "font-semibold" : ""}>{TRAD_LABEL[t] ?? t}</span>
              <span>{(p * 100).toFixed(1)}%</span>
            </div>
            <div className="h-3 rounded bg-stone-200">
              <div className={`h-3 rounded ${t === a.prediction.tradition ? "bg-amber-700" : "bg-stone-400"}`} style={{ width: `${p * 100}%` }} />
            </div>
          </div>
        ))}
    </div>
  );
}

function Viewer({ a, src }: { a: Analysis; src: string }) {
  const [showBoxes, setShowBoxes] = useState<Record<string, boolean>>({});
  const [cam, setCam] = useState(60);
  const { width, height } = a.image;
  const comps = a.components;
  useEffect(() => setShowBoxes(Object.fromEntries(comps.map((c) => [c.label, true]))), [a]); // eslint-disable-line
  return (
    <div>
      <div className="relative inline-block max-w-full">
        <img src={src} alt="analysed temple" className="block max-h-[480px] max-w-full rounded" />
        {a.gradcam_png_base64 && (
          <img src={`data:image/png;base64,${a.gradcam_png_base64}`} alt="Grad-CAM overlay" className="pointer-events-none absolute inset-0 h-full w-full rounded" style={{ opacity: cam / 100 }} />
        )}
        <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {comps.map((c, i) =>
            showBoxes[c.label] ? (
              <g key={c.label}>
                <rect x={c.box_xyxy[0]} y={c.box_xyxy[1]} width={c.box_xyxy[2] - c.box_xyxy[0]} height={c.box_xyxy[3] - c.box_xyxy[1]} fill="none" stroke={COLORS[i % COLORS.length]} strokeWidth={Math.max(2, width / 300)} />
                <text x={c.box_xyxy[0] + 3} y={c.box_xyxy[1] + width / 45} fontSize={width / 45} fill={COLORS[i % COLORS.length]} stroke="white" strokeWidth={width / 400} paintOrder="stroke">
                  {c.display_name}
                </text>
              </g>
            ) : null,
          )}
        </svg>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <label className="flex items-center gap-2">
          Grad-CAM
          <input type="range" min={0} max={100} value={cam} onChange={(e) => setCam(+e.target.value)} aria-label="Grad-CAM opacity" />
          <span className="w-8 text-right">{cam}%</span>
        </label>
        {comps.map((c, i) => (
          <label key={c.label} className="flex items-center gap-1">
            <input type="checkbox" checked={!!showBoxes[c.label]} onChange={(e) => setShowBoxes({ ...showBoxes, [c.label]: e.target.checked })} />
            <span style={{ color: COLORS[i % COLORS.length] }}>{c.display_name}</span>
          </label>
        ))}
      </div>
      <p className="mt-1 text-xs text-stone-500">The heat-map covers the central crop the model sees. It shows where the image model looked; it is not proof of correctness.</p>
    </div>
  );
}

function ShapChart({ a }: { a: Analysis }) {
  const rows = Object.entries(a.shap.by_component).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
  const max = Math.max(0.01, ...rows.map(([, v]) => Math.abs(v)));
  if (!rows.length) return <p className="text-sm text-stone-600">No components detected, so there are no component contributions.</p>;
  return (
    <div>
      <p className="mb-2 text-sm text-stone-600">
        Contribution of each detected component to the probability of <b>{TRAD_LABEL[a.shap.explained_class] ?? a.shap.explained_class}</b> in the {a.shap.model} (SHAP). This explains the part-based model
        {a.shap.parts_model_prediction !== a.shap.explained_class ? ` — which on its own would have said ${TRAD_LABEL[a.shap.parts_model_prediction]}` : ""}.
      </p>
      <div className="space-y-1">
        {rows.map(([k, v]) => (
          <div key={k} className="grid grid-cols-[110px_1fr_60px] items-center gap-2 text-sm">
            <span>{a.components.find((c) => c.label === k)?.display_name ?? k}</span>
            <div className="relative h-4 bg-stone-100">
              <div className="absolute inset-y-0 left-1/2 w-px bg-stone-400" />
              <div className={`absolute inset-y-0 ${v >= 0 ? "bg-teal-600" : "bg-rose-600"}`} style={v >= 0 ? { left: "50%", width: `${(v / max) * 50}%` } : { right: "50%", width: `${(-v / max) * 50}%` }} />
            </div>
            <span className="text-right tabular-nums">{v >= 0 ? "+" : ""}{v.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ComponentList({ comps, pred }: { comps: Component[]; pred: string }) {
  if (!comps.length) return <p className="text-sm text-stone-600">No architectural components were detected above threshold.</p>;
  return (
    <ul className="space-y-1 text-sm">
      {comps.map((c) => (
        <li key={c.label}>
          <b>{c.display_name}</b> — detector score {c.score.toFixed(2)};{" "}
          {c.supports_prediction ? <span className="text-teal-700">cited as evidence for {TRAD_LABEL[pred]}</span> : <span className="text-stone-500">not cited as specific to {TRAD_LABEL[pred]}</span>}
        </li>
      ))}
    </ul>
  );
}

function Result({ a, src }: { a: Analysis; src: string }) {
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <section className="space-y-4">
        <div className="rounded-md border border-stone-300 bg-white p-4">
          <h2 className="text-lg font-semibold">
            Predicted tradition: {TRAD_LABEL[a.prediction.tradition]} <span className="font-normal text-stone-600">({(a.prediction.probability * 100).toFixed(0)}%)</span>
          </h2>
          {a.true_tradition && (
            <p className="text-sm">
              Label in dataset (rule-based): <b>{TRAD_LABEL[a.true_tradition]}</b> — {a.true_tradition === a.prediction.tradition ? <span className="text-teal-700">prediction matches</span> : <span className="text-rose-700">prediction differs</span>}
            </p>
          )}
          <div className="mt-3"><ProbBars a={a} /></div>
        </div>
        <div className="rounded-md border border-stone-300 bg-white p-4">
          <Viewer a={a} src={src} />
          {a.attribution && (
            <p className="mt-2 text-xs text-stone-500">
              Image: {a.attribution.author ?? "unknown author"}, {a.attribution.licence}, <a className="underline" href={a.attribution.source_url} target="_blank" rel="noreferrer">Wikimedia Commons</a>
            </p>
          )}
        </div>
        <div className="rounded-md border border-stone-300 bg-white p-4">
          <h3 className="mb-2 font-semibold">Detected components</h3>
          <ComponentList comps={a.components} pred={a.prediction.tradition} />
        </div>
      </section>
      <section className="space-y-4">
        <div className="rounded-md border border-stone-300 bg-white p-4">
          <h3 className="mb-2 font-semibold">Component contributions</h3>
          <ShapChart a={a} />
        </div>
        <article className="prose-sm rounded-md border border-stone-300 bg-white p-4 text-sm leading-relaxed [&_a]:text-amber-800 [&_a]:underline [&_blockquote]:border-l-4 [&_blockquote]:border-amber-300 [&_blockquote]:pl-3 [&_h2]:mb-2 [&_h2]:text-lg [&_h2]:font-semibold [&_h3]:mb-1 [&_h3]:mt-3 [&_h3]:font-semibold [&_li]:my-1 [&_ol]:list-decimal [&_ol]:pl-5 [&_ul]:list-disc [&_ul]:pl-5">
          <ReactMarkdown>{a.report_markdown}</ReactMarkdown>
        </article>
      </section>
    </div>
  );
}

function GalleryStrip({ title, items, onPick }: { title: string; items: GalleryItem[]; onPick: (g: GalleryItem) => void }) {
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <div className="flex flex-wrap gap-3">
        {items.map((g) => (
          <button key={g.image_id} onClick={() => onPick(g)} className="w-32 rounded border border-stone-300 bg-white p-1 text-left text-xs hover:border-amber-600 focus:outline-2 focus:outline-amber-600">
            <img src={galleryImageUrl(g.image_id)} alt={`${g.true} temple`} className="h-20 w-full rounded object-cover" loading="lazy" />
            <div className="mt-1">true {TRAD_LABEL[g.true]}</div>
            <div className="text-stone-500">pred {TRAD_LABEL[g.predicted]}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [prov, setProv] = useState<Provenance | null>(null);
  const [gallery, setGallery] = useState<Gallery | null>(null);
  const [res, setRes] = useState<Analysis | null>(null);
  const [src, setSrc] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getHealth().then((h) => setProv(h.provenance)).catch((e) => setErr(`Backend not reachable: ${e.message}`));
    getGallery().then(setGallery).catch(() => undefined);
  }, []);

  const run = async (fn: () => Promise<Analysis>, url: string) => {
    setBusy(true); setErr(""); setSrc(url);
    try { setRes(await fn()); } catch (e: any) { setErr(e.message); setRes(null); } finally { setBusy(false); }
  };
  const onFile = (f: File | undefined) => f && run(() => analyzeFile(f), URL.createObjectURL(f));
  const caveat = res?.caveat ?? "Tradition labels follow the project's operational taxonomy (Nagara / Dravida / Deccan-Vesara). Real temples mix traits; treat results as evidence for a grouping, not a verdict.";

  return (
    <main className="mx-auto max-w-6xl space-y-5 px-4 py-6">
      <header>
        <h1 className="text-2xl font-bold">Temple XAI</h1>
        <p className="text-stone-600">Explainable recognition of Indian temple architectural traditions: CNN + detected components + cited knowledge.</p>
      </header>
      <Caveat text={caveat} />
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); onFile(e.dataTransfer.files[0]); }}
        className="flex flex-col items-center gap-2 rounded-md border-2 border-dashed border-stone-400 bg-white p-6 text-center"
      >
        <p>Drop a temple photograph here, or</p>
        <button className="rounded bg-amber-700 px-4 py-2 text-white hover:bg-amber-800 disabled:opacity-50" disabled={busy} onClick={() => fileRef.current?.click()}>
          {busy ? "Analysing…" : "Choose an image"}
        </button>
        <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => onFile(e.target.files?.[0])} />
        <p className="text-xs text-stone-500">Runs entirely on this machine. Exterior photographs work best.</p>
      </div>
      {err && <div role="alert" className="rounded border border-rose-300 bg-rose-50 px-4 py-2 text-sm text-rose-800">{err}</div>}
      {res && <Result a={res} src={src} />}
      {gallery && (gallery.correct.length > 0 || gallery.incorrect.length > 0) && (
        <section className="space-y-3 rounded-md border border-stone-300 bg-white p-4">
          <h2 className="font-semibold">Test-split examples (temples the model never trained on)</h2>
          <GalleryStrip title="Predicted correctly" items={gallery.correct} onPick={(g) => run(() => analyzeGallery(g.image_id), galleryImageUrl(g.image_id))} />
          <GalleryStrip title="Predicted incorrectly" items={gallery.incorrect} onPick={(g) => run(() => analyzeGallery(g.image_id), galleryImageUrl(g.image_id))} />
        </section>
      )}
      <MethodPanel prov={res?.provenance ?? prov} />
    </main>
  );
}
