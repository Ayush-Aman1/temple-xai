import type { Analysis, Gallery, Provenance } from "./types";

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status}: ${(await r.text()).slice(0, 200)}`);
  return r.json() as Promise<T>;
}

export const analyzeFile = (f: File) => {
  const fd = new FormData();
  fd.append("file", f);
  return fetch("/api/analyze", { method: "POST", body: fd }).then((r) => j<Analysis>(r));
};
export const analyzeGallery = (id: string) => fetch(`/api/gallery/analyze/${id}`, { method: "POST" }).then((r) => j<Analysis>(r));
export const getGallery = () => fetch("/api/gallery").then((r) => j<Gallery>(r));
export const getHealth = () => fetch("/api/health").then((r) => j<{ status: string; provenance: Provenance }>(r));
export const getMethod = () => fetch("/api/method").then((r) => j<Record<string, any>>(r));
export const galleryImageUrl = (id: string) => `/api/gallery/image/${id}`;
