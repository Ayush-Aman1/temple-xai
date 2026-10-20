export type Probs = Record<string, number>;
export interface Component {
  label: string; display_name: string; score: number; box_xyxy: number[];
  supports_prediction: boolean; linked_traditions: string[]; shap: number | null; cam_energy: number | null;
}
export interface Provenance {
  label_mode: string; knowledge_mode: string; detector_mode: string; gold_eval: string;
  models: Record<string, string>; data_hash?: string; git_commit: string;
}
export interface Analysis {
  image: { width: number; height: number };
  prediction: { tradition: string; probability: number; probabilities: Probs };
  components: Component[];
  gradcam_png_base64: string | null;
  shap: { by_component: Record<string, number>; explained_class: string; model: string; parts_model_prediction: string };
  report_markdown: string; caveat: string; provenance: Provenance;
  attribution?: Attribution; true_tradition?: string;
}
export interface Attribution { image_id: string; licence: string; author: string | null; source_url: string }
export interface GalleryItem extends Attribution { true: string; predicted: string }
export interface Gallery { correct: GalleryItem[]; incorrect: GalleryItem[] }
