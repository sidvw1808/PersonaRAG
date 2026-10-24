import type { Grounding, Source } from "./api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  grounding?: Grounding;
  status: "streaming" | "done" | "error";
  error?: string;
}

export interface DrawerState {
  title: string;
  sources: Source[];
  cited: number[];
  active: number | null;
}

export const uid = () => Math.random().toString(36).slice(2, 10);
