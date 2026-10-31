export type DocumentSummary = {
  id: string;
  filename: string;
  content_type: string;
  page_count: number;
  chunk_count: number;
  created_at: string;
};

export type Citation = {
  document_id: string;
  filename: string;
  excerpt: string;
  page_number: number | null;
  score: number;
};

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  grounded: boolean;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  citations?: Citation[];
  grounded?: boolean;
};