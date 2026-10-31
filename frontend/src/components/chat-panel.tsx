"use client";

import { useEffect, useRef, useState } from "react";

import { ChatMessage } from "@/lib/types";
import { SourceCitations } from "@/components/source-citations";

type ChatPanelProps = {
  hasDocuments: boolean;
  pending: boolean;
  error: string | null;
  messages: ChatMessage[];
  onAsk: (question: string) => Promise<void>;
};

export function ChatPanel({
  hasDocuments,
  pending,
  error,
  messages,
  onAsk,
}: ChatPanelProps) {
  const [question, setQuestion] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || pending || !hasDocuments) {
      return;
    }

    setQuestion("");
    await onAsk(trimmed);
  }

  return (
    <section className="flex h-full min-h-0 flex-col bg-[var(--dt-bg-panel)] p-3 text-[var(--dt-text-primary)]">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-semibold tracking-wide text-[var(--dt-text-secondary)]">
            Chat
          </p>
        </div>
      </div>

      <div className="mt-3 flex min-h-0 flex-1 flex-col space-y-3 overflow-y-auto rounded-lg border border-[var(--dt-border-base)] bg-[var(--dt-bg-chat)] p-3">
        {error ? (
          <div className="rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-sm text-red-200">
            {error}
          </div>
        ) : null}

        {!messages.length ? (
          <div className="flex min-h-[18rem] items-center justify-center px-4 py-6 text-center text-base text-[var(--dt-text-muted)]">
            Ask a question after indexing documents.
          </div>
        ) : null}

        {messages.map((message) => (
          <article
            key={message.id}
            className={
              message.role === "user"
                ? "ml-auto max-w-2xl rounded-lg bg-[var(--dt-bg-msg-user)] px-4 py-3 text-[var(--dt-text-primary)]"
                : message.role === "system"
                  ? "max-w-2xl rounded-lg border border-red-400/40 bg-red-500/10 px-4 py-3 text-red-200"
                  : "max-w-2xl rounded-lg border border-[var(--dt-border-inner)] bg-[var(--dt-bg-msg-assistant)] px-4 py-3 text-[var(--dt-text-primary)]"
            }
          >
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--dt-text-muted)]">
              {message.role}
            </div>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-6">
              {message.content}
            </p>
            {message.role === "assistant" && message.citations?.length ? (
              <SourceCitations citations={message.citations} />
            ) : null}
          </article>
        ))}

        {pending ? (
          <div className="max-w-2xl rounded-lg border border-[var(--dt-border-inner)] bg-[var(--dt-bg-msg-assistant)] px-4 py-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[var(--dt-text-muted)]">assistant</div>
            <div className="mt-3 flex items-center gap-1.5">
              <span className="typing-dot inline-block h-2 w-2 rounded-full bg-slate-400" />
              <span className="typing-dot inline-block h-2 w-2 rounded-full bg-slate-400" />
              <span className="typing-dot inline-block h-2 w-2 rounded-full bg-slate-400" />
            </div>
          </div>
        ) : null}

        <div ref={messagesEndRef} />
      </div>

      <form className="mt-4" onSubmit={handleSubmit}>
        <div className="flex items-center gap-2 rounded-xl border border-[var(--dt-border-inner)] bg-[var(--dt-bg-surface)] p-2">
          <textarea
            className="min-h-10 flex-1 resize-none bg-transparent px-2 py-1 text-sm leading-6 text-[var(--dt-text-primary)] outline-none placeholder:text-[var(--dt-text-muted)]"
            placeholder={
              hasDocuments
                ? "Start typing..."
                : "Upload at least one document before asking a question."
            }
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            disabled={!hasDocuments || pending}
            rows={1}
          />
          <button
            className="inline-flex h-9 w-9 items-center justify-center rounded-full border border-[var(--dt-border-btn)] bg-[var(--dt-bg-msg-assistant)] text-[var(--dt-text-primary)] transition hover:bg-[var(--dt-bg-btn-hover)] disabled:cursor-not-allowed disabled:opacity-60"
            type="submit"
            disabled={!hasDocuments || pending || !question.trim()}
            aria-label={pending ? "Searching" : "Send"}
          >
            <span aria-hidden="true">➜</span>
          </button>
        </div>
      </form>
    </section>
  );
}