import "@testing-library/jest-dom/vitest";

// jsdom does not implement scrollIntoView, which ChatPanel calls in an effect
// on every message change. Without this stub every test that renders the chat
// panel fails with "scrollIntoView is not a function".
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
