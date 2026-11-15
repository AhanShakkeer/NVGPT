/*
 * Safely render plain text containing URLs as clickable links.
 *
 * To use from public/index.html, load this file before the inline app script,
 * then replace `content.textContent = text` in addMessage() with
 * `renderMessageWithLinks(content, text)`.
 */
(function (global) {
  "use strict";

  const URL_PATTERN = /https?:\/\/[^\s<>"']+/gi;
  const TRAILING_PUNCTUATION = /[.,!?;:)}\]]+$/;

  function renderMessageWithLinks(container, text) {
    container.replaceChildren();
    const source = String(text ?? "");
    let lastIndex = 0;

    for (const match of source.matchAll(URL_PATTERN)) {
      const rawUrl = match[0];
      const start = match.index;
      let urlText = rawUrl;
      let trailing = "";

      const punctuation = urlText.match(TRAILING_PUNCTUATION);
      if (punctuation) {
        trailing = punctuation[0];
        urlText = urlText.slice(0, -trailing.length);
      }

      if (!urlText) continue;

      container.appendChild(document.createTextNode(source.slice(lastIndex, start)));
      try {
        const url = new URL(urlText);
        if (url.protocol !== "http:" && url.protocol !== "https:") {
          container.appendChild(document.createTextNode(rawUrl));
        } else {
          const link = document.createElement("a");
          link.href = url.href;
          link.textContent = urlText;
          link.target = "_blank";
          link.rel = "noopener noreferrer";
          container.appendChild(link);
          container.appendChild(document.createTextNode(trailing));
        }
      } catch {
        container.appendChild(document.createTextNode(rawUrl));
      }

      lastIndex = start + rawUrl.length;
    }

    container.appendChild(document.createTextNode(source.slice(lastIndex)));
  }

  global.renderMessageWithLinks = renderMessageWithLinks;
})(window);
