"""Lightweight local web interface for CiteWise PDF ingestion and question answering.

Provides a clean, dependency-free local UI using Python's built-in http.server.
Enables PDF upload, ingestion report inspection, and citation-backed question answering.
"""

import argparse
from dataclasses import asdict
from email import message_from_bytes
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional

from citewise.answer_generator import INSUFFICIENT_EVIDENCE, generate_answer
from citewise.embedding_index import load_embedding_index
from citewise.evidence_gate import evaluate_evidence_gate
from citewise.hybrid_retriever import HybridRetriever
from citewise.ocr import OCRProcessingError
from citewise.pdf_loader import PDFLoadError
from citewise.pipeline import process_pdf
from citewise.retriever import BM25Retriever, load_chunks


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CiteWise — PDF Question Answering</title>
  <style>
    :root {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --accent: #f1f5f9;
      --success-bg: #ecfdf5;
      --success-border: #a7f3d0;
      --success-text: #065f46;
      --warn-bg: #fffbeb;
      --warn-border: #fde68a;
      --warn-text: #92400e;
      --danger-bg: #fef2f2;
      --danger-border: #fecaca;
      --danger-text: #991b1b;
      --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-family);
      background-color: var(--bg);
      color: var(--text-main);
      line-height: 1.5;
      padding: 24px;
    }
    header {
      max-width: 1200px;
      margin: 0 auto 24px auto;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    header h1 {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    header .subtitle {
      font-size: 0.875rem;
      color: var(--text-muted);
    }
    .layout {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 24px;
      align-items: start;
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .card h2 {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 16px;
      color: var(--text-main);
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    label {
      display: block;
      font-size: 0.8125rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    input[type="text"], input[type="file"], textarea, select {
      width: 100%;
      padding: 10px 12px;
      font-size: 0.9375rem;
      font-family: inherit;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text-main);
      margin-bottom: 14px;
    }
    textarea {
      resize: vertical;
      min-height: 90px;
    }
    input:focus, textarea:focus, select:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(37,99,235,0.1);
    }
    button {
      background-color: var(--primary);
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 10px 16px;
      font-size: 0.875rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: background-color 0.15s ease;
    }
    button:hover {
      background-color: var(--primary-hover);
    }
    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .btn-secondary {
      background-color: var(--accent);
      color: var(--text-main);
      border: 1px solid var(--border);
    }
    .btn-secondary:hover {
      background-color: #e2e8f0;
    }
    .doc-item {
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-bottom: 10px;
      background: var(--bg);
      font-size: 0.875rem;
    }
    .doc-item .name {
      font-weight: 600;
      color: var(--text-main);
      word-break: break-all;
    }
    .doc-item .meta {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
      display: flex;
      gap: 12px;
    }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      font-size: 0.7rem;
      font-weight: 600;
      border-radius: 4px;
    }
    .badge-success { background: var(--success-bg); color: var(--success-text); }
    .badge-warn { background: var(--warn-bg); color: var(--warn-text); }
    .badge-danger { background: var(--danger-bg); color: var(--danger-text); }
    .alert {
      padding: 12px 14px;
      border-radius: 6px;
      font-size: 0.875rem;
      margin-bottom: 16px;
      display: none;
    }
    .alert-success { background: var(--success-bg); border: 1px solid var(--success-border); color: var(--success-text); }
    .alert-warn { background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn-text); }
    .alert-danger { background: var(--danger-bg); border: 1px solid var(--danger-border); color: var(--danger-text); }
    .source-badge {
      display: inline-block;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      color: #1e40af;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.75rem;
      margin-right: 4px;
    }
    .citation-link {
      display: inline-flex;
      align-items: center;
      background: #eff6ff;
      border: 1px solid #93c5fd;
      color: #1d4ed8;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.8rem;
      text-decoration: none;
      cursor: pointer;
      margin: 0 2px;
      transition: background 0.15s ease;
    }
    .citation-link:hover {
      background: #dbeafe;
      border-color: #60a5fa;
    }
    .unverified-citation {
      display: inline-flex;
      align-items: center;
      background: #fef2f2;
      border: 1px dashed #f87171;
      color: #991b1b;
      padding: 1px 6px;
      border-radius: 4px;
      font-size: 0.78rem;
      margin: 0 2px;
    }
    .source-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      margin-top: 10px;
      background: #ffffff;
      font-size: 0.875rem;
      transition: border-color 0.2s, background-color 0.2s, box-shadow 0.2s;
    }
    .source-card.cited {
      border-color: #93c5fd;
      background: #f8fafc;
      box-shadow: 0 1px 3px rgba(37,99,235,0.06);
    }
    .source-card.target-highlight {
      border-color: var(--primary) !important;
      background: #eff6ff !important;
      box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
    }
    .source-card .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      font-weight: 600;
      color: var(--text-main);
    }
    .source-card .excerpt {
      color: #334155;
      font-size: 0.84rem;
      line-height: 1.5;
      border-left: 3px solid #cbd5e1;
      padding-left: 10px;
      margin-top: 6px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .source-card.cited .excerpt {
      border-left-color: #3b82f6;
    }
    .btn-toggle-excerpt {
      background: none;
      border: none;
      color: var(--primary);
      font-size: 0.78rem;
      font-weight: 600;
      cursor: pointer;
      padding: 4px 0 0 0;
      text-decoration: underline;
      display: inline-block;
    }
    .answer-box {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
      font-size: 0.95rem;
      line-height: 1.6;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    .answer-body p { margin-bottom: 0.75rem; }
    .answer-body p:last-child { margin-bottom: 0; }
    .answer-body ul, .answer-body ol { margin: 0.5rem 0 0.75rem 1.5rem; }
    .answer-body li { margin-bottom: 0.25rem; }
    .answer-body code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.85em;
      background: #f1f5f9;
      padding: 2px 5px;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
    }
    .math-display {
      margin: 12px 0;
      padding: 10px 14px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      overflow-x: auto;
      text-align: center;
      font-family: "KaTeX_Main", "Times New Roman", Cambria, serif;
      font-size: 1.05rem;
    }
    .math-inline {
      font-family: "KaTeX_Main", "Times New Roman", Cambria, serif;
      font-style: normal;
      padding: 0 2px;
      font-size: 0.98em;
    }
    .code-block-wrapper {
      margin: 12px 0;
      border-radius: 6px;
      overflow: hidden;
      border: 1px solid #334155;
      background: #0f172a;
    }
    .code-block-header {
      background: #1e293b;
      color: #94a3b8;
      font-size: 0.72rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      padding: 4px 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid #334155;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .answer-body pre {
      background: #0f172a;
      color: #f8fafc;
      padding: 12px;
      border-radius: 0;
      overflow-x: auto;
      margin: 0;
      font-size: 0.85rem;
      line-height: 1.45;
    }
    .answer-body pre code {
      background: transparent;
      color: inherit;
      border: none;
      padding: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .unsupported-passages-details {
      margin-top: 16px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      background: #f8fafc;
      overflow: hidden;
    }
    .unsupported-passages-details summary {
      padding: 10px 14px;
      font-weight: 600;
      font-size: 0.875rem;
      color: #64748b;
      cursor: pointer;
      user-select: none;
      outline: none;
      background: #f1f5f9;
      transition: background 0.15s ease;
    }
    .unsupported-passages-details summary:hover {
      background: #e2e8f0;
      color: #334155;
    }
    .unsupported-passages-body {
      padding: 12px 14px;
      border-top: 1px solid #e2e8f0;
    }
    .unsupported-passages-note {
      font-size: 0.8rem;
      color: #64748b;
      margin-bottom: 10px;
      font-style: italic;
    }
    .citations-banner {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      margin-top: 12px;
      font-size: 0.8125rem;
    }
    .spinner {
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-radius: 50%;
      border-top-color: #fff;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .row { display: flex; gap: 12px; }
    .row > * { flex: 1; }
  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" crossorigin="anonymous">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" crossorigin="anonymous"></script>
</head>
<body>
  <header>
    <div>
      <h1>CiteWise</h1>
      <div class="subtitle">Grounded PDF Question Answering & Verification Console</div>
    </div>
  </header>

  <div class="layout">
    <!-- Left Column: Ingestion & Document Manager -->
    <aside>
      <div class="card">
        <h2>Upload & Ingest PDF</h2>
        <div id="uploadAlert" class="alert"></div>
        <form id="uploadForm">
          <label for="pdfFile">Select PDF Document</label>
          <input type="file" id="pdfFile" accept=".pdf" required>
          <button type="submit" id="uploadBtn">Ingest Document</button>
        </form>
      </div>

      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
          <h2 style="margin-bottom:0; border:none; padding:0;">User Documents</h2>
          <div>
            <button type="button" class="btn-secondary" id="toggleSelectAllBtn" style="padding: 2px 6px; font-size: 0.75rem; margin-right:4px;" onclick="toggleSelectAllDocs()">Deselect All</button>
            <button type="button" class="btn-secondary" style="padding: 2px 6px; font-size: 0.75rem;" onclick="loadDocuments()">Refresh</button>
          </div>
        </div>
        <div id="docList">Loading documents...</div>
      </div>
    </aside>

    <!-- Right Column: Question Answering Console -->
    <main>
      <div class="card">
        <h2>Ask a Grounded Question</h2>
        <div id="queryAlert" class="alert"></div>
        <form id="queryForm">
          <label for="questionInput">Your Question</label>
          <textarea id="questionInput" placeholder="e.g., What are the four primary functions that comprise the NIST AI RMF Core?" required></textarea>

          <div class="row">
            <div>
              <label for="retrieverSelect">Retriever Engine</label>
              <select id="retrieverSelect">
                <option value="bm25" selected>BM25 (Offline Keyword Relevance)</option>
                <option value="hybrid">Hybrid (BM25 + Semantic Cosine RRF)</option>
              </select>
            </div>
            <div>
              <label for="topKInput">Passage Limit (Top-K)</label>
              <select id="topKInput">
                <option value="3">Top 3 Passages</option>
                <option value="5" selected>Top 5 Passages</option>
                <option value="8">Top 8 Passages</option>
              </select>
            </div>
          </div>

          <button type="submit" id="askBtn">Ask Question</button>
        </form>
      </div>

      <!-- Results Area -->
      <div id="resultsCard" class="card" style="display:none;">
        <h2 id="resultTitle">Generated Answer</h2>
        <div id="answerContent" class="answer-box"></div>

        <div id="sourcesContainer">
          <h3 style="font-size: 0.9rem; margin-top: 16px; margin-bottom: 8px; color: var(--text-muted);">Supporting Evidence Passages</h3>
          <div id="sourcesList"></div>
        </div>
      </div>
    </main>
  </div>

  <script>
    async function loadDocuments() {
      const docList = document.getElementById('docList');
      try {
        const res = await fetch('/api/documents');
        const data = await res.json();
        if (!data.documents || data.documents.length === 0) {
          docList.innerHTML = '<div style="color:var(--text-muted); font-size:0.875rem;">No indexed documents found in user workspace. Upload a PDF above.</div>';
          return;
        }
        docList.innerHTML = data.documents.map(d => `
          <div class="doc-item" style="display:flex; align-items:flex-start; gap:8px;">
            <input type="checkbox" class="doc-checkbox" value="${escapeHtml(d.filename)}" checked style="margin-top:4px;">
            <div style="flex:1;">
              <div class="name" style="font-weight:600;">${escapeHtml(d.filename)}</div>
              <div class="meta" style="margin-top:2px;">
                <span>Pages: ${d.page_count}</span>
                <span>Chunks: ${d.chunk_count}</span>
                <span class="badge ${d.status === 'success' ? 'badge-success' : 'badge-warn'}">${d.status}</span>
              </div>
            </div>
          </div>
        `).join('');
      } catch (err) {
        docList.innerHTML = `<div style="color:var(--danger-text); font-size:0.8125rem;">Failed to load documents: ${escapeHtml(err.message)}</div>`;
      }
    }

    function toggleSelectAllDocs() {
      const checkboxes = document.querySelectorAll('.doc-checkbox');
      const allChecked = Array.from(checkboxes).every(cb => cb.checked);
      checkboxes.forEach(cb => cb.checked = !allChecked);
      const btn = document.getElementById('toggleSelectAllBtn');
      if (btn) btn.textContent = allChecked ? 'Select All' : 'Deselect All';
    }

    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('pdfFile');
      const uploadBtn = document.getElementById('uploadBtn');
      const alert = document.getElementById('uploadAlert');
      
      if (!fileInput.files || fileInput.files.length === 0) return;
      const file = fileInput.files[0];

      uploadBtn.disabled = true;
      uploadBtn.textContent = 'Processing PDF...';
      alert.style.display = 'none';

      try {
        const formData = new FormData();
        formData.append('pdf', file);

        const res = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();

        if (res.ok && data.status === 'success') {
          alert.className = 'alert alert-success';
          alert.textContent = `Successfully ingested "${data.filename}" (${data.pages} pages, ${data.chunks} chunks).`;
          alert.style.display = 'block';
          fileInput.value = '';
          loadDocuments();
        } else {
          alert.className = 'alert alert-danger';
          alert.textContent = `Ingestion error: ${data.error || 'Unknown error'}`;
          alert.style.display = 'block';
        }
      } catch (err) {
        alert.className = 'alert alert-danger';
        alert.textContent = `Network error: ${err.message}`;
        alert.style.display = 'block';
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Ingest Document';
      }
    });

    document.getElementById('queryForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const question = document.getElementById('questionInput').value.trim();
      const retriever = document.getElementById('retrieverSelect').value;
      const topK = parseInt(document.getElementById('topKInput').value, 10);
      const askBtn = document.getElementById('askBtn');
      const alert = document.getElementById('queryAlert');
      const resultsCard = document.getElementById('resultsCard');
      const answerContent = document.getElementById('answerContent');
      const sourcesList = document.getElementById('sourcesList');

      if (!question) return;

      const checkedBoxes = Array.from(document.querySelectorAll('.doc-checkbox:checked'));
      const selectedDocs = checkedBoxes.map(cb => cb.value);

      askBtn.disabled = true;
      askBtn.innerHTML = '<span class="spinner"></span> Searching & Generating...';
      alert.style.display = 'none';
      resultsCard.style.display = 'none';

      try {
        const res = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, retriever, top_k: topK, selected_documents: selectedDocs }),
        });
        const data = await res.json();

        if (!res.ok || data.status === 'error') {
          alert.className = 'alert alert-danger';
          alert.textContent = data.error || 'Failed to process question.';
          alert.style.display = 'block';
          return;
        }

        resultsCard.style.display = 'block';

        if (data.state === 'generation_error') {
          alert.className = 'alert alert-danger';
          alert.innerHTML = `<strong>Generation Service Error:</strong> ${escapeHtml(data.error)}<br><small style="color:var(--danger-text);">Retrieved evidence passages are preserved below.</small>`;
          alert.style.display = 'block';
          answerContent.innerHTML = `<div style="color:var(--text-muted); font-style:italic;">Generation could not be completed. Inspect retrieved passages below.</div>`;
        } else if (data.state === 'insufficient_evidence') {
          alert.style.display = 'none';
          let explanationText = '';
          if (data.gate && data.gate.decision === 'reject') {
            explanationText = data.gate.explanation || data.gate.reason || 'Insufficient evidence found in document corpus.';
          } else {
            explanationText = data.answer || 'The provided PDF passages do not contain enough evidence to answer this question reliably.';
          }
          answerContent.innerHTML = `<div class="alert alert-warn" style="margin-bottom:0; display:block;">
            <strong>Insufficient Evidence:</strong> ${escapeHtml(explanationText)}
          </div>`;
        } else {
          alert.style.display = 'none';
          let gateBadge = '';
          if (data.gate && data.gate.decision === 'borderline') {
            gateBadge = `<div style="margin-bottom:12px;"><span class="badge badge-warn">Evidence Gate: Borderline</span> <small style="color:var(--text-muted);">${escapeHtml(data.gate.explanation || '')}</small></div>`;
          } else if (data.gate && data.gate.decision === 'pass') {
            gateBadge = `<div style="margin-bottom:12px;"><span class="badge badge-success">Evidence Gate: Verified</span></div>`;
          }

          let citationsBanner = '';
          if (data.citations && data.citations.length > 0) {
            const validCitationPills = data.citations.map(c => {
              const matchingPassage = (data.passages || []).find(p => p.chunk_id === c.chunk_id);
              if (!matchingPassage) return '';
              const sId = matchingPassage.source_id || c.source_id || 'S1';
              return `<a href="#source-card-${sId}" class="citation-link" onclick="highlightSource('${sId}'); return false;">[${escapeHtml(sId)}: ${escapeHtml(c.filename)} p.${c.page_number}]</a>`;
            }).filter(Boolean);

            if (validCitationPills.length > 0) {
              citationsBanner = `
                <div class="citations-banner">
                  <strong style="color:var(--text-muted); margin-right:4px;">Citations:</strong>
                  ${validCitationPills.join('')}
                </div>
              `;
            }
          }

          answerContent.innerHTML = gateBadge + `<div class="answer-body">${renderMarkdown(data.answer, data.citations, data.passages)}</div>` + citationsBanner;
        }

        // Render Sources Container conditionally based on response state
        const sourcesContainer = document.getElementById('sourcesContainer');
        const passages = data.passages || [];

        function renderPassageCards(passagesList, isInsufficient) {
          if (!passagesList || passagesList.length === 0) {
            return '<div style="color:var(--text-muted); font-size:0.875rem;">No passages retrieved.</div>';
          }
          return passagesList.map(p => {
            const isCited = !isInsufficient && data.citations && data.citations.some(c => c.chunk_id === p.chunk_id);
            const text = p.text || '';
            const isLong = text.length > 260;
            const shortText = isLong ? text.substring(0, 240) + '...' : text;
            return `
              <div id="source-card-${escapeHtml(p.source_id)}" class="source-card ${isCited ? 'cited' : ''}">
                <div class="header">
                  <div>
                    <span class="source-badge">${escapeHtml(p.source_id)}</span>
                    <strong>${escapeHtml(p.filename)}</strong> — Page ${p.page_number}
                    ${isCited ? '<span class="badge badge-success" style="margin-left:6px;">Cited in Answer</span>' : ''}
                  </div>
                  <div style="font-size:0.75rem; color:var(--text-muted); font-family:monospace;">${escapeHtml(p.chunk_id)}</div>
                </div>
                <div class="excerpt" id="excerpt-${escapeHtml(p.chunk_id)}" data-short="${escapeHtml(shortText)}" data-full="${escapeHtml(text)}" data-expanded="false">${escapeHtml(shortText)}</div>
                ${isLong ? `<button type="button" class="btn-toggle-excerpt" id="toggle-btn-${escapeHtml(p.chunk_id)}" onclick="toggleExcerpt('${escapeHtml(p.chunk_id)}')">Show full excerpt</button>` : ''}
              </div>
            `;
          }).join('');
        }

        if (data.state === 'insufficient_evidence') {
          // Unsupported / out-of-context query:
          // Do NOT display retrieved passages under a "Supporting Evidence" heading.
          // Instead, put them into a collapsed details section for inspection.
          sourcesContainer.innerHTML = `
            <details class="unsupported-passages-details">
              <summary>Retrieved passages (not sufficient to answer) — ${passages.length} retrieved</summary>
              <div class="unsupported-passages-body">
                <div class="unsupported-passages-note">
                  The passages below were retrieved from the document corpus, but did not contain sufficient grounded evidence to answer your query. They are preserved here for transparency.
                </div>
                <div id="sourcesList">${renderPassageCards(passages, true)}</div>
              </div>
            </details>
          `;
        } else if (data.state === 'generation_error') {
          sourcesContainer.innerHTML = `
            <h3 style="font-size: 0.9rem; margin-top: 16px; margin-bottom: 8px; color: var(--danger-text);">
              Retrieved Passages (Preserved upon service error)
            </h3>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:8px;">
              Answer generation failed, but the retrieved document passages below were preserved.
            </div>
            <div id="sourcesList">${renderPassageCards(passages, false)}</div>
          `;
        } else {
          // Supported answered state
          sourcesContainer.innerHTML = `
            <h3 style="font-size: 0.9rem; margin-top: 16px; margin-bottom: 8px; color: var(--text-muted);">
              Supporting Evidence Passages
            </h3>
            <div id="sourcesList">${renderPassageCards(passages, false)}</div>
          `;
        }

      } catch (err) {
        alert.className = 'alert alert-danger';
        alert.textContent = `Client error: ${err.message}`;
        alert.style.display = 'block';
      } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Ask Question';
      }
    });

    function formatSimpleMath(latexStr) {
      if (!latexStr) return '';
      return escapeHtml(latexStr)
        .replace(/\\times/g, '×')
        .replace(/\\le(q)?/g, '≤')
        .replace(/\\ge(q)?/g, '≥')
        .replace(/\\neq/g, '≠')
        .replace(/\\approx/g, '≈')
        .replace(/\\pm/g, '±')
        .replace(/\\cdot/g, '·')
        .replace(/\\alpha/g, 'α')
        .replace(/\\beta/g, 'β')
        .replace(/\\gamma/g, 'γ')
        .replace(/\\delta/g, 'δ')
        .replace(/\\pi/g, 'π')
        .replace(/\\sigma/g, 'σ')
        .replace(/\\theta/g, 'θ')
        .replace(/\\lambda/g, 'λ')
        .replace(/\\mu/g, 'μ')
        .replace(/\\infty/g, '∞')
        .replace(/\\sum/g, '∑')
        .replace(/\\prod/g, '∏')
        .replace(/\\int/g, '∫')
        .replace(/\\sqrt\{([^}]+)\}/g, '√($1)')
        .replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '($1 / $2)')
        .replace(/\^\{([^}]+)\}/g, '<sup>$1</sup>')
        .replace(/\^([0-9a-zA-Z])/g, '<sup>$1</sup>')
        .replace(/_\{([^}]+)\}/g, '<sub>$1</sub>')
        .replace(/_([0-9a-zA-Z])/g, '<sub>$1</sub>');
    }

    function renderMathExpression(expr, displayMode) {
      const cleanExpr = expr.trim();
      if (typeof window !== 'undefined' && window.katex && typeof window.katex.renderToString === 'function') {
        try {
          return window.katex.renderToString(cleanExpr, {
            displayMode: displayMode,
            throwOnError: false,
          });
        } catch (e) {
          // Graceful fallback to styled HTML
        }
      }
      if (displayMode) {
        return `<div class="math-display"><span class="math-content">${formatSimpleMath(cleanExpr)}</span></div>`;
      }
      return `<span class="math-inline">${formatSimpleMath(cleanExpr)}</span>`;
    }

    function renderMarkdown(text, citations, passages) {
      if (!text) return '';

      const tokens = [];
      let str = text;

      // 1. Code blocks with language specification
      str = str.replace(/```([a-zA-Z0-9_\-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const token = `___TOKEN_CODE_BLOCK_${tokens.length}___`;
        const langClean = (lang || 'code').trim().toLowerCase();
        const escapedCode = escapeHtml(code.trim());
        const html = `
          <div class="code-block-wrapper">
            <div class="code-block-header">
              <span>${escapeHtml(langClean)}</span>
              <span>code</span>
            </div>
            <pre><code class="language-${escapeHtml(langClean)}">${escapedCode}</code></pre>
          </div>
        `.trim();
        tokens.push({ token, html });
        return token;
      });

      // 2. Display Math: \[ ... \] or $$ ... $$
      str = str.replace(/\\\[([\s\S]*?)\\\]/g, (match, math) => {
        const token = `___TOKEN_MATH_DISP_${tokens.length}___`;
        const html = renderMathExpression(math, true);
        tokens.push({ token, html });
        return token;
      });
      str = str.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
        const token = `___TOKEN_MATH_DISP_${tokens.length}___`;
        const html = renderMathExpression(math, true);
        tokens.push({ token, html });
        return token;
      });

      // 3. Inline Math: \( ... \) or $ ... $ (avoid matching escaped or multiple dollars)
      str = str.replace(/\\\(([\s\S]*?)\\\)/g, (match, math) => {
        const token = `___TOKEN_MATH_INLINE_${tokens.length}___`;
        const html = renderMathExpression(math, false);
        tokens.push({ token, html });
        return token;
      });
      str = str.replace(/(^|[^\\$])\$([^\$\n]+?)\$(?!\$)/g, (match, prefix, math) => {
        const token = `___TOKEN_MATH_INLINE_${tokens.length}___`;
        const html = renderMathExpression(math, false);
        tokens.push({ token, html });
        return prefix + token;
      });

      // 4. Inline code
      str = str.replace(/`([^`\n]+)`/g, (match, code) => {
        const token = `___TOKEN_INLINE_CODE_${tokens.length}___`;
        const html = `<code>${escapeHtml(code)}</code>`;
        tokens.push({ token, html });
        return token;
      });

      // 5. Escape HTML for the remaining prose
      let safe = escapeHtml(str);

      // 6. Markdown Bold & Italic
      safe = safe.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
      safe = safe.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');

      // 7. Citation bracket links [S1], [S2]
      const validSourceIds = new Set();
      if (Array.isArray(passages)) {
        passages.forEach(p => { if (p && p.source_id) validSourceIds.add(p.source_id); });
      }
      if (Array.isArray(citations)) {
        citations.forEach(c => { if (c && c.source_id) validSourceIds.add(c.source_id); });
      }

      safe = safe.replace(/\[(S\d+)\]/g, (match, sId) => {
        if (validSourceIds.size === 0 || validSourceIds.has(sId)) {
          return `<a href="#source-card-${sId}" class="citation-link" onclick="highlightSource('${sId}'); return false;">[${sId}]</a>`;
        }
        return `<span class="unverified-citation" title="Unverified citation: passage not in retrieved sources">[${sId} (unverified)]</span>`;
      });

      // 8. Paragraphs and lists
      const lines = safe.split('\n');
      let inList = false;
      let listType = '';
      const htmlLines = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/);
        const numMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);

        if (bulletMatch) {
          if (!inList || listType !== 'ul') {
            if (inList) htmlLines.push(`</${listType}>`);
            htmlLines.push('<ul>');
            inList = true;
            listType = 'ul';
          }
          htmlLines.push(`<li>${bulletMatch[1]}</li>`);
        } else if (numMatch) {
          if (!inList || listType !== 'ol') {
            if (inList) htmlLines.push(`</${listType}>`);
            htmlLines.push('<ol>');
            inList = true;
            listType = 'ol';
          }
          htmlLines.push(`<li>${numMatch[2]}</li>`);
        } else {
          if (inList) {
            htmlLines.push(`</${listType}>`);
            inList = false;
            listType = '';
          }
          if (trimmed.length > 0) {
            if (trimmed.startsWith('___TOKEN_CODE_BLOCK_') || trimmed.startsWith('___TOKEN_MATH_DISP_')) {
              htmlLines.push(trimmed);
            } else {
              htmlLines.push(`<p>${trimmed}</p>`);
            }
          }
        }
      }
      if (inList) htmlLines.push(`</${listType}>`);

      let finalHtml = htmlLines.join('\n');

      // 9. Reinsert tokens
      tokens.forEach(item => {
        finalHtml = finalHtml.split(item.token).join(item.html);
      });

      return finalHtml;
    }

    function highlightSource(sourceId) {
      const el = document.getElementById(`source-card-${sourceId}`);
      if (!el) return;
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      el.classList.add('target-highlight');
      setTimeout(() => {
        el.classList.remove('target-highlight');
      }, 2000);
    }

    function toggleExcerpt(chunkId) {
      const excerptEl = document.getElementById(`excerpt-${chunkId}`);
      const btn = document.getElementById(`toggle-btn-${chunkId}`);
      if (!excerptEl || !btn) return;
      const isExpanded = excerptEl.getAttribute('data-expanded') === 'true';
      if (isExpanded) {
        excerptEl.textContent = excerptEl.getAttribute('data-short');
        excerptEl.setAttribute('data-expanded', 'false');
        btn.textContent = 'Show full excerpt';
      } else {
        excerptEl.textContent = excerptEl.getAttribute('data-full');
        excerptEl.setAttribute('data-expanded', 'true');
        btn.textContent = 'Show less';
      }
    }

    function escapeHtml(text) {
      if (!text) return '';
      return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    loadDocuments();
  </script>
</body>
</html>
"""


class CiteWiseRequestHandler(SimpleHTTPRequestHandler):
    """HTTP Request Handler serving CiteWise web UI and JSON API endpoints."""

    reports_dir: Path = Path("user_data/reports")
    index_path: Path = Path("data/semantic_index.json")
    uploads_dir: Path = Path("user_data/uploads")

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        if self.path == "/api/documents":
            self.handle_get_documents()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        if self.path == "/api/upload":
            self.handle_upload()
            return

        if self.path == "/api/ask":
            self.handle_ask()
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def handle_get_documents(self) -> None:
        """Return list of ingested PDF documents from reports directory."""
        documents = []
        if self.reports_dir.is_dir():
            for report_file in sorted(self.reports_dir.glob("*.json")):
                try:
                    payload = json.loads(report_file.read_text(encoding="utf-8"))
                    doc = payload.get("document", {})
                    documents.append({
                        "filename": doc.get("filename", report_file.name),
                        "document_id": doc.get("document_id", ""),
                        "page_count": doc.get("page_count", 0),
                        "chunk_count": payload.get("chunk_count", len(payload.get("chunks", []))),
                        "status": payload.get("status", "success"),
                        "report_file": report_file.name,
                    })
                except Exception:
                    continue

        self.send_json(HTTPStatus.OK, {"documents": documents})

    def handle_upload(self) -> None:
        """Handle PDF upload, process through pipeline, and save report."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "Expected multipart/form-data upload."},
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            # Parse multipart message using standard library email parser
            msg_bytes = f"Content-Type: {content_type}\r\n\r\n".encode("latin-1") + body_bytes
            msg = message_from_bytes(msg_bytes)

            file_payload = None
            filename = None
            if msg.is_multipart():
                for part in msg.get_payload():
                    fn = part.get_filename()
                    name = part.get_param("name", header="content-disposition")
                    if name == "pdf" or (fn and fn.lower().endswith(".pdf")):
                        filename = fn or "uploaded.pdf"
                        file_payload = part.get_payload(decode=True)
                        break

            if not file_payload or not filename:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"status": "error", "error": "Missing valid PDF file in upload."},
                )
                return

            filename = Path(filename).name
            if not filename.lower().endswith(".pdf"):
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"status": "error", "error": "Only .pdf files are accepted."},
                )
                return

            self.uploads_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)

            saved_pdf_path = self.uploads_dir / filename
            saved_pdf_path.write_bytes(file_payload)

            # Ingest through pipeline
            result = process_pdf(saved_pdf_path)
            document = result.report.document

            # Write ingestion report
            output_path = self.reports_dir / f"{document.document_id[:12]}.json"
            from dataclasses import asdict

            report_payload = {
                "document": asdict(document),
                "chunks": [asdict(c) for c in result.chunks],
                "pages_still_needing_ocr": result.report.empty_or_low_text_pages,
                "chunk_count": len(result.chunks),
                "status": (
                    "needs_review"
                    if result.report.empty_or_low_text_pages
                    else "success"
                ),
            }
            output_path.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "success",
                    "filename": document.filename,
                    "document_id": document.document_id,
                    "pages": document.page_count,
                    "chunks": len(result.chunks),
                    "pages_needing_ocr": result.report.empty_or_low_text_pages,
                },
            )

        except (PDFLoadError, OCRProcessingError, ValueError) as exc:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": str(exc)},
            )
        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "error", "error": f"Internal ingestion failure: {exc}"},
            )

    def handle_ask(self) -> None:
        """Handle question answering request using BM25 or Hybrid retrieval."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "Empty request body."},
            )
            return

        try:
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception as exc:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": f"Invalid JSON payload: {exc}"},
            )
            return

        question = body.get("question", "").strip()
        retriever_type = str(body.get("retriever", "bm25")).lower()
        if retriever_type not in ("bm25", "hybrid"):
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": f"Invalid retriever '{retriever_type}'. Supported retrievers: bm25, hybrid."},
            )
            return

        top_k_raw = body.get("top_k", 5)
        try:
            if isinstance(top_k_raw, bool):
                raise ValueError("top_k must be an integer, not boolean.")
            top_k = int(top_k_raw)
            if top_k < 1:
                raise ValueError("top_k must be a positive integer.")
        except (ValueError, TypeError):
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "Invalid top_k: top_k must be a positive integer."},
            )
            return

        selected_documents = body.get("selected_documents")

        if not question:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "Question must not be empty."},
            )
            return

        if not self.reports_dir.is_dir():
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": f"Reports directory not found: {self.reports_dir}"},
            )
            return

        report_files = sorted(self.reports_dir.glob("*.json"))
        if not report_files:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "No indexed PDF reports found. Please ingest a PDF first."},
            )
            return

        try:
            # If documents are explicitly selected, filter report files
            if selected_documents is not None:
                if not isinstance(selected_documents, list) or len(selected_documents) == 0:
                    self.send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"status": "error", "error": "No documents selected. Please select at least one document."},
                    )
                    return
                selected_set = set(selected_documents)
                filtered_reports = []
                for rf in report_files:
                    try:
                        rep = json.loads(rf.read_text(encoding="utf-8"))
                        doc_info = rep.get("document", {})
                        fn = doc_info.get("filename", rf.name)
                        did = doc_info.get("document_id", "")
                        if fn in selected_set or did in selected_set or rf.name in selected_set:
                            filtered_reports.append(rf)
                    except Exception:
                        continue
                if not filtered_reports:
                    self.send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"status": "error", "error": f"No indexed reports match selected documents: {selected_documents}"},
                    )
                    return
                report_files = filtered_reports

            chunks = load_chunks(report_files)
            if selected_documents is not None:
                selected_set = set(selected_documents)
                chunks = [
                    c for c in chunks
                    if c.get("filename") in selected_set or c.get("document_id") in selected_set
                ]

            if not chunks:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"status": "error", "error": "Selected reports contain zero chunks."},
                )
                return

            if retriever_type == "hybrid":
                if not self.index_path.is_file():
                    self.send_json(
                        HTTPStatus.BAD_REQUEST,
                        {
                            "status": "error",
                            "error": f"Semantic index not found: {self.index_path}. Build it first or use BM25.",
                        },
                    )
                    return
                index = load_embedding_index(self.index_path)
                if selected_documents is not None:
                    selected_set = set(selected_documents)
                    filtered_items = [
                        item for item in index.get("items", [])
                        if item.get("filename") in selected_set or item.get("document_id") in selected_set
                    ]
                    index = {"dimensions": index.get("dimensions", 768), "items": filtered_items}
                retriever = HybridRetriever(chunks=chunks, index=index)
                passages = retriever.search(question, top_k=top_k)
            else:
                retriever = BM25Retriever(chunks=chunks)
                passages = retriever.search(question, top_k=top_k)

        except Exception as exc:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"status": "error", "error": f"Retrieval error: {exc}"},
            )
            return

        # Pre-Generation Evidence Gate evaluation
        gate = evaluate_evidence_gate(question, passages, retriever_type=retriever_type)
        formatted_passages = [
            {
                "source_id": f"S{idx}",
                "filename": p.get("filename", ""),
                "page_number": p.get("page_number", 0),
                "chunk_id": p.get("chunk_id", ""),
                "score": round(p.get("score", 0.0), 4) if "score" in p and isinstance(p.get("score"), (int, float)) else None,
                "text": p.get("text", ""),
            }
            for idx, p in enumerate(passages, start=1)
        ]

        if not gate.should_generate:
            # Local gate rejected query: SKIP GENERATION API CALL!
            self.send_json(
                HTTPStatus.OK,
                {
                    "state": "insufficient_evidence",
                    "status": "insufficient_evidence",
                    "question": question,
                    "answerable": False,
                    "answer": INSUFFICIENT_EVIDENCE,
                    "source_ids": [],
                    "citations": [],
                    "gate": asdict(gate),
                    "passages": formatted_passages,
                },
            )
            return

        # Generate answer using grounded pipeline
        try:
            gen_res = generate_answer(question, passages)
            answerable = gen_res.get("answerable", True)
            answer = gen_res.get("answer", "")
            source_ids = gen_res.get("source_ids", [])
            citations = gen_res.get("citations", [])

            is_answered = (
                answerable is True
                and bool(answer.strip())
                and answer != INSUFFICIENT_EVIDENCE
                and bool(citations or source_ids)
            )
            state = "answered" if is_answered else "insufficient_evidence"
            status = "success" if is_answered else "insufficient_evidence"

            self.send_json(
                HTTPStatus.OK,
                {
                    "state": state,
                    "status": status,
                    "question": question,
                    "answerable": is_answered,
                    "answer": answer if is_answered else INSUFFICIENT_EVIDENCE,
                    "source_ids": source_ids if is_answered else [],
                    "citations": citations if is_answered else [],
                    "gate": asdict(gate),
                    "passages": formatted_passages,
                },
            )
        except Exception as exc:
            # Structured generation error: do NOT display insufficient evidence
            raw_err = str(exc)
            sanitized_err = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", raw_err)
            sanitized_err = re.sub(r"key=[0-9A-Za-z-_]+", "key=[REDACTED_KEY]", sanitized_err)
            sanitized_err = re.sub(r"Bearer\s+[0-9A-Za-z-_.]+", "Bearer [REDACTED_TOKEN]", sanitized_err)
            upper_err = sanitized_err.upper()
            if "503" in upper_err or "UNAVAILABLE" in upper_err:
                user_msg = "Generation service temporarily unavailable (503). Please retry shortly."
            elif "429" in upper_err or "RESOURCE_EXHAUSTED" in upper_err or "QUOTA" in upper_err:
                user_msg = "Generation rate limit or quota exceeded (429). Please wait before retrying."
            elif "TIMED OUT" in upper_err or "TIMEOUT" in upper_err:
                user_msg = "Generation request timed out. Please retry shortly."
            elif "NETWORK" in upper_err or "CONNECTION" in upper_err or "FAILED TO CONNECT" in upper_err:
                user_msg = "Network connection error contacting generation service. Please check your network and retry."
            else:
                user_msg = f"Answer Generation Error: {sanitized_err}"

            self.send_json(
                HTTPStatus.OK,
                {
                    "state": "generation_error",
                    "status": "error",
                    "error": user_msg,
                    "question": question,
                    "answerable": False,
                    "answer": "",
                    "source_ids": [],
                    "citations": [],
                    "gate": asdict(gate),
                    "passages": formatted_passages,
                },
            )

    def send_json(self, status: int, data: dict[str, Any]) -> None:
        """Send JSON response with appropriate headers."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))


def run_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    reports_dir: str | Path = "user_data/reports",
    index_path: str | Path = "data/semantic_index.json",
    uploads_dir: str | Path = "user_data/uploads",
) -> None:
    """Start the CiteWise web server."""
    r_dir = Path(reports_dir)
    r_dir.mkdir(parents=True, exist_ok=True)

    u_dir = Path(uploads_dir)
    u_dir.mkdir(parents=True, exist_ok=True)

    i_path = Path(index_path)

    CiteWiseRequestHandler.reports_dir = r_dir
    CiteWiseRequestHandler.index_path = i_path
    CiteWiseRequestHandler.uploads_dir = u_dir

    server = HTTPServer((host, port), CiteWiseRequestHandler)
    print(f"CiteWise Web UI running at http://{host}:{port}/")
    print(f"User reports directory: {r_dir}")
    print(f"User uploads directory: {u_dir}")
    print(f"Semantic index:         {i_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch the CiteWise local web user interface."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    parser.add_argument(
        "--reports",
        default="user_data/reports",
        help="Directory containing PDF ingestion reports (default: user_data/reports)",
    )
    parser.add_argument(
        "--index",
        default="data/semantic_index.json",
        help="Path to semantic index (default: data/semantic_index.json)",
    )
    parser.add_argument(
        "--uploads",
        default="user_data/uploads",
        help="Directory to save uploaded PDFs (default: user_data/uploads)",
    )
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reports_dir=args.reports,
        index_path=args.index,
        uploads_dir=args.uploads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
