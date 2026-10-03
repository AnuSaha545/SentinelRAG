import { useEffect, useState } from 'react'

import './App.css'

import {
  getAuditSummary,
  getHealth,
  getDocuments,
  uploadDocument,
  deleteDocument,
  queryDocument,
} from './api'

const API_BASE_URL = 'http://localhost:8000'

function App() {
  const [page, setPage] = useState('overview')
  const [summary, setSummary] = useState(null)
  const [backendOnline, setBackendOnline] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    loadDashboard()
  }, [])

  async function loadDashboard() {
    try {
      const [summaryData, healthData] = await Promise.all([
        getAuditSummary(),
        getHealth(),
      ])

      setSummary(summaryData)
      setBackendOnline(healthData.status === 'ok')
      setError('')
    } catch {
      setBackendOnline(false)
      setError('Unable to connect to the backend.')
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <h1>SentinelRAG</h1>
            <span>Reliable RAG System</span>
          </div>
        </div>

        <nav className="nav">
          <button
            className={page === 'overview' ? 'nav-item active' : 'nav-item'}
            onClick={() => setPage('overview')}
          >
            <span>◉</span>
            Overview
          </button>

          <button
            className={page === 'documents' ? 'nav-item active' : 'nav-item'}
            onClick={() => setPage('documents')}
          >
            <span>▣</span>
            Documents
          </button>

          <button
            className={page === 'query' ? 'nav-item active' : 'nav-item'}
            onClick={() => setPage('query')}
          >
            <span>⌕</span>
            Query
          </button>

          <button
            className={page === 'audit' ? 'nav-item active' : 'nav-item'}
            onClick={() => setPage('audit')}
          >
            <span>◫</span>
            Audit Logs
          </button>
        </nav>

        <div className="sidebar-status">
          <div className="status-dot"></div>
          <div>
            <strong>System Status</strong>
            <span>
              {backendOnline
                ? 'All systems operational'
                : 'Backend offline'}
            </span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        {error && <div className="error-banner">{error}</div>}

        {page === 'overview' && (
          <OverviewPage summary={summary} />
        )}

        {page === 'documents' && (
          <DocumentsPage />
        )}

        {page === 'query' && (
          <QueryPage />
        )}

        {page === 'audit' && (
          <AuditLogsPage />
        )}
      </main>
    </div>
  )
}

function OverviewPage({ summary }) {
  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">SYSTEM OVERVIEW</p>
          <h2>RAG Reliability Dashboard</h2>
          <p>
            Monitor retrieval, confidence, verification and decision outcomes.
          </p>
        </div>
      </header>

      <section className="stats-grid">
        <StatCard
          label="Total Queries"
          value={summary?.total_queries ?? 0}
        />

        <StatCard
          label="Average Confidence"
          value={
            summary
              ? `${(summary.average_confidence * 100).toFixed(2)}%`
              : '0%'
          }
        />

        <StatCard
          label="Average Latency"
          value={
            summary
              ? `${(summary.average_latency_ms / 1000).toFixed(2)}s`
              : '0s'
          }
        />

        <StatCard
          label="Accepted"
          value={summary?.accepted_count ?? 0}
        />
      </section>

      <section className="dashboard-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">PIPELINE</p>
              <h3>RAG Processing Flow</h3>
            </div>
          </div>

          <div className="pipeline">
            <PipelineStep
              number="01"
              title="Retrieve"
              text="Hybrid vector + keyword search"
            />

            <PipelineStep
              number="02"
              title="Rerank"
              text="Cross-encoder relevance scoring"
            />

            <PipelineStep
              number="03"
              title="Generate"
              text="Local LLM answer generation"
            />

            <PipelineStep
              number="04"
              title="Verify"
              text="Semantic and NLI verification"
            />

            <PipelineStep
              number="05"
              title="Decide"
              text="Accept, retry or human review"
            />
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">DECISION ENGINE</p>
              <h3>System Behaviour</h3>
            </div>
          </div>

          <div className="decision-list">
            <div>
              <span className="decision-badge accept">ACCEPT</span>
              <p>High-confidence answers are returned directly.</p>
            </div>

            <div>
              <span className="decision-badge retry">RETRY</span>
              <p>Low-confidence answers trigger expanded retrieval.</p>
            </div>

            <div>
              <span className="decision-badge review">REVIEW</span>
              <p>Uncertain answers are flagged for human review.</p>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}

function DocumentsPage() {
  const [file, setFile] = useState(null)
  const [documents, setDocuments] = useState([])
  const [uploading, setUploading] = useState(false)
  const [deletingId, setDeletingId] = useState(null)
  const [loadingDocuments, setLoadingDocuments] = useState(true)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    loadDocuments()
  }, [])

  async function loadDocuments() {
    try {
      setLoadingDocuments(true)

      const data = await getDocuments()
      setDocuments(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoadingDocuments(false)
    }
  }

  async function handleUpload() {
    if (!file) return

    setUploading(true)
    setError('')
    setResult(null)

    try {
      const data = await uploadDocument(file)

      setResult(data)
      setFile(null)

      await loadDocuments()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(documentId, filename) {
    const confirmed = window.confirm(
      `Delete "${filename}"?\n\nThis will remove the document and its chunks. Audit logs will be preserved.`
    )

    if (!confirmed) return

    setDeletingId(documentId)
    setError('')

    try {
      await deleteDocument(documentId)
      await loadDocuments()
    } catch (err) {
      setError(err.message)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">DOCUMENTS</p>
          <h2>Knowledge Base</h2>
          <p>
            Upload PDF or TXT documents for retrieval and question answering.
          </p>
        </div>
      </header>

      <section className="panel upload-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">UPLOAD</p>
            <h3>Add a Document</h3>
          </div>
        </div>

        <div className="upload-area">
          <input
            type="file"
            accept=".pdf,.txt"
            onChange={(event) => {
              setFile(event.target.files?.[0] || null)
              setResult(null)
              setError('')
            }}
          />

          {file && (
            <div className="selected-file">
              <strong>{file.name}</strong>
              <span>{(file.size / 1024).toFixed(1)} KB</span>
            </div>
          )}

          <button
            className="primary-button"
            onClick={handleUpload}
            disabled={!file || uploading}
          >
            {uploading ? 'Processing...' : 'Upload Document'}
          </button>
        </div>

        {result && (
          <div className="success-card">
            <strong>Document uploaded successfully</strong>
            <span>{result.filename}</span>
            <span>{result.chunks} chunks processed</span>
          </div>
        )}

        {error && (
          <div className="error-card">
            {error}
          </div>
        )}
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">STORED DOCUMENTS</p>
            <h3>Knowledge Base Files</h3>
          </div>

          <span>
            {documents.length} document{documents.length !== 1 ? 's' : ''}
          </span>
        </div>

        {loadingDocuments ? (
          <p>Loading documents...</p>
        ) : documents.length === 0 ? (
          <p>No documents uploaded yet.</p>
        ) : (
          <div>
            {documents.map((document) => (
              <div
                key={document.document_id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '20px',
                  padding: '18px 0',
                  borderBottom: '1px solid rgba(0, 0, 0, 0.08)',
                }}
              >
                <div>
                  <strong>{document.filename}</strong>

                  <div
                    style={{
                      marginTop: '5px',
                      fontSize: '13px',
                      opacity: 0.6,
                    }}
                  >
                    Uploaded{' '}
                    {document.created_at
                      ? new Date(document.created_at).toLocaleString()
                      : 'Unknown'}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() =>
                    handleDelete(
                      document.document_id,
                      document.filename
                    )
                  }
                  disabled={deletingId === document.document_id}
                  style={{
                    border: '1px solid #dc2626',
                    background: 'transparent',
                    color: '#dc2626',
                    borderRadius: '8px',
                    padding: '8px 14px',
                    cursor:
                      deletingId === document.document_id
                        ? 'not-allowed'
                        : 'pointer',
                    opacity:
                      deletingId === document.document_id ? 0.6 : 1,
                  }}
                >
                  {deletingId === document.document_id
                    ? 'Deleting...'
                    : 'Delete'}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  )
}

function QueryPage() {
  const [documents, setDocuments] = useState([])
  const [documentId, setDocumentId] = useState('')
  const [query, setQuery] = useState('')
  const [loadingDocuments, setLoadingDocuments] = useState(true)
  const [querying, setQuerying] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    loadDocuments()
  }, [])

  async function loadDocuments() {
    try {
      setLoadingDocuments(true)
      setError('')

      const data = await getDocuments()

      setDocuments(data)

      if (data.length > 0) {
        setDocumentId((current) =>
          data.some((doc) => doc.document_id === current)
            ? current
            : data[0].document_id
        )
      } else {
        setDocumentId('')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoadingDocuments(false)
    }
  }

  async function handleQuery(event) {
    event.preventDefault()

    if (!documentId || !query.trim()) {
      setError('Select a document and enter a question.')
      return
    }

    setQuerying(true)
    setError('')
    setResult(null)

    try {
      const data = await queryDocument(
        query.trim(),
        documentId,
        5
      )

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setQuerying(false)
    }
  }

  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">QUERY</p>
          <h2>Ask Your Documents</h2>
          <p>
            Query the knowledge base and inspect the reliability decision.
          </p>
        </div>
      </header>

      <section className="panel query-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">QUESTION ANSWERING</p>
            <h3>Run a RAG Query</h3>
          </div>
        </div>

        <form onSubmit={handleQuery}>
          <label className="form-label">Document</label>

          <select
            className="document-select"
            value={documentId}
            onChange={(event) => setDocumentId(event.target.value)}
            disabled={loadingDocuments || querying}
          >
            {loadingDocuments && (
              <option value="">Loading documents...</option>
            )}

            {!loadingDocuments && documents.length === 0 && (
              <option value="">No documents available</option>
            )}

            {documents.map((document) => (
              <option
                key={document.document_id}
                value={document.document_id}
              >
                {document.filename}
              </option>
            ))}
          </select>

          <label className="form-label">Question</label>

          <textarea
            className="query-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask a question about the selected document..."
            rows={5}
            disabled={querying}
          />

          <button
            className="primary-button query-button"
            type="submit"
            disabled={querying || !documentId || !query.trim()}
          >
            {querying ? 'Processing Query...' : 'Ask Question'}
          </button>
        </form>

        {error && (
          <div className="error-card">
            {error}
          </div>
        )}
      </section>

      {result && <QueryResult result={result} />}
    </>
  )
}

function QueryResult({ result }) {
  const decisionClass =
    result.decision === 'accept'
      ? 'accept'
      : result.decision === 'human_review'
        ? 'review'
        : 'retry'

  return (
    <section className="query-results">
      <div className="panel answer-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">GENERATED ANSWER</p>
            <h3>Answer</h3>
          </div>

          <span className={`decision-badge ${decisionClass}`}>
            {result.decision.replaceAll('_', ' ').toUpperCase()}
          </span>
        </div>

        <div className="answer-content">
          {result.answer}
        </div>

        <div className="result-stats">
          <ResultStat
            label="Confidence"
            value={`${(result.confidence * 100).toFixed(2)}%`}
          />

          <ResultStat
            label="Latency"
            value={`${(result.latency_ms / 1000).toFixed(2)}s`}
          />

          <ResultStat
            label="Retry"
            value={result.retry ? 'Yes' : 'No'}
          />

          <ResultStat
            label="Human Review"
            value={result.human_review ? 'Required' : 'No'}
          />
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">RETRIEVAL</p>
            <h3>Retrieved Context</h3>
          </div>
        </div>

        <div className="retrieved-list">
          {result.results?.length ? (
            result.results.map((item, index) => (
              <div
                className="retrieved-item"
                key={index}
              >
                <div className="retrieved-header">
                  <strong>
                    Chunk {item.chunk_index ?? index + 1}
                  </strong>

                  {item.rerank_score !== undefined && (
                    <span>
                      Rerank:{' '}
                      {Number(item.rerank_score).toFixed(3)}
                    </span>
                  )}
                </div>

                <p>{item.content}</p>
              </div>
            ))
          ) : (
            <p>No retrieval results returned.</p>
          )}
        </div>
      </div>
    </section>
  )
}

function AuditLogsPage() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    loadLogs()
  }, [])

  async function loadLogs() {
    try {
      setLoading(true)
      setError('')

      const response = await fetch(`${API_BASE_URL}/audit/logs`)

      if (!response.ok) {
        throw new Error('Failed to fetch audit logs')
      }

      const data = await response.json()

      setLogs(Array.isArray(data) ? data : data.logs || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">AUDIT</p>
          <h2>Audit Logs</h2>
          <p>
            Inspect query decisions, confidence, retries and latency.
          </p>
        </div>
      </header>

      <section className="panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">QUERY HISTORY</p>
            <h3>System Activity</h3>
          </div>

          <button
            className="primary-button"
            onClick={loadLogs}
            disabled={loading}
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>

        {error && (
          <div className="error-card">
            {error}
          </div>
        )}

        {loading ? (
          <p>Loading audit logs...</p>
        ) : logs.length === 0 ? (
          <p>No audit logs available.</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table
              style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: '14px',
              }}
            >
              <thead>
                <tr>
                  <th style={tableHeaderStyle}>Timestamp</th>
                  <th style={tableHeaderStyle}>Query</th>
                  <th style={tableHeaderStyle}>Confidence</th>
                  <th style={tableHeaderStyle}>Decision</th>
                  <th style={tableHeaderStyle}>Retry</th>
                  <th style={tableHeaderStyle}>Latency</th>
                </tr>
              </thead>

              <tbody>
                {logs.map((log) => (
                  <tr key={log.id}>
                    <td style={tableCellStyle}>
                      {log.created_at
                        ? new Date(log.created_at).toLocaleString()
                        : '-'}
                    </td>

                    <td style={tableCellStyle}>
                      {log.query}
                    </td>

                    <td style={tableCellStyle}>
                      {log.confidence !== undefined
                        ? `${(log.confidence * 100).toFixed(2)}%`
                        : '-'}
                    </td>

                    <td style={tableCellStyle}>
                      <span
                        className={`decision-badge ${
                          log.decision === 'accept'
                            ? 'accept'
                            : log.decision === 'human_review'
                              ? 'review'
                              : 'retry'
                        }`}
                      >
                        {log.decision
                          ? log.decision
                              .replaceAll('_', ' ')
                              .toUpperCase()
                          : '-'}
                      </span>
                    </td>

                    <td style={tableCellStyle}>
                      {log.retry ? 'Yes' : 'No'}
                    </td>

                    <td style={tableCellStyle}>
                      {log.latency_ms !== undefined
                        ? `${(log.latency_ms / 1000).toFixed(2)}s`
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  )
}

const tableHeaderStyle = {
  textAlign: 'left',
  padding: '12px',
  borderBottom: '1px solid rgba(0, 0, 0, 0.1)',
}

const tableCellStyle = {
  padding: '12px',
  borderBottom: '1px solid rgba(0, 0, 0, 0.06)',
  verticalAlign: 'top',
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function PipelineStep({ number, title, text }) {
  return (
    <div className="pipeline-step">
      <span className="pipeline-number">{number}</span>

      <div>
        <strong>{title}</strong>
        <span>{text}</span>
      </div>
    </div>
  )
}

function ResultStat({ label, value }) {
  return (
    <div className="result-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default App