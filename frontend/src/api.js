const API_BASE_URL = 'http://localhost:8000'

export async function getAuditSummary() {
  const response = await fetch(`${API_BASE_URL}/audit/summary`)
  if (!response.ok) throw new Error('Failed to fetch audit summary')
  return response.json()
}

export async function getAuditLogs() {
  const response = await fetch(`${API_BASE_URL}/audit/logs`)
  if (!response.ok) throw new Error('Failed to fetch audit logs')
  return response.json()
}

export async function getHealth() {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw new Error('Backend unavailable')
  return response.json()
}

export async function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: 'POST',
    body: formData,
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.detail || 'Document upload failed')
  }

  return data
}

export async function getDocuments() {
  const response = await fetch(`${API_BASE_URL}/documents`)

  if (!response.ok) {
    throw new Error('Failed to fetch documents')
  }

  return response.json()
}
export async function queryDocument(query, documentId, limit = 5) {
  const response = await fetch(`${API_BASE_URL}/documents/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      document_id: documentId,
      limit,
    }),
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.detail || 'Query failed')
  }

  return data
}
export async function deleteDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
    method: 'DELETE',
  })

  const data = await response.json()

  if (!response.ok) {
    throw new Error(data.detail || 'Document deletion failed')
  }

  return data
}