import axios from 'axios'
import type {
  QueryRequest,
  QueryResponse,
  GraphStats,
  NeighborResponse,
  ListGraphsResponse,
  SystemSettings,
  DeleteResponse,
} from '@/types/api'

const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

export const api = {
  // Query the knowledge graph
  query: async (request: QueryRequest): Promise<QueryResponse> => {
    const { graph_name, ...body } = request
    const params = graph_name ? { graph_name } : {}
    const response = await apiClient.post<QueryResponse>('/query', body, { params })
    return response.data
  },

  // Get graph statistics
  getGraphStats: async (graphName: string = 'default'): Promise<GraphStats> => {
    const response = await apiClient.get<GraphStats>(`/graph/${graphName}/stats`)
    return response.data
  },

  // Get entity neighbors (for lazy loading graph expansion)
  getNeighbors: async (
    entityId: string,
    graphName: string = 'default',
    limit: number = 20
  ): Promise<NeighborResponse> => {
    const response = await apiClient.get<NeighborResponse>(
      `/graph/${graphName}/neighbors/${entityId}`,
      { params: { limit } }
    )
    return response.data
  },

  // Health check
  health: async (): Promise<{ status: string }> => {
    const response = await apiClient.get<{ status: string }>('/health')
    return response.data
  },

  // List available graphs (datasets)
  listGraphs: async (): Promise<ListGraphsResponse> => {
    const response = await apiClient.get<ListGraphsResponse>('/graphs')
    return response.data
  },

  // Get system settings
  getSettings: async (): Promise<SystemSettings> => {
    const response = await apiClient.get<SystemSettings>('/settings')
    return response.data
  },

  // Delete a graph (knowledge base)
  deleteGraph: async (graphName: string): Promise<DeleteResponse> => {
    const response = await apiClient.delete<DeleteResponse>(`/graph/${graphName}`)
    return response.data
  },
}

// Import types
export interface ImportRequest {
  sources: string[]
  chunk_documents: boolean
  chunk_size: number
  chunk_overlap: number
  extract_triplets: boolean
  graph_name?: string
}

export interface ImportResponse {
  success: boolean
  num_sources: number
  num_documents: number
  num_chunks: number
  num_entities: number
  num_relations: number
  errors: string[]
}

export interface UploadOptions {
  chunkDocuments?: boolean
  chunkSize?: number
  extractTriplets?: boolean
  graphName?: string
}

// Import documents from URLs or file paths
export async function importDocuments(request: ImportRequest): Promise<ImportResponse> {
  const response = await apiClient.post<ImportResponse>('/import', request)
  return response.data
}

// Upload files directly
export async function uploadFiles(
  files: File[],
  options: UploadOptions = {}
): Promise<ImportResponse> {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))

  if (options.chunkDocuments !== undefined) {
    formData.append('chunk_documents', String(options.chunkDocuments))
  }
  if (options.chunkSize !== undefined) {
    formData.append('chunk_size', String(options.chunkSize))
  }
  if (options.extractTriplets !== undefined) {
    formData.append('extract_triplets', String(options.extractTriplets))
  }
  if (options.graphName) {
    formData.append('graph_name', options.graphName)
  }

  const response = await apiClient.post<ImportResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export default api
