import axios from 'axios'
import type {
  QueryRequest,
  QueryResponse,
  GraphStats,
  NeighborResponse,
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
    const response = await apiClient.post<QueryResponse>('/query', request)
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
}

export default api
