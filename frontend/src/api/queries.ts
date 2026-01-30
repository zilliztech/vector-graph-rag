import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import api from './client'
import type { QueryRequest } from '@/types/api'

// Query keys
export const queryKeys = {
  graphStats: (graphName: string) => ['graphStats', graphName] as const,
  neighbors: (entityId: string, graphName: string) =>
    ['neighbors', entityId, graphName] as const,
  health: ['health'] as const,
}

// Search mutation
export const useSearch = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (request: QueryRequest) => api.query(request),
    onSuccess: (data) => {
      // Prefetch neighbors for top entities
      const topEntities = data.subgraph.entities.slice(0, 5)
      topEntities.forEach((entity) => {
        queryClient.prefetchQuery({
          queryKey: queryKeys.neighbors(entity.id, 'default'),
          queryFn: () => api.getNeighbors(entity.id),
          staleTime: 5 * 60 * 1000, // 5 minutes
        })
      })
    },
  })
}

// Graph stats query
export const useGraphStats = (graphName: string = 'default') => {
  return useQuery({
    queryKey: queryKeys.graphStats(graphName),
    queryFn: () => api.getGraphStats(graphName),
    staleTime: 60 * 1000, // 1 minute
    retry: 1,
  })
}

// Neighbors query
export const useNeighbors = (
  entityId: string,
  graphName: string = 'default',
  enabled: boolean = true
) => {
  return useQuery({
    queryKey: queryKeys.neighbors(entityId, graphName),
    queryFn: () => api.getNeighbors(entityId, graphName),
    enabled: enabled && !!entityId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  })
}

// Health check query
export const useHealth = () => {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => api.health(),
    staleTime: 30 * 1000, // 30 seconds
    retry: 2,
  })
}
